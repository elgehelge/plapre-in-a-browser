"""ONNX-friendly HiFT vocoder: real-valued (i)STFT + deterministic source.

HiFTGenerator can't be exported to ONNX because it runs `torch.stft` /
`torch.istft` on complex tensors (no ONNX op) and seeds its sine source with
`torch.rand` / `torch.randn` (nondeterministic RNG). This module provides:

  * `stft_real` / `istft_real` — drop-in real-valued equivalents of the tiny
    transforms HiFT uses (n_fft=16, hop=4, Hann window, center=True, onesided),
    built from precomputed DFT cos/sin bases + ConvTranspose1d overlap-add.
  * `patch_vocoder_for_onnx` — rebinds a loaded HiFTGenerator's `_stft`/`_istft`
    to the real versions and zeroes the source-module RNG, in place.

`self_test()` checks the transforms against torch and the patched vocoder
against the stock one. Run `python hift_onnx.py` to execute it.
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F


def _dft_basis(n_fft: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward onesided DFT cos/sin bases, shape (F, n_fft), F = n_fft//2 + 1."""
    n = torch.arange(n_fft, dtype=torch.float32)
    k = torch.arange(n_fft // 2 + 1, dtype=torch.float32)
    ang = 2.0 * math.pi * k[:, None] * n[None, :] / n_fft
    return torch.cos(ang), torch.sin(ang)


def stft_real(
    x: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real-valued analogue of torch.stft(..., center=True, onesided=True).

    x: (B, L) -> (real, imag) each (B, F, T), F = n_fft//2 + 1.
    """
    cos_b, sin_b = _dft_basis(n_fft)
    cos_b = cos_b.to(x.device)
    sin_b = sin_b.to(x.device)
    pad = n_fft // 2
    x = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    frames = x.unfold(dimension=1, size=n_fft, step=hop)  # (B, T, n_fft)
    frames = frames * window  # analysis window
    real = torch.einsum("btn,fn->bft", frames, cos_b)
    imag = -torch.einsum("btn,fn->bft", frames, sin_b)
    return real, imag


def istft_real(
    real: torch.Tensor, imag: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor
) -> torch.Tensor:
    """Real-valued analogue of torch.istft(..., center=True, onesided=True).

    real, imag: (B, F, T) -> waveform (B, L).
    """
    b, fdim, t = real.shape
    n = torch.arange(n_fft, dtype=torch.float32, device=real.device)
    k = torch.arange(fdim, dtype=torch.float32, device=real.device)
    ang = 2.0 * math.pi * k[:, None] * n[None, :] / n_fft  # (F, n_fft)
    cos_a = torch.cos(ang)
    sin_a = torch.sin(ang)

    # Hermitian-symmetric inverse onesided DFT: DC and Nyquist count once, the
    # rest count twice; all divided by n_fft.
    scale = torch.full((fdim,), 2.0, device=real.device)
    scale[0] = 1.0
    if n_fft % 2 == 0:
        scale[-1] = 1.0
    scale = scale / n_fft

    real_s = real * scale[None, :, None]
    imag_s = imag * scale[None, :, None]
    frames = torch.einsum("bft,fn->bnt", real_s, cos_a) - torch.einsum(
        "bft,fn->bnt", imag_s, sin_a
    )  # (B, n_fft, T)
    frames = frames * window[None, :, None]  # synthesis window

    # Overlap-add: place sample j of frame t at output index t*hop + j.
    ola_kernel = torch.eye(n_fft, device=real.device).view(n_fft, 1, n_fft)
    signal = F.conv_transpose1d(frames, ola_kernel, stride=hop).squeeze(1)  # (B, L_full)

    # Window-overlap normalization envelope (NOLA).
    win_sq = (window**2).view(1, n_fft, 1).expand(1, n_fft, t)
    env = F.conv_transpose1d(win_sq, ola_kernel, stride=hop).squeeze(1)  # (1, L_full)
    signal = signal / (env + 1e-8)

    pad = n_fft // 2  # undo center padding
    return signal[:, pad:-pad]


def patch_vocoder_for_onnx(vocoder: torch.nn.Module) -> torch.nn.Module:
    """Make a loaded HiFTGenerator exportable: real (i)STFT + zeroed source RNG."""
    n_fft = vocoder.istft_n_fft
    hop = vocoder.istft_hop_len
    window = vocoder.stft_window

    def _stft(self, x):  # noqa: ANN001
        return stft_real(x, n_fft, hop, window.to(x.device))

    def _istft(self, magnitude, phase):  # noqa: ANN001
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        return istft_real(real, imag, n_fft, hop, window.to(magnitude.device))

    vocoder._stft = types.MethodType(_stft, vocoder)
    vocoder._istft = types.MethodType(_istft, vocoder)

    # Make the sine source deterministic: drop random phase init + Gaussian noise.
    sin_gen = vocoder.m_source.l_sin_gen
    sin_gen._onnx_deterministic = True

    def _f02sine(self, f0_values):  # noqa: ANN001 — mirrors SineGen2._f02sine, RNG removed
        rad_values = (f0_values / self.sampling_rate) % 1
        # original: rand_ini = torch.rand(...); rand_ini[:, 0] = 0 -> use zeros.
        rad_values = torch.nn.functional.interpolate(
            rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear"
        ).transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * math.pi
        phase = torch.nn.functional.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        return torch.sin(phase)

    def _forward(self, f0):  # noqa: ANN001 — mirrors SineGen2.forward, noise removed
        fn = torch.multiply(
            f0, torch.arange(1, self.harmonic_num + 2, dtype=f0.dtype, device=f0.device)
        )
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        sine_waves = sine_waves * uv  # + noise (dropped for determinism)
        return sine_waves, uv, torch.zeros_like(sine_waves)

    sin_gen._f02sine = types.MethodType(_f02sine, sin_gen)
    sin_gen.forward = types.MethodType(_forward, sin_gen)
    return vocoder


def self_test() -> None:
    torch.manual_seed(0)
    n_fft, hop = 16, 4
    window = torch.from_numpy(
        __import__("scipy.signal", fromlist=["get_window"]).get_window(
            "hann", n_fft, fftbins=True
        ).astype("float32")
    )

    # ---- transform parity vs torch ----
    x = torch.randn(2, 4000)
    spec = torch.stft(x, n_fft, hop, n_fft, window=window, return_complex=True)
    r_ref, i_ref = spec.real, spec.imag
    r, i = stft_real(x, n_fft, hop, window)
    print("stft real max|diff|:", float((r - r_ref).abs().max()))
    print("stft imag max|diff|:", float((i - i_ref).abs().max()))

    inv_ref = torch.istft(torch.complex(r_ref, i_ref), n_fft, hop, n_fft, window=window)
    inv = istft_real(r_ref, i_ref, n_fft, hop, window)
    m = min(inv.shape[-1], inv_ref.shape[-1])
    print("istft shapes:", tuple(inv.shape), tuple(inv_ref.shape))
    print("istft max|diff|:", float((inv[:, :m] - inv_ref[:, :m]).abs().max()))


if __name__ == "__main__":
    self_test()
