"""Phase 0 (gate): export the HiFT vocoder to ONNX.

mel-spectrogram (1, 80, T) -> 24 kHz waveform (1, samples).

This was the make-or-break export. Stock `HiFTGenerator` cannot be exported:
its source signal + synthesis run through `torch.stft`/`torch.istft` on complex
tensors (no ONNX op), and the sine source uses `torch.rand`/`torch.randn`
(nondeterministic). `hift_onnx.patch_vocoder_for_onnx` swaps in real-valued
(i)STFT (the transforms are tiny: n_fft=16, hop=4) and a deterministic source;
the real (i)STFT matches `torch.istft` to ~1e-8 (see `hift_onnx.self_test`).

Two more export gotchas, handled here:
  * `HiFTGenerator.inference` is wrapped in `@torch.inference_mode()`, whose
    tensors cannot be traced by `torch.export`. We replicate its body in the
    wrapper instead (note: `inference` does NOT transpose the mel, unlike
    `forward`).
  * The example mel must be a *normal* tensor; a mel straight out of
    `kanade.decode` is an inference-mode tensor and breaks `torch.export`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import hift_onnx

REPO = "frothywater/kanade-25hz-clean"
OUT = Path(__file__).parent.parent / "web" / "public" / "models" / "hift_vocoder.onnx"
N_MELS = 80
SEED = 1234
# Tolerance on a *realistic* (decoder-produced) mel. Random mels blow up through
# the conv_post exp() and aren't a fair conditioning test.
WAV_ATOL = 1e-3


class VocoderWrapper(torch.nn.Module):
    """HiFTGenerator.inference() body without the @inference_mode() decorator."""

    def __init__(self, vocoder: torch.nn.Module) -> None:
        super().__init__()
        self.v = vocoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:  # (1, n_mels, T) -> (1, samples)
        v = self.v
        f0 = v.f0_predictor(mel)
        source = v.f0_upsamp(f0[:, None]).transpose(1, 2)
        source, _, _ = v.m_source(source)
        source = source.transpose(1, 2)
        return v.decode(x=mel, s=source)


def _representative_mel() -> torch.Tensor:
    """A realistic, well-conditioned mel from the Kanade decoder, as a *normal*
    tensor (kanade.decode returns inference-mode tensors that break export)."""
    from kanade_tokenizer import KanadeModel

    kanade = KanadeModel.from_pretrained(REPO).eval()
    tokens = torch.randint(0, 100, (50,), dtype=torch.long)
    emb = torch.randn(128)
    with torch.no_grad():
        mel = kanade.decode(global_embedding=emb, content_token_indices=tokens)
    return torch.from_numpy(mel.unsqueeze(0).cpu().numpy())


def main() -> None:
    from kanade_tokenizer import load_vocoder

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    vocoder = hift_onnx.patch_vocoder_for_onnx(load_vocoder("hift").eval())
    wrapper = VocoderWrapper(vocoder).eval()

    mel = _representative_mel()

    with torch.no_grad():
        reference_wav = wrapper(mel).cpu().numpy()

    torch.onnx.export(
        wrapper,
        (mel,),
        str(OUT),
        input_names=["mel"],
        output_names=["wav"],
        dynamic_axes={"mel": {2: "frames"}, "wav": {1: "samples"}},
        dynamo=True,
    )
    print(f"Exported HiFT vocoder -> {OUT}")

    _validate_cpu(mel.numpy(), reference_wav)


def _validate_cpu(mel: np.ndarray, reference_wav: np.ndarray) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    out = sess.run(["wav"], {"mel": mel})[0]
    max_diff = float(np.abs(out - reference_wav).max())
    status = "OK" if max_diff <= WAV_ATOL else "FAIL"
    print(f"ORT-CPU wav parity: shape={out.shape} max|diff|={max_diff:.4g} [{status}]")
    if max_diff > WAV_ATOL:
        raise SystemExit(f"vocoder parity exceeded tolerance {WAV_ATOL}")


if __name__ == "__main__":
    main()
