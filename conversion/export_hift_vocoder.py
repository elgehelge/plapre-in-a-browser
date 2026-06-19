"""Phase 0 (gate): export the HiFT vocoder to ONNX.

mel-spectrogram -> 24 kHz waveform.

================================ GATE FINDING ================================
This is the make-or-break export, and as of 2026-06 it DOES NOT export as-is.

HiFTGenerator (kanade_tokenizer/module/hift.py) runs its harmonic source signal
and final synthesis through torch.stft / torch.istft on complex tensors:

    _stft:   torch.stft(..., return_complex=True) -> torch.view_as_real
    _istft:  torch.complex(real, img) -> torch.istft(...)

Both the legacy and the TorchDynamo ONNX exporters reject this:
  - legacy (dynamo=False): aten::istft / aten::complex have no ONNX symbolic.
  - dynamo=True:           "Failed to decompose the FX graph" (no decomposition
                           for aten.istft).

Switching to the `kanade-25hz` (Vocos) variant does NOT help: Vocos's ISTFTHead
uses torch.istft too. The blocker is the (i)STFT/complex math, not HiFT itself.

============================== REMEDIATION PLAN =============================
The transforms are TINY: istft_n_fft=16, istft_hop_len=4. So replace the
complex (i)STFT with real-valued, ONNX/ORT-Web-friendly ops on a HiFT subclass
that keeps the trained weights:

  STFT(x):   frame x (Unfold, win=16, hop=4) -> apply Hann window
             -> matmul with precomputed real DFT cos/sin basis (16x9 each)
             -> (real, imag)                                  # no complex dtype
  ISTFT:     real,imag from magnitude*cos(phase), magnitude*sin(phase)
             -> matmul with inverse DFT basis -> window -> overlap-add
                via ConvTranspose1d -> divide by window-overlap envelope
  source:    SourceModuleHnNSF2 seeds sine phase with torch.rand (RNG). Make it
             deterministic (fixed/zero phase) so the graph has no RandomUniform
             and outputs are reproducible.

Validate the patched vocoder in-process against the stock `torch.istft` path
(same mel in, compare waveforms within tolerance) before exporting, then re-run
the dynamo export and check ORT-CPU parity like the decoder script does.

NOTE: not yet implemented. This script currently documents the blocker and
fails loudly rather than emitting a broken graph.
"""

from __future__ import annotations

import sys
from pathlib import Path

OUT = Path(__file__).parent.parent / "web" / "public" / "models" / "hift_vocoder.onnx"


def main() -> None:
    sys.stderr.write(
        "export_hift_vocoder: BLOCKED on torch.stft/istft/complex (see module "
        "docstring). Implement the real-valued (i)STFT HiFT subclass before "
        "exporting; refusing to emit a broken graph.\n"
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
