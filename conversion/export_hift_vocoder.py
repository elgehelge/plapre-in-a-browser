"""Phase 0 (gate): export the HiFT vocoder to ONNX.

mel-spectrogram -> 24 kHz waveform.

This is the highest-risk export: HiFT-style vocoders use weight-normalized
convolutions and (i)STFT / complex ops that ONNX Runtime Web may not support.
Verify the exported graph runs under BOTH onnxruntime (CPU) here and
onnxruntime-web (wasm + webgpu) in the browser before trusting it.

The vocoder comes from CosyVoice 2 via kanade-tokenizer's load_vocoder()
(model.config.vocoder_name for kanade-25hz-clean).
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).parent.parent / "web" / "public" / "models" / "hift_vocoder.onnx"
OPSET = 17


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from kanade_tokenizer import KanadeModel, load_vocoder

    kanade = KanadeModel.from_pretrained("frothywater/kanade-25hz-clean").eval()
    vocoder = load_vocoder(kanade.config.vocoder_name).eval()

    # TODO: confirm the vocoder forward signature. In kanade_tokenizer the public
    # call is vocode(vocoder, mel.unsqueeze(0)). We need the underlying nn.Module
    # forward that maps (B, n_mels, T) -> (B, samples). Wrap it if vocode() does
    # pre/post-processing that must live inside the graph.
    n_mels = 80  # TODO: read from config
    dummy_mel = torch.randn(1, n_mels, 100)

    class VocoderWrapper(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, mel):  # (B, n_mels, T) -> (B, samples)
            # TODO: replace with the exact vocode() math (sans numpy/host ops).
            return self.v(mel)

    wrapper = VocoderWrapper(vocoder).eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_mel,),
            str(OUT),
            input_names=["mel"],
            output_names=["waveform"],
            dynamic_axes={"mel": {0: "batch", 2: "frames"}, "waveform": {0: "batch", 1: "samples"}},
            opset_version=OPSET,
            do_constant_folding=True,
        )

    print(f"Exported HiFT vocoder -> {OUT}")
    print("TODO: validate with onnxruntime CPU here, then onnxruntime-web (wasm + webgpu).")
    print("If STFT/complex ops are unsupported on web: try the kanade-25hz (Vocos) variant.")


if __name__ == "__main__":
    main()
