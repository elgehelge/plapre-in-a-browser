"""Phase 0: export the Kanade decoder to ONNX.

content-token indices (+ 128-dim global speaker embedding) -> mel spectrogram.

Only the DECODE path is needed (no encoder / WavLM frontend), since built-in
speakers ship as precomputed 128-dim embeddings.

Traced from plapre/inference.py:_tokens_to_audio:
    mel = self.kanade.decode(content_token_indices=tokens_tensor,  # (seq_len,)
                             global_embedding=speaker_emb)         # (128,)
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).parent.parent / "web" / "public" / "models" / "kanade_decoder.onnx"
OPSET = 17
SPEAKER_DIM = 128


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from kanade_tokenizer import KanadeModel

    kanade = KanadeModel.from_pretrained("frothywater/kanade-25hz-clean").eval()

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, content_token_indices, global_embedding):
            # (seq_len,), (128,) -> (n_mels, T)
            return self.k.decode(
                content_token_indices=content_token_indices,
                global_embedding=global_embedding,
            )

    wrapper = DecoderWrapper(kanade).eval()

    dummy_tokens = torch.zeros(100, dtype=torch.long)
    dummy_emb = torch.randn(SPEAKER_DIM)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_tokens, dummy_emb),
            str(OUT),
            input_names=["content_token_indices", "global_embedding"],
            output_names=["mel"],
            dynamic_axes={"content_token_indices": {0: "seq_len"}, "mel": {1: "frames"}},
            opset_version=OPSET,
            do_constant_folding=True,
        )

    print(f"Exported Kanade decoder -> {OUT}")
    print("TODO: validate output mel matches golden/mel.npy within tolerance.")


if __name__ == "__main__":
    main()
