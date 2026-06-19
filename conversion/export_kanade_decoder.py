"""Phase 0: export the Kanade decoder to ONNX.

content-token indices (+ 128-dim global speaker embedding) -> mel spectrogram.

Only the DECODE path is needed (no encoder / WavLM frontend), since built-in
speakers ship as precomputed 128-dim embeddings.

    mel = kanade.decode(global_embedding=emb,            # (128,)
                        content_token_indices=tokens)    # (seq_len,) -> (n_mels, T)

GATE FINDING (2026-06): the decoder's transformer uses *rotary position
embeddings implemented with complex tensors*. The legacy TorchScript exporter
(`torch.onnx.export(..., dynamo=False)`) rejects this with:

    RuntimeError: ScalarType ComplexFloat is an unexpected tensor scalar type

The TorchDynamo exporter (`dynamo=True`, requires `onnxscript`) decomposes the
complex RoPE into real-valued ops (cos/sin/mul/neg/cat) and exports cleanly.
ORT-CPU parity vs PyTorch is ~1e-2 max abs on the mel (acceptable; the spread is
dominated by the RoPE decomposition + fp32 reordering, not a correctness bug).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

REPO = "frothywater/kanade-25hz-clean"
OUT = Path(__file__).parent.parent / "web" / "public" / "models" / "kanade_decoder.onnx"
SPEAKER_DIM = 128
SEED = 1234
# Parity tolerance on the mel. Empirically ~0.011; 0.05 leaves margin for
# different ORT builds without masking a real regression.
MEL_ATOL = 5e-2


class DecoderWrapper(torch.nn.Module):
    def __init__(self, kanade: torch.nn.Module) -> None:
        super().__init__()
        self.k = kanade

    def forward(
        self, content_token_indices: torch.Tensor, global_embedding: torch.Tensor
    ) -> torch.Tensor:
        # (seq_len,), (128,) -> (n_mels, T)
        return self.k.decode(
            content_token_indices=content_token_indices,
            global_embedding=global_embedding,
        )


def main() -> None:
    from kanade_tokenizer import KanadeModel

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    kanade = KanadeModel.from_pretrained(REPO).eval()
    wrapper = DecoderWrapper(kanade).eval()

    # Random embedding is fine here: we validate the GRAPH, not audio quality.
    tokens = torch.randint(0, 100, (50,), dtype=torch.long)
    emb = torch.randn(SPEAKER_DIM)

    with torch.no_grad():
        reference_mel = wrapper(tokens, emb).cpu().numpy()

    # dynamo=True is required: the legacy exporter chokes on the complex RoPE.
    torch.onnx.export(
        wrapper,
        (tokens, emb),
        str(OUT),
        input_names=["content_token_indices", "global_embedding"],
        output_names=["mel"],
        dynamic_axes={"content_token_indices": {0: "seq_len"}, "mel": {1: "frames"}},
        dynamo=True,
    )
    print(f"Exported Kanade decoder -> {OUT}")

    _validate_cpu(tokens.numpy(), emb.numpy(), reference_mel)


def _validate_cpu(tokens: np.ndarray, emb: np.ndarray, reference_mel: np.ndarray) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    feeds = {
        "content_token_indices": tokens,
        "global_embedding": emb,
    }
    out = sess.run(["mel"], feeds)[0]
    max_diff = float(np.abs(out - reference_mel).max())
    status = "OK" if max_diff <= MEL_ATOL else "FAIL"
    print(f"ORT-CPU mel parity: shape={out.shape} max|diff|={max_diff:.4g} [{status}]")
    if max_diff > MEL_ATOL:
        raise SystemExit(f"decoder parity exceeded tolerance {MEL_ATOL}")


if __name__ == "__main__":
    main()
