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
# With attention dropout disabled (see _disable_attention_dropout) decode is
# deterministic, so ORT-CPU should match torch to float32 noise.
MEL_ATOL = 1e-3


def _disable_attention_dropout(module: torch.nn.Module) -> int:
    """Zero attention dropout for inference.

    Upstream bug: kanade_tokenizer's non-flash attention path
    (module/transformer.py) calls
    `F.scaled_dot_product_attention(..., dropout_p=self.dropout)` WITHOUT the
    `if self.training` guard the flash path has. On CPU (no FlashAttention) this
    applies dropout at inference — making decode nondeterministic and slightly
    degraded. Setting `.dropout = 0` on the Attention modules fixes both.
    """
    patched = 0
    for m in module.modules():
        if type(m).__name__ == "Attention" and hasattr(m, "dropout"):
            m.dropout = 0.0
            patched += 1
    return patched


class DecoderWrapper(torch.nn.Module):
    """Decode-only path: content-token indices + speaker emb -> mel.

    Holds ONLY the decode submodules (quantizer + mel prenet/upsample/decoder/
    postnet) — NOT the WavLM SSL encoder — so the export stays small. Replicates
    KanadeModel.decode -> forward_mel; the only thing the full model needs the
    SSL encoder for on this path is integer length math, and that reduces to
    mel_length = downsample_factor * seq_len = 2 * seq_len (verified exact across
    lengths).
    """

    def __init__(self, kanade: torch.nn.Module) -> None:
        super().__init__()
        self.local_quantizer = kanade.local_quantizer
        self.mel_prenet = kanade.mel_prenet
        self.mel_conv_upsample = kanade.mel_conv_upsample
        self.mel_decoder = kanade.mel_decoder
        self.mel_postnet = kanade.mel_postnet
        self.interp_mode = kanade.config.mel_interpolation_mode
        self.upsample = kanade.downsample_factor

    def forward(
        self, content_token_indices: torch.Tensor, global_embedding: torch.Tensor
    ) -> torch.Tensor:
        # (seq_len,), (128,) -> (n_mels, T)
        content = self.local_quantizer.decode(content_token_indices)  # (seq_len, dim)
        content = content.unsqueeze(0)  # (1, seq_len, dim)
        mel_length = content_token_indices.shape[0] * self.upsample

        local_latent = self.mel_prenet(content)
        if self.mel_conv_upsample is not None:
            local_latent = self.mel_conv_upsample(local_latent.transpose(1, 2)).transpose(1, 2)
        local_latent = torch.nn.functional.interpolate(
            local_latent.transpose(1, 2), size=mel_length, mode=self.interp_mode
        ).transpose(1, 2)

        condition = global_embedding.unsqueeze(0).unsqueeze(1)  # (1, 1, 128)
        mel = self.mel_decoder(local_latent, condition=condition)
        mel = self.mel_postnet(mel.transpose(1, 2))
        return mel.squeeze(0)  # (n_mels, T)


def main() -> None:
    from kanade_tokenizer import KanadeModel

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    kanade = KanadeModel.from_pretrained(REPO).eval()
    n_patched = _disable_attention_dropout(kanade)
    print(f"disabled attention dropout on {n_patched} Attention module(s)")
    wrapper = DecoderWrapper(kanade).eval()

    # Random embedding is fine here: we validate the GRAPH, not audio quality.
    tokens = torch.randint(0, 100, (50,), dtype=torch.long)
    emb = torch.randn(SPEAKER_DIM)

    # Reference = the full model's decode(); also confirm the slim wrapper
    # reproduces it (so dropping the SSL encoder changed nothing on this path).
    with torch.no_grad():
        reference_mel = kanade.decode(
            global_embedding=emb, content_token_indices=tokens
        ).cpu().numpy()
        slim_mel = wrapper(tokens, emb).cpu().numpy()
    slim_diff = float(np.abs(slim_mel - reference_mel).max())
    print(f"slim-vs-full decode max|diff|={slim_diff:.4g}")
    if slim_diff > 1e-4:
        raise SystemExit("slim decoder wrapper diverged from KanadeModel.decode")

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
