"""Phase 1: export the Plapre LM (SmolLM2 / LLaMA) to ONNX for the browser.

The browser decode loop (web/src/pipeline/lm.ts) is already proven against a toy
LM that follows this exact I/O contract (see conversion/gen_toy_lm.py and the
Phase 1 browser gate). This script produces the SAME contract from the real
weights:

  inputs : input_ids            int64   [seq]
           prefix_embeds         float32 [k, hidden]   (k=1 prefill, k=0 decode)
           past_key_values.{i}.{key,value} float32 [1, kvHeads, past, headDim]
  outputs: logits               float32 [1, k+seq, vocab]
           present.{i}.{key,value}         float32 [1, kvHeads, past+k+seq, headDim]

Why a custom wrapper (not plain `optimum-cli export`): Plapre conditions the LM
by prepending a speaker hidden vector as the first input embedding, so we must
run on `inputs_embeds`, not token ids. The wrapper embeds the ids internally and
concatenates an optional `prefix_embeds` row, keeping the (large) embedding
matrix inside the graph and matching the loop's prefill/decode calls exactly.

Outputs (web/public/models/lm/):
  model.onnx (+ .onnx.data sidecar if >2 GB)  and  meta.json
  {numLayers, kvHeads, headDim, hidden}

STATUS: written to the validated contract but UNVALIDATED end-to-end because the
weights are gated (see conversion/_gated.py). Run it once authenticated. The
runtime side is already proven by the toy gate, so failures here are export-shape
issues, not loop logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from _gated import CHECKPOINT, ensure_access

OUT_DIR = Path(__file__).parent.parent / "web" / "public" / "models" / "lm"


class LmExportWrapper(nn.Module):
    """Embeds ids, prepends an optional prefix-embedding row, runs the decoder
    with a KV cache, and returns logits + flattened present cache."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model  # *ForCausalLM
        self.decoder = model.model  # the LLaMA decoder stack
        self.embed = model.get_input_embeddings()
        self.lm_head = model.get_output_embeddings()

    def forward(self, input_ids, prefix_embeds, *past_flat):  # noqa: ANN001
        from transformers.cache_utils import DynamicCache

        tok = self.embed(input_ids)  # [seq, H]
        inputs_embeds = torch.cat([prefix_embeds, tok], dim=0).unsqueeze(0)  # [1, S, H]

        legacy = tuple(
            (past_flat[2 * i], past_flat[2 * i + 1]) for i in range(len(past_flat) // 2)
        )
        past_len = legacy[0][0].shape[2] if legacy else 0
        cache = DynamicCache.from_legacy_cache(legacy) if past_len > 0 else None

        seq_total = inputs_embeds.shape[1]
        position_ids = torch.arange(past_len, past_len + seq_total).unsqueeze(0)

        out = self.decoder(
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            position_ids=position_ids,
            use_cache=True,
        )
        logits = self.lm_head(out.last_hidden_state)  # [1, S, vocab]

        present = out.past_key_values
        present_legacy = present.to_legacy_cache() if hasattr(present, "to_legacy_cache") else present
        flat: list[torch.Tensor] = []
        for k, v in present_legacy:
            flat += [k, v]
        return (logits, *flat)


def _names(prefix: str, n: int) -> list[str]:
    out: list[str] = []
    for i in range(n):
        out += [f"{prefix}.{i}.key", f"{prefix}.{i}.value"]
    return out


def main() -> None:
    ensure_access()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    # eager attention exports far more cleanly than SDPA/flash for our wrapper.
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    cfg = model.config
    layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", hidden // cfg.num_attention_heads)

    wrapper = LmExportWrapper(model).eval()

    seq, k = 4, 1
    example = (
        torch.randint(0, cfg.vocab_size, (seq,), dtype=torch.long),
        torch.randn(k, hidden),
        *[torch.zeros(1, kv_heads, 0, head_dim) for _ in range(2 * layers)],
    )
    dynamic = {"input_ids": {0: "seq"}, "prefix_embeds": {0: "k"}, "logits": {1: "total"}}
    for nme in _names("past_key_values", layers):
        dynamic[nme] = {2: "past"}
    for nme in _names("present", layers):
        dynamic[nme] = {2: "total"}

    out_path = OUT_DIR / "model.onnx"
    torch.onnx.export(
        wrapper,
        example,
        str(out_path),
        input_names=["input_ids", "prefix_embeds", *_names("past_key_values", layers)],
        output_names=["logits", *_names("present", layers)],
        dynamic_axes=dynamic,
        opset_version=17,
        dynamo=False,
    )

    (OUT_DIR / "meta.json").write_text(
        json.dumps({"numLayers": layers, "kvHeads": kv_heads, "headDim": head_dim, "hidden": hidden})
    )
    print(f"exported LM -> {out_path}")
    print(f"meta: layers={layers} kvHeads={kv_heads} headDim={head_dim} hidden={hidden}")
    print("NOTE: copy tokenizer.json from the repo via fetch_tokenizer.py; then run smoke_reference.py.")


if __name__ == "__main__":
    main()
