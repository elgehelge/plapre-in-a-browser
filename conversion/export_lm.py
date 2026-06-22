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

STATUS: VALIDATED. Exported from the gated weights (transformers 5.x DynamicCache)
and checked two ways: a single-step KV-cache parity (this script, ORT vs torch,
max|diff|~1e-5) and a full 172-token greedy decode that reproduces the torch
golden bit-for-bit (conversion/validate_lm_golden.py). meta: layers=30, kvHeads=3,
headDim=64, hidden=576; the 521 MB graph stays inline (no external-data sidecar).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from _gated import checkpoint_for, ensure_access, parse_model, variant_dir


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

        # Seed a cache with the incoming past UNCONDITIONALLY (even a 0-length
        # past), so the past_key_values inputs are wired into the traced graph as
        # `cat(past, new)`. Positions/causal mask are left to the decoder, which
        # derives `cache_position` from the (dynamic) cache length — never baking
        # the past length as a constant the way an explicit position_ids would.
        cache = DynamicCache()
        for i in range(len(past_flat) // 2):
            cache.update(past_flat[2 * i], past_flat[2 * i + 1], i)

        out = self.decoder(
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            use_cache=True,
        )
        logits = self.lm_head(out.last_hidden_state)  # [1, S, vocab]

        flat: list[torch.Tensor] = []
        for layer in out.past_key_values.layers:
            flat += [layer.keys, layer.values]
        return (logits, *flat)


def _names(prefix: str, n: int) -> list[str]:
    out: list[str] = []
    for i in range(n):
        out += [f"{prefix}.{i}.key", f"{prefix}.{i}.value"]
    return out


def main() -> None:
    model_id = parse_model()
    checkpoint = checkpoint_for(model_id)
    ensure_access(checkpoint)
    out_dir = variant_dir(model_id) / "lm"
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    # eager attention exports far more cleanly than SDPA/flash for our wrapper.
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    cfg = model.config
    layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", hidden // cfg.num_attention_heads)

    wrapper = LmExportWrapper(model).eval()

    # Trace with a NON-EMPTY past (past=3): HF's cache `update` short-circuits on
    # an empty cache (`keys.numel()==0` → replace instead of concat), and the
    # legacy tracer bakes whichever branch it sees. A non-empty past bakes the
    # `cat(past, new)` branch, which also handles a 0-length past at runtime
    # (cat with an empty tensor is a no-op). Validated below.
    seq, k, past = 4, 1, 3
    example = (
        torch.randint(0, cfg.vocab_size, (seq,), dtype=torch.long),
        torch.randn(k, hidden),
        *[torch.randn(1, kv_heads, past, head_dim) for _ in range(2 * layers)],
    )
    dynamic = {"input_ids": {0: "seq"}, "prefix_embeds": {0: "k"}, "logits": {1: "total"}}
    for nme in _names("past_key_values", layers):
        dynamic[nme] = {2: "past"}
    for nme in _names("present", layers):
        dynamic[nme] = {2: "total"}

    out_path = out_dir / "model.onnx"
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

    # torch.onnx writes external data next to the model only when tensors exceed
    # the protobuf 2 GB limit; record the sidecar name so the runtime can mount
    # it (ORT-Web needs it explicitly). Pico usually stays inline.
    sidecar = out_path.name + ".data"
    meta = {"numLayers": layers, "kvHeads": kv_heads, "headDim": head_dim, "hidden": hidden}
    if (out_dir / sidecar).exists():
        meta["externalData"] = sidecar
    (out_dir / "meta.json").write_text(json.dumps(meta))
    print(f"exported {model_id} LM -> {out_path}")
    print(f"meta: layers={layers} kvHeads={kv_heads} headDim={head_dim} hidden={hidden}")

    _validate_cpu(wrapper, out_path, layers, kv_heads, head_dim, hidden, cfg.vocab_size)
    print(
        f"NOTE: copy tokenizer.json via fetch_tokenizer.py --model {model_id}; "
        f"then run smoke_reference.py --model {model_id}."
    )


def _validate_cpu(
    wrapper: LmExportWrapper,
    out_path: Path,
    layers: int,
    kv_heads: int,
    head_dim: int,
    hidden: int,
    vocab: int,
) -> None:
    """Prove the exported graph's KV cache is wired: a single-shot prefill in
    torch must equal a prefill+decode split run through ONNX Runtime. If the
    export had dropped the past (the `numel()==0` trap), the split run's final
    logits would diverge from the one-shot reference.
    """
    import numpy as np
    import onnxruntime as ort

    past_names = _names("past_key_values", layers)
    present_names = _names("present", layers)
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    torch.manual_seed(0)
    prompt = torch.randint(0, vocab, (5,), dtype=torch.long)
    prefix = torch.randn(1, hidden)
    empty = [np.zeros((1, kv_heads, 0, head_dim), np.float32) for _ in range(2 * layers)]

    with torch.no_grad():
        ref_logits = wrapper(prompt, prefix, *[torch.zeros(1, kv_heads, 0, head_dim)] * (2 * layers))[0]
    ref_last = ref_logits[0, -1].numpy()

    def run(ids: np.ndarray, pfx: np.ndarray, past: list[np.ndarray]):
        feeds = {"input_ids": ids, "prefix_embeds": pfx}
        feeds.update({n: t for n, t in zip(past_names, past)})
        outs = sess.run(["logits", *present_names], feeds)
        return outs[0], outs[1:]

    # Stage 1: prefix + prompt[:-1].  Stage 2: prompt[-1] using the returned cache.
    _, present = run(prompt[:-1].numpy().astype(np.int64), prefix.numpy().astype(np.float32), empty)
    logits2, _ = run(prompt[-1:].numpy().astype(np.int64), np.zeros((0, hidden), np.float32), present)
    ort_last = logits2[0, -1]

    diff = float(np.abs(ort_last - ref_last).max())
    match = int(ort_last.argmax()) == int(ref_last.argmax())
    print(f"KV-cache parity: ORT(prefill+decode) vs torch(one-shot) max|diff|={diff:.2e}, argmax_match={match}")
    if diff > 1e-3 or not match:
        raise SystemExit(f"LM export KV-cache parity FAILED (diff={diff:.2e}, argmax_match={match})")


if __name__ == "__main__":
    main()
