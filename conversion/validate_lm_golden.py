"""Phase 1 end-to-end validation: the EXPORTED lm.onnx must reproduce the torch
golden token ids (conversion/golden/tokens.json) under a greedy decode that
mirrors web/src/pipeline/lm.ts bit-for-bit.

Unlike export_lm.py's single-step parity check, this runs the full
autoregressive loop (prefill the prompt with the speaker-hidden prefix, then feed
the KV cache back each step) entirely through onnxruntime, using the SAME
artifacts the browser loads:

  web/public/models/lm/model.onnx          the exported graph
  web/public/models/tokenizer.json         prompt tokenization
  web/public/models/speakers.json          raw 128-dim speaker embedding
  web/public/models/speaker_proj.json      128->hidden projection (applied here)

A match proves the export + speaker projection + decode contract are correct end
to end; the JS loop is already proven equivalent to this loop shape by the toy
browser gate, so a green run here closes Phase 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from _gated import golden_dir, parse_model, variant_dir
from smoke_reference import SENTENCE, SPEAKER, _normalize


def _speaker_hidden(models: Path) -> np.ndarray:
    """The speaker's precomputed hidden vector (what the browser feeds as the
    prefix), cross-checked against applying speaker_proj.json to the raw emb."""
    spk = json.loads((models / "speakers.json").read_text())[SPEAKER]
    hidden = np.asarray(spk["hidden"], dtype=np.float32)

    raw = np.asarray(spk["raw"], dtype=np.float32)
    proj = json.loads((models / "speaker_proj.json").read_text())
    weight = np.asarray(proj["weight"], dtype=np.float32)  # [out, in]
    bias = np.asarray(proj["bias"], dtype=np.float32)
    projected = weight @ raw + bias
    diff = float(np.abs(projected - hidden).max())
    print(f"speaker_proj.json vs precomputed hidden: max|diff|={diff:.2e}")
    if diff > 1e-3:
        raise SystemExit(f"speaker_proj.json inconsistent with speakers.json hidden (diff={diff:.2e})")
    return hidden  # [hidden]


def _prompt_ids(tok: Tokenizer) -> tuple[list[int], int]:
    def tid(t: str) -> int:
        return tok.token_to_id(t)

    norm = _normalize(SENTENCE)
    text_ids = tok.encode(norm, add_special_tokens=False).ids
    prompt = [tid("<text>"), *text_ids, tid("<audio>")]
    eos = tid("<eos>")  # AutoTokenizer.eos_token_id == 0 == "<eos>" for this model
    return prompt, eos


def main() -> None:
    model_id = parse_model()
    models = variant_dir(model_id)
    golden_path = golden_dir(model_id) / "tokens.json"

    meta = json.loads((models / "lm" / "meta.json").read_text())
    layers, kv_heads, head_dim, hidden = (
        meta["numLayers"],
        meta["kvHeads"],
        meta["headDim"],
        meta["hidden"],
    )
    golden = json.loads(golden_path.read_text())["ids"]

    tok = Tokenizer.from_file(str(models / "tokenizer.json"))
    prompt, eos = _prompt_ids(tok)
    speaker_hidden = _speaker_hidden(models).reshape(1, hidden).astype(np.float32)

    sess = ort.InferenceSession(str(models / "lm" / "model.onnx"), providers=["CPUExecutionProvider"])
    past_names = [f"past_key_values.{i}.{kv}" for i in range(layers) for kv in ("key", "value")]
    present_names = [f"present.{i}.{kv}" for i in range(layers) for kv in ("key", "value")]

    def run(input_ids: list[int], prefix: np.ndarray, past: list[np.ndarray]):
        feeds = {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "prefix_embeds": prefix,
        }
        feeds.update({n: t for n, t in zip(past_names, past)})
        outs = sess.run(["logits", *present_names], feeds)
        return outs[0], outs[1:]

    empty = [np.zeros((1, kv_heads, 0, head_dim), np.float32) for _ in range(2 * layers)]
    logits, present = run(prompt, speaker_hidden, empty)

    ids: list[int] = []
    no_prefix = np.zeros((0, hidden), np.float32)
    for _ in range(len(golden) + 5):
        nxt = int(logits[0, -1].argmax())
        if nxt == eos:
            break
        ids.append(nxt)
        logits, present = run([nxt], no_prefix, present)

    ok = ids == golden
    n = min(len(ids), len(golden))
    first_div = next((i for i in range(n) if ids[i] != golden[i]), n)
    print(f"ORT greedy: {len(ids)} ids; golden: {len(golden)} ids")
    print(f"match={ok}; first divergence at index {first_div if not ok else '-'}")
    if not ok:
        print(f"  ort[:8]   = {ids[:8]}")
        print(f"  golden[:8]= {golden[:8]}")
        raise SystemExit("Phase 1 LM golden parity FAILED")
    print("Phase 1 LM end-to-end parity PASSED (exported ONNX == torch golden).")


if __name__ == "__main__":
    main()
