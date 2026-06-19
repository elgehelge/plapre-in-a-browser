"""Build a tiny TOY LM that follows the Phase 1 export contract, to validate the
browser KV-cache decode loop (`OrtLmGraph` in web/src/pipeline/lm.ts) under real
onnxruntime-web — independent of the gated Plapre weights.

It is a genuine causal multi-head-attention transformer with a real KV cache
(past K/V are concatenated each step), so it exercises exactly the wiring the
real export needs:

  inputs : input_ids   int64   [seq]
           prefix_embeds float32 [k, hidden]
           past_key_values.{i}.{key,value} float32 [1, kvHeads, past, headDim]
  outputs: logits      float32 [1, k+seq, vocab]
           present.{i}.{key,value}         float32 [1, kvHeads, past+k+seq, headDim]

Writes web/public/models/lm_toy/model.onnx (+ meta.json) and a greedy golden
(phase1_toy_golden.json) the browser harness reproduces. All regenerable /
git-ignored.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = Path(__file__).parent.parent / "web" / "public" / "models"
OUT = MODELS / "lm_toy" / "model.onnx"
META = MODELS / "lm_toy" / "meta.json"
GOLDEN = MODELS / "phase1_toy_golden.json"

VOCAB = 20
HIDDEN = 16
KV_HEADS = 2
HEAD_DIM = 8  # HIDDEN == KV_HEADS * HEAD_DIM
LAYERS = 2
EOS = 19
SEED = 0


class ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.mlp = nn.Sequential(nn.Linear(HIDDEN, 4 * HIDDEN), nn.GELU(), nn.Linear(4 * HIDDEN, HIDDEN))

    def _split(self, t: torch.Tensor) -> torch.Tensor:  # [1,S,H] -> [1,heads,S,d]
        s = t.shape[1]
        return t.view(1, s, KV_HEADS, HEAD_DIM).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self._split(self.q(x))  # [1,heads,S,d]
        k = self._split(self.k(x))
        v = self._split(self.v(x))
        k = torch.cat([past_k, k], dim=2)  # [1,heads,T,d], T=past+S
        v = torch.cat([past_v, v], dim=2)

        s = q.shape[2]
        t = k.shape[2]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)  # [1,heads,S,T]
        # Causal mask: query i (absolute pos t-s+i) may attend to keys 0..t-s+i.
        qpos = torch.arange(s).unsqueeze(1) + (t - s)
        kpos = torch.arange(t).unsqueeze(0)
        mask = (kpos > qpos).unsqueeze(0).unsqueeze(0)  # [1,1,S,T] True = block
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1) @ v  # [1,heads,S,d]
        attn = attn.transpose(1, 2).reshape(1, s, HIDDEN)
        x = x + self.o(attn)
        x = x + self.mlp(x)
        return x, k, v


class ToyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList(ToyLayer() for _ in range(LAYERS))
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, input_ids, prefix_embeds, *past):  # noqa: ANN001
        tok = self.embed(input_ids)  # [seq,H]
        x = torch.cat([prefix_embeds, tok], dim=0).unsqueeze(0)  # [1,S,H]
        presents: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, past[2 * i], past[2 * i + 1])
            presents += [k, v]
        return (self.head(x), *presents)


def _past_names() -> list[str]:
    names = []
    for i in range(LAYERS):
        names += [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
    return names


def _present_names() -> list[str]:
    names = []
    for i in range(LAYERS):
        names += [f"present.{i}.key", f"present.{i}.value"]
    return names


def _greedy(model: ToyLM, prompt_ids: list[int], max_tokens: int) -> list[int]:
    """Reference decode loop matching PlapreLM.generate (greedy)."""
    empty = [torch.zeros(1, KV_HEADS, 0, HEAD_DIM) for _ in range(2 * LAYERS)]
    prefix = torch.zeros(1, HIDDEN)  # toy golden uses a zero speaker prefix
    with torch.no_grad():
        out = model(torch.tensor(prompt_ids, dtype=torch.long), prefix, *empty)
    logits, present = out[0], list(out[1:])
    ids: list[int] = []
    for _ in range(max_tokens):
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS:
            break
        ids.append(nxt)
        with torch.no_grad():
            out = model(
                torch.tensor([nxt], dtype=torch.long), torch.zeros(0, HIDDEN), *present
            )
        logits, present = out[0], list(out[1:])
    return ids


def main() -> None:
    torch.manual_seed(SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = ToyLM().eval()

    seq, k = 4, 1
    example = (
        torch.randint(0, VOCAB, (seq,), dtype=torch.long),
        torch.randn(k, HIDDEN),
        *[torch.zeros(1, KV_HEADS, 0, HEAD_DIM) for _ in range(2 * LAYERS)],
    )
    dynamic = {"input_ids": {0: "seq"}, "prefix_embeds": {0: "k"}, "logits": {1: "total"}}
    for n in _past_names():
        dynamic[n] = {2: "past"}
    for n in _present_names():
        dynamic[n] = {2: "total"}

    torch.onnx.export(
        model,
        example,
        str(OUT),
        input_names=["input_ids", "prefix_embeds", *_past_names()],
        output_names=["logits", *_present_names()],
        dynamic_axes=dynamic,
        opset_version=17,
        dynamo=False,  # legacy exporter handles *past varargs + dynamic_axes cleanly
    )
    print(f"exported toy LM -> {OUT}")

    META.write_text(
        json.dumps({"numLayers": LAYERS, "kvHeads": KV_HEADS, "headDim": HEAD_DIM, "hidden": HIDDEN})
    )

    # Golden: a fixed prompt decoded greedily by the reference loop.
    prompt = [0, 7, 3, 11, 1]
    ids = _greedy(model, prompt, max_tokens=30)
    GOLDEN.write_text(json.dumps({"prompt": prompt, "eos": EOS, "ids": ids}))
    print(f"toy golden: prompt={prompt} -> {len(ids)} ids: {ids}")

    _validate_cpu(prompt, ids)


def _validate_cpu(prompt: list[int], golden_ids: list[int]) -> None:
    """Sanity-check the ONNX greedy loop (CPU) matches the torch golden."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])

    def run(input_ids, prefix, past):
        feeds = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "prefix_embeds": prefix.astype(np.float32),
        }
        for n, t in zip(_past_names(), past):
            feeds[n] = t
        outs = sess.run(["logits", *_present_names()], feeds)
        return outs[0], outs[1:]

    empty = [np.zeros((1, KV_HEADS, 0, HEAD_DIM), np.float32) for _ in range(2 * LAYERS)]
    logits, present = run(prompt, np.zeros((1, HIDDEN)), empty)
    ids: list[int] = []
    for _ in range(30):
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS:
            break
        ids.append(nxt)
        logits, present = run([nxt], np.zeros((0, HIDDEN)), present)
    ok = ids == golden_ids
    print(f"ORT-CPU greedy ids match torch golden: {ok}")
    if not ok:
        raise SystemExit(f"toy LM ORT/torch mismatch: {ids} != {golden_ids}")


if __name__ == "__main__":
    main()
