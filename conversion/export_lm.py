"""Phase 1: export the Plapre LM (SmolLM2 / LLaMA) to ONNX.

The LM must be exported to accept `inputs_embeds` (NOT token ids) plus a KV
cache, because Plapre injects the speaker hidden vector as the first input
embedding (see docs/ARCHITECTURE.md, Stage 2).

Preferred route: 🤗 optimum-cli, which handles the decoder + KV-cache graph and
produces a transformers.js-compatible layout.

    optimum-cli export onnx \
        --model syvai/plapre-pico \
        --task text-generation-with-past \
        ../web/public/models/lm/

Open items to verify after export:
  1. The exported graph exposes `inputs_embeds` as an input. text-generation
     exports are usually token-id based; we may need a custom ONNX config / a
     wrapper module whose forward takes inputs_embeds, or to run embed_tokens
     as a separate tiny ONNX (gather) and concat the speaker hidden in JS.
  2. Quantization: start fp16/q8 for the LM to keep the download small
     (Pico q8 ≈ 121 MB). Validate quality vs golden audio.
  3. Ship tokenizer.json + config.json alongside for the JS tokenizer.

This file documents the intended command and wraps it for convenience; the
custom-config work for inputs_embeds is the real task (TODO).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "web" / "public" / "models" / "lm"
CHECKPOINT = "syvai/plapre-pico"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", CHECKPOINT,
        "--task", "text-generation-with-past",
        str(OUT_DIR),
    ]
    print("Running:", " ".join(cmd))
    print("TODO: confirm `inputs_embeds` is an input; otherwise add a custom ONNX config.")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    sys.exit(main())
