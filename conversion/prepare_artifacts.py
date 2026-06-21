"""One command to produce the converted ONNX artifacts the web app consumes.

Runs the individual export/golden scripts in dependency order so you don't have
to remember the sequence. Outputs land in ../web/public/models/ (git-ignored).

Stages are split by credential requirement:

* PUBLIC  — Kanade decoder + HiFT vocoder + clone encoder and their golden
            fixtures. The Kanade repo is public, so these need no login.
* GATED   — the Plapre language model, tokenizer, and speaker embeddings. These
            come from the license-gated syvai/plapre-* repo: accept the model
            conditions on Hugging Face and `huggingface-cli login` first.

Usage (from conversion/, with the env active — `uv run python prepare_artifacts.py`):

    python prepare_artifacts.py              # public stages only
    python prepare_artifacts.py --gated      # public + gated (needs HF login)
    python prepare_artifacts.py --only export_lm precompute_speakers
    python prepare_artifacts.py --list       # show the stages and exit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent


@dataclass(frozen=True)
class Stage:
    name: str
    script: str
    gated: bool
    produces: str


# Order matters: later stages may depend on earlier outputs.
STAGES: tuple[Stage, ...] = (
    Stage("export_kanade_decoder", "export_kanade_decoder.py", False,
          "kanade_decoder.onnx (+ .onnx.data)"),
    Stage("export_hift_vocoder", "export_hift_vocoder.py", False,
          "hift_vocoder.onnx (+ .onnx.data)"),
    Stage("gen_phase0_golden", "gen_phase0_golden.py", False,
          "phase0_golden.json (decoder+vocoder browser gate)"),
    Stage("export_clone_encoder", "export_clone_encoder.py", False,
          "clone_encoder.onnx + clone_golden.json (voice cloning)"),
    Stage("gen_toy_lm", "gen_toy_lm.py", False,
          "lm_toy/ + phase1_toy_golden.json (LM-loop browser gate)"),
    Stage("gen_num2words_fixtures", "gen_num2words_fixtures.py", False,
          "../web/src/pipeline/__fixtures__/num2words-da.json"),
    Stage("fetch_tokenizer", "fetch_tokenizer.py", True,
          "tokenizer.json (+ config.json)"),
    Stage("export_lm", "export_lm.py", True,
          "lm/model.onnx (+ data) + lm/meta.json"),
    Stage("precompute_speakers", "precompute_speakers.py", True,
          "speakers.json + speaker_proj.json"),
    Stage("smoke_reference", "smoke_reference.py", True,
          "golden/tokens.json + kanade.json (torch oracle)"),
    Stage("validate_lm_golden", "validate_lm_golden.py", True,
          "asserts exported lm.onnx == torch golden"),
)


def run_stage(stage: Stage) -> None:
    print(f"\n=== {stage.name} -> {stage.produces} ===", flush=True)
    started = time.monotonic()
    subprocess.run([sys.executable, stage.script], cwd=HERE, check=True)
    print(f"--- {stage.name} done in {time.monotonic() - started:.1f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gated", action="store_true",
                        help="also run gated stages (requires Hugging Face login)")
    parser.add_argument("--only", nargs="+", metavar="STAGE",
                        help="run only these stage names (implies their credential needs)")
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    args = parser.parse_args()

    if args.list:
        for s in STAGES:
            print(f"  {'[gated]' if s.gated else '[public]'} {s.name:<24} {s.produces}")
        return 0

    if args.only:
        unknown = [n for n in args.only if n not in {s.name for s in STAGES}]
        if unknown:
            parser.error(f"unknown stage(s): {', '.join(unknown)}")
        selected = [s for s in STAGES if s.name in set(args.only)]
    else:
        selected = [s for s in STAGES if not s.gated or args.gated]

    gated = [s for s in selected if s.gated]
    if gated and not args.only:
        print("Running GATED stages — these need an authenticated Hugging Face "
              "session (huggingface-cli login) and accepted model terms.")

    for stage in selected:
        try:
            run_stage(stage)
        except subprocess.CalledProcessError as exc:
            print(f"\nFAILED at stage '{stage.name}' (exit {exc.returncode}).",
                  file=sys.stderr)
            if stage.gated:
                print("This is a gated stage — confirm you accepted the model "
                      "terms and ran `huggingface-cli login`.", file=sys.stderr)
            return exc.returncode

    print(f"\nAll {len(selected)} stage(s) complete. Artifacts in "
          f"web/public/models/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
