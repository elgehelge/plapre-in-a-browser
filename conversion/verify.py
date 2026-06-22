"""One interface to quality-check the artifacts you just generated.

This is the "verify the integrity of the files" step of conversion. After
exporting the ONNX models + golden fixtures, `verify.py` runs the integrity /
sample-comparison checks and prints a single PASS/FAIL summary, so you don't
have to remember which individual scripts to run or how to read their output.

Checks (each remains runnable standalone — this just sequences them and
collects the headline numbers):

  lm-parity    validate_lm_golden.py   exported lm.onnx reproduces the torch
                                       token golden bit-for-bit (greedy decode)
  e2e-sanity   validate_e2e.py         text -> audio through all three ONNX
                                       graphs; waveform is finite / sane level
  ref-compare  gen_reference_audio.py  OUR ONNX decoder+vocoder vs the REAL
                                       torch kanade.decode()+vocode() on the
                                       SAME tokens (mel/wav max|diff| + corr)

All three need the gated weights and the generated artifacts present (HF
login). Produce them first with `python prepare_artifacts.py --gated`.

Tuning the sampling temperature is a separate, developer-facing concern (the
artifacts are temperature-agnostic) — see tune_temperature.py and docs/TUNING.md.

Usage:
  python verify.py                       # run every check
  python verify.py --only lm-parity      # a subset
  python verify.py --list                # names only
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from _gated import DEFAULT_MODEL, MODELS

HERE = Path(__file__).parent


def _selected_model() -> str:
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
        if model not in MODELS:
            raise SystemExit(f"unknown model {model!r}; choose one of {', '.join(MODELS)}")
        return model
    return DEFAULT_MODEL


@dataclass
class Check:
    name: str
    script: str
    summary: str
    # Pull a short, human-readable metric line out of the script's stdout.
    metrics: "callable"


def _search(pattern: str, text: str, group: int = 1) -> str | None:
    m = re.search(pattern, text)
    return m.group(group) if m else None


def _lm_metrics(out: str) -> str:
    match = _search(r"match=(\w+)", out)
    ids = _search(r"ORT greedy: (\d+) ids", out)
    return f"match={match}, ids={ids}"


def _e2e_metrics(out: str) -> str:
    dur = _search(r"= ([\d.]+)s @", out)
    peak = _search(r"peak=([\d.]+)", out)
    toks = _search(r"-> (\d+) kanade indices", out)
    return f"{dur}s, peak={peak}, {toks} kanade tokens"


def _ref_metrics(out: str) -> str:
    mel = _search(r"decoder mel\s+max\|diff\|.*?=\s*(\S+)", out)
    wav = _search(r"vocoder wav\s+max\|diff\|.*?=\s*(\S+);", out)
    corr = _search(r"corr = (\S+)", out)
    return f"mel|diff|={mel}, wav|diff|={wav}, corr={corr}"


CHECKS: list[Check] = [
    Check("lm-parity", "validate_lm_golden.py", "exported lm.onnx == torch golden", _lm_metrics),
    Check("e2e-sanity", "validate_e2e.py", "text -> audio, waveform sane", _e2e_metrics),
    Check("ref-compare", "gen_reference_audio.py", "ONNX decode/vocode vs real torch", _ref_metrics),
]


def run_check(check: Check, model: str) -> tuple[bool, str, float]:
    """Run one check as a subprocess. Returns (passed, metric_line, seconds)."""
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(HERE / check.script), "--model", model],
        capture_output=True,
        text=True,
        cwd=HERE,
    )
    elapsed = time.monotonic() - started
    out = proc.stdout + proc.stderr
    passed = proc.returncode == 0
    try:
        metric = check.metrics(out)
    except Exception:  # noqa: BLE001 - metric parsing must never mask a result
        metric = "(metrics unavailable)"
    if not passed:
        # Surface the script's own output so the failure is actionable.
        print(f"\n----- {check.name} output -----")
        print(out.rstrip())
        print("------------------------------\n")
        last = out.rstrip().splitlines()
        metric = last[-1] if last else "(no output)"
    return passed, metric, elapsed


def main() -> None:
    if "--list" in sys.argv:
        for c in CHECKS:
            print(f"{c.name:12s} {c.script:24s} {c.summary}")
        return

    selected = CHECKS
    if "--only" in sys.argv:
        wanted = sys.argv[sys.argv.index("--only") + 1 :]
        selected = [c for c in CHECKS if c.name in wanted]
        if not selected:
            raise SystemExit(f"--only matched nothing; known: {[c.name for c in CHECKS]}")

    model = _selected_model()
    print(f"Verifying {len(selected)} check(s) for model '{model}' against the generated artifacts…\n")
    results: list[tuple[Check, bool, str, float]] = []
    for c in selected:
        print(f"  running {c.name} ({c.script}) …", flush=True)
        passed, metric, secs = run_check(c, model)
        results.append((c, passed, metric, secs))

    print("\n" + "=" * 72)
    print(f"{'check':12s} {'result':7s} {'time':>7s}  details")
    print("-" * 72)
    for c, passed, metric, secs in results:
        mark = "PASS" if passed else "FAIL"
        print(f"{c.name:12s} {mark:7s} {secs:6.1f}s  {metric}")
    print("=" * 72)

    failed = [c.name for c, ok, _, _ in results if not ok]
    if failed:
        raise SystemExit(f"VERIFY FAILED: {', '.join(failed)}")
    print("VERIFY PASSED: generated artifacts are internally consistent.")


if __name__ == "__main__":
    main()
