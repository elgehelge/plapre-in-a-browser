# conversion

Python tooling to turn the upstream PyTorch models into the ONNX artifacts the
web app consumes, and to produce "golden" reference outputs for parity testing.

These scripts are the **Phase 0 / Phase 1** deliverables. They are intentionally
thin and explicit; each prints what it produced and where.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Hugging Face access (checked 2026-06)

| Repo                            | Used by                         | Access        |
| ------------------------------- | ------------------------------- | ------------- |
| `frothywater/kanade-25hz-clean` | decoder + vocoder (**Phase 0**) | **public**    |
| `syvai/plapre-pico` / `-nano`   | LM, `speaker_proj.pt`, `speakers.json` (**Phase 1**) | **gated** (`gated: auto`) |

So **Phase 0 (vocoder/decoder de-risk) needs no credentials** — the Kanade repo
is public. Phase 1 (LM export + speaker precompute) is gated: accept the
conditions on Hugging Face and `huggingface-cli login` first.

### num2words parity tooling (no model downloads)

`gen_num2words_fixtures.py` only needs `num2words`; it regenerates the golden
fixture the web `num2words-da` parity test asserts against:

```bash
python -m venv .venv && source .venv/bin/activate && pip install num2words
.venv/bin/python gen_num2words_fixtures.py
```

## Scripts (run in this order)

| Script                       | Produces                                   | Phase |
| ---------------------------- | ------------------------------------------ | ----- |
| `smoke_reference.py`         | golden token ids + mel + wav for a fixed sentence/speaker | 0 |
| `export_kanade_decoder.py`   | `kanade_decoder.onnx`                       | 0 |
| `export_hift_vocoder.py`     | `hift_vocoder.onnx`                         | 0 |
| `precompute_speakers.py`     | `speakers.json` (raw 128-dim + projected hidden) | 1 |
| `export_lm.py`               | `lm.onnx` (+ external data)                 | 1 |

Outputs are written to `../web/public/models/` by default; golden fixtures to
`./golden/`.

## Status

All scripts are **scaffolds** with the intended structure, the exact upstream
calls to wrap (traced from `plapre/inference.py`), and `TODO` markers where the
tracing/export wiring must be completed. Start with `smoke_reference.py` +
`export_hift_vocoder.py` (the Phase 0 gate).
