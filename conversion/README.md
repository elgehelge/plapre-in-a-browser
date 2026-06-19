# conversion

Python tooling to turn the upstream PyTorch models into the ONNX artifacts the
web app consumes, and to produce "golden" reference outputs for parity testing.

These scripts are the **Phase 0 / Phase 1** deliverables. They are intentionally
thin and explicit; each prints what it produced and where.

## Setup

```bash
# Use Python 3.13 — torch has no wheels for 3.14 yet.
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

ONNX export uses the **TorchDynamo exporter** (`torch.onnx.export(dynamo=True)`),
which needs `onnxscript`. The legacy TorchScript exporter cannot handle the
decoder's complex-tensor RoPE (see Phase 0 findings below).

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

## Phase 0 gate findings (2026-06)

- **`export_kanade_decoder.py` — works.** Exports via the dynamo exporter and
  self-checks ORT-CPU mel parity vs PyTorch (max|diff| ≈ 0.008). Writes
  `kanade_decoder.onnx` (+ a large `.onnx.data`; see size follow-up below).
  The legacy exporter fails here on the transformer's complex-tensor RoPE.
- **`export_hift_vocoder.py` — blocked.** The HiFT vocoder uses
  `torch.stft`/`torch.istft` on complex tensors, which no ONNX exporter
  supports. The script documents the real-valued (i)STFT remediation (the
  transforms are tiny: n_fft=16, hop=4) and refuses to emit a broken graph.
  Vocos has the same `torch.istft` blocker, so it is not a fallback.
- **Size follow-up:** the decoder export currently embeds the whole
  `KanadeModel` (~365 MB, includes the unused WavLM encoder). Export only the
  decode submodules before shipping to the browser.

## Status

Remaining scripts (`smoke_reference.py`, `precompute_speakers.py`,
`export_lm.py`) are **scaffolds** with the intended structure and the exact
upstream calls to wrap (traced from `plapre/inference.py`). Phase 0 next step:
implement the real-valued (i)STFT HiFT subclass, then run the in-browser
wasm/webgpu verification.
