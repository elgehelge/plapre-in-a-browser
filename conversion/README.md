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

## Scripts

**Working today — Phase 0** (no credentials; the Kanade repo is public). Outputs
go to `../web/public/models/` (all git-ignored):

| Script                      | Produces                                                    |
| --------------------------- | ----------------------------------------------------------- |
| `export_kanade_decoder.py`  | `kanade_decoder.onnx` (+ `.onnx.data`) — slim decode-only   |
| `export_hift_vocoder.py`    | `hift_vocoder.onnx` (+ `.onnx.data`)                        |
| `gen_phase0_golden.py`      | `phase0_golden.json` — golden mel + wav for the browser gate |
| `hift_onnx.py`              | nothing; `python hift_onnx.py` self-tests the real (i)STFT  |
| `gen_num2words_fixtures.py` | `../web/src/pipeline/__fixtures__/num2words-da.json`        |

**Scaffolds — Phase 1** (LM port; gated repos). Intended structure + the exact
upstream calls to wrap, not yet runnable end-to-end:

| Script                   | Will produce                                     |
| ------------------------ | ------------------------------------------------ |
| `smoke_reference.py`     | golden token ids for a fixed sentence/speaker    |
| `precompute_speakers.py` | `speakers.json` (raw 128-dim + projected hidden) |
| `export_lm.py`           | `lm/model.onnx` (+ external data)                |

## Reproduce the Phase 0 gate end-to-end

From `conversion/` with the venv active:

```bash
# 1. Export both models. Each self-checks ORT-CPU parity vs PyTorch and refuses
#    to write a graph that drifts past tolerance (so a green run == a good graph).
python export_kanade_decoder.py   # slim decode-only wrapper; ~1e-5 mel parity
python export_hift_vocoder.py     # real-valued (i)STFT rewrite; ~1e-6 wav parity

# 2. Produce the golden mel+wav from the SAME wrappers, on fixed inputs.
python gen_phase0_golden.py
```

What "slim" / "deterministic" mean here (both happen inside
`export_kanade_decoder.py`, see its module docstring + `_disable_attention_dropout`):

- **Slim:** we export a wrapper holding only the decode submodules (quantizer +
  mel prenet/upsample/decoder/postnet), not the whole `KanadeModel`, so the WavLM
  SSL encoder is excluded by construction. (The dynamo exporter also tree-shakes
  it; the wrapper makes the boundary explicit. The remaining ~365 MB is the
  genuine decode path — see size note above.)
- **Deterministic:** we zero the upstream inference-time attention dropout, so
  the exported graph and the golden are reproducible.

Then verify in a **real browser** (from `../web`):

```bash
npm install
npm run dev                        # serves http://localhost:5173
# open http://localhost:5173/phase0.html                 (WASM)
#  and http://localhost:5173/phase0.html?backend=webgpu   (WebGPU)
```

`phase0.html` loads both ONNX models via onnxruntime-web, runs decoder → vocoder
on the golden inputs, and prints mel/wav `max|diff|` vs the golden (also on
`window.__phase0`). Expected: both backends PASS (mel ≈ 1e-5, wav ≈ 1e-7). See
`docs/PLAN.md` for the recorded numbers and the WebGPU `SkipLayerNormalization`
caveat (`ort.ts` uses `graphOptimizationLevel: "basic"` on WebGPU to avoid it).

## Phase 0 gate findings (2026-06)

- **`export_kanade_decoder.py` — works.** Exports a slim decode-only wrapper
  (no SSL encoder) via the dynamo exporter and self-checks ORT-CPU mel parity
  vs PyTorch (max|diff| ≈ 1e-5). The legacy exporter fails on the transformer's
  complex-tensor RoPE. Also disables a buggy inference-time attention dropout
  (upstream's non-flash attention omits the `if self.training` guard), which
  otherwise makes decode nondeterministic.
- **`export_hift_vocoder.py` — works** (via `hift_onnx.py`). Stock HiFT uses
  `torch.stft`/`torch.istft` on complex tensors (no ONNX op; Vocos has the same
  blocker) and a `torch.rand`/`torch.randn` sine source.
  `hift_onnx.patch_vocoder_for_onnx` swaps in real-valued (i)STFT (n_fft=16,
  hop=4; matches `torch.istft` to ~1e-8) and a deterministic source. ORT-CPU
  wav parity vs PyTorch ≈ 9e-7. `python hift_onnx.py` self-tests the transforms.
- **Size follow-up:** the decoder ONNX is ~365 MB — the genuine decode path
  (mel_prenet ≈170 MB + mel_decoder ≈185 MB), not WavLM bloat (already
  excluded). Vocoder ~83 MB. Shrinking needs fp16/int8 quantization, deferred
  until after the browser gate.

## Status

**Phase 0 is cleared end-to-end:** the exported decoder + vocoder run under
onnxruntime-web on both WASM and WebGPU and reproduce the PyTorch golden
(`gen_phase0_golden.py` → `web/phase0.html`). See `docs/PLAN.md` for numbers and
the WebGPU `SkipLayerNormalization` caveat.

Remaining scripts (`smoke_reference.py`, `precompute_speakers.py`,
`export_lm.py`) are **scaffolds** with the intended structure and the exact
upstream calls to wrap (traced from `plapre/inference.py`) — these are Phase 1
(the LM port).
