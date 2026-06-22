# conversion

Python tooling to turn the upstream PyTorch models into the ONNX artifacts the
web app consumes, and to produce "golden" reference outputs for parity testing.

These scripts are the **Phase 0 / Phase 1** deliverables. They are intentionally
thin and explicit; each prints what it produced and where.

## Setup

Recommended ([uv](https://docs.astral.sh/uv/)); `.python-version` pins 3.13
(torch has no wheels for 3.14 yet):

```bash
uv sync                      # creates .venv from pyproject.toml
uv run python prepare_artifacts.py          # public stages
uv run python prepare_artifacts.py --gated  # + gated LM stages (needs HF login)
```

Or with plain pip:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### One command vs. individual scripts

`prepare_artifacts.py` runs the export/golden scripts below in dependency order
(`--list` to see them, `--only NAME ...` to run a subset). The individual
scripts still work standalone; the orchestrator just sequences them.

ONNX export uses whichever exporter each model needs (both ship with torch):
the **TorchDynamo exporter** (`dynamo=True`, needs `onnxscript`) for the Kanade
decoder, whose complex-tensor RoPE the legacy exporter rejects; the **legacy
exporter** (`dynamo=False`) for the LM and the clone encoder, where dynamo trips
on data-dependent control flow (WavLM) / variadic KV-cache inputs. Each script
picks the right one and self-checks parity.

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

**Working today — Phase 5 voice cloning** (no credentials; Kanade is public):

| Script                    | Produces                                                   |
| ------------------------- | ---------------------------------------------------------- |
| `export_clone_encoder.py` | `clone_encoder.onnx` + `clone_golden.json` (WavLM + GlobalEncoder → 128-dim; ORT-CPU cosine 1.0) |
| `gen_toy_lm.py`           | `lm_toy/model.onnx` + `meta.json`, `phase1_toy_golden.json` — a toy LM matching the export contract, for the Phase 1 browser gate |

**Gated — Phase 1 LM** (`syvai/plapre-pico`). Written to the contract the browser
loop is already validated against; runs end-to-end once authenticated:

| Script                   | Produces                                              |
| ------------------------ | ----------------------------------------------------- |
| `fetch_tokenizer.py`     | `tokenizer.json` (+ `config.json`)                    |
| `export_lm.py`           | `lm/model.onnx` (+ data) + `lm/meta.json`             |
| `precompute_speakers.py` | `speakers.json` (raw 128 + hidden) + `speaker_proj.json` |
| `smoke_reference.py`     | `golden/tokens.json` + `kanade.json` (greedy torch oracle mirroring `lm.ts`; stage-3 mel/wav only if upstream `plapre` is importable) |
| `validate_lm_golden.py`  | nothing; asserts the exported `lm.onnx` reproduces `golden/tokens.json` bit-for-bit under ORT greedy decode |
| `validate_e2e.py`        | `golden/e2e.wav`; chains lm+decoder+vocoder ONNX (text -> audio) and sanity-checks the waveform |
| `gen_reference_audio.py` | `golden/reference_torch*.wav` + `reference_onnx*.wav`; full-precision torch ground truth vs our ONNX decode/vocode on the same tokens (mel/wav `max|diff|` + corr) |

### Verify the artifacts you just generated (one interface)

`verify.py` is the **quality sanity check** — "did the files come out right?" It
runs the integrity / sample-comparison checks above and prints a single
PASS/FAIL summary so you don't have to run them one by one:

```bash
python verify.py            # lm-parity + e2e-sanity + ref-compare
python verify.py --list     # the checks and what they assert
python verify.py --only lm-parity e2e-sanity   # a subset (skips the torch one)
```

| Check         | Script                  | Asserts                                            |
| ------------- | ----------------------- | -------------------------------------------------- |
| `lm-parity`   | `validate_lm_golden.py` | exported `lm.onnx` == torch token golden (bit-exact) |
| `e2e-sanity`  | `validate_e2e.py`       | text → audio through all 3 ONNX graphs; waveform sane |
| `ref-compare` | `gen_reference_audio.py`| our ONNX decode/vocode vs the **real** torch one (mel/wav `max|diff|` + corr) |

Recorded green run (2026-06): `lm-parity` match=True (172 ids); `e2e-sanity`
3.28s peak 0.488; `ref-compare` mel`|diff|`≈8.6e-6, wav`|diff|`≈0.039,
corr≈0.9998. All need the gated weights + generated artifacts present.

### Tune the sampling temperature (a developer endpoint, not a check)

Temperature is a **runtime** knob (the artifacts are temperature-agnostic — the
LM only emits logits). `tune_temperature.py` renders a long text through the
full ONNX pipeline at several temperatures so you can pick one to lock into the
library via `generation.temperature`:

```bash
python tune_temperature.py --temps 0.5,0.6,0.7,0.8 --speaker tor
afplay golden/demo_t0.6.wav
```

See [`docs/TUNING.md`](../docs/TUNING.md) for where/how to set it in the library
(engine default, per-engine, per-request) and the OpenAI/ElevenLabs adapter
caveat.

`_gated.py` is the shared access check: it turns a `GatedRepoError` into one
actionable message. Unblock with:

```bash
huggingface-cli login            # after clicking "Agree" on the model page
python fetch_tokenizer.py && python export_lm.py && python precompute_speakers.py
python smoke_reference.py        # golden token ids (greedy torch oracle)
python verify.py                 # lm-parity + e2e-sanity + ref-compare (one interface)
```

`export_lm.py` self-checks single-step KV-cache parity; `validate_lm_golden.py`
checks the full autoregressive decode. The generation stop token is `<eos>`
(id 0), not SmolLM2's `<|endoftext|>` (absent here) — the browser tokenizer
resolves `<eos>` accordingly.

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
`window.__phase0`). Expected: both backends PASS (mel ≈ 1e-5, wav ≈ 1e-7). The
WebGPU `SkipLayerNormalization` fusion rejects our LayerNorm bias, so `ort.ts`
uses `graphOptimizationLevel: "basic"` on WebGPU to avoid it; see the root
`README.md` ("Verified correctness") for the recorded numbers.

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

## Reproduce the Phase 1 (LM loop) + Phase 5 (clone) gates

Both gates are **public** — no gated weights needed (they validate the browser
runtime against the export contract / a public model):

```bash
python gen_toy_lm.py            # toy LM (ORT-CPU greedy == torch golden)
python export_clone_encoder.py  # clone encoder (ORT-CPU cosine 1.0 vs torch)
```

Then in a browser (from `../web`, `npm run dev`):

- `phase1.html` (+ `?backend=webgpu`) — drives the real `OrtLmGraph`+`PlapreLM`
  KV-cache loop against the toy LM; expect 30/30 golden ids on both backends.
- `phase5.html` (+ `?backend=webgpu`) — runs the clone encoder under ORT-Web;
  expect cosine 1.0 on both backends.
- `bench.html?backend=webgpu&iters=10` — Phase 4 RTF/latency for the stages
  present.

## Status

- **Phase 0 (decoder + vocoder)** — cleared end-to-end under onnxruntime-web on
  WASM + WebGPU vs the PyTorch golden.
- **Phase 1 (LM loop)** — the browser KV-cache decode loop is validated (unit
  tests + toy-LM browser gate on both backends). The real gated export
  (`export_lm.py`, `precompute_speakers.py`, `fetch_tokenizer.py`,
  `smoke_reference.py`) is written to that proven contract and runs once
  authenticated.
- **Phase 5 (voice cloning)** — cleared: `clone_encoder.onnx` runs under
  onnxruntime-web reproducing the torch embedding (cosine 1.0).

See the root `README.md` ("Verified correctness" and "Making it run as ONNX in
the browser") for recorded numbers and the WebGPU `SkipLayerNormalization` caveat.
