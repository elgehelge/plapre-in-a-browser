# plapre-in-a-browser

Run [**Plapre**](https://huggingface.co/syvai/plapre-pico) — an open-source Danish
text-to-speech model — **fully in the browser**, with no server and no cloud calls.
Inference runs on `onnxruntime-web` (WebGPU, with a WASM fallback).

This is a research/PoC project. The long-term goal is a reusable in-browser Danish
TTS module that can be dropped into a Chrome MV3 extension (offscreen document) as a
local TTS provider.

It is designed as a **plug-in replacement for hosted TTS APIs**: a provider-neutral
engine emits canonical 24 kHz PCM, and thin adapters make it a drop-in stand-in for
the **OpenAI** (`audio.speech`) and **ElevenLabs** (`text-to-speech`) APIs — same
request/voice/format shapes, but running locally with no server and no API key. See
[docs/INTERFACE.md](docs/INTERFACE.md) for the contract and the adapter mapping.

> Spelling note: the upstream model is **plapre** (`syvai/plapre-*`). This repo uses
> that spelling throughout.

## Why this is non-trivial

Plapre is **not** a single self-contained TTS model. It is a 3-stage
autoregressive LLM + neural-codec pipeline:

```
Danish text
  │  (1) normalize + BPE tokenize, build prompt: [<text>] ids [<audio>]
  ▼
Plapre LM (SmolLM2 / LLaMA; Pico 135M or Nano 360M)
  │      speaker identity is injected as the FIRST input embedding
  │      (128-dim speaker vec → Linear → prepended to inputs_embeds)
  │  (2) autoregressively sample ~25 audio tokens / second of speech
  ▼
audio token ids  →  Kanade content-token indices (subtract <audio_0> offset)
  │  (3a) Kanade.decode(tokens, speaker_embedding) → mel spectrogram
  ▼
HiFT vocoder (from CosyVoice 2)
  │  (3b) mel → 24 kHz waveform
  ▼
PCM audio (24 kHz, mono, float32)
```

So porting to the browser means converting **three** neural components to ONNX and
writing a custom autoregressive generation loop in JS. None of these have published
web builds today — that conversion is the core of this project.

| Component        | What it is                         | Size (Pico)        | Status |
| ---------------- | ---------------------------------- | ------------------ | ------ |
| Plapre LM        | SmolLM2/LLaMA, autoregressive      | 135M (q8 ≈ 121 MB) | loop validated; gated export written |
| Kanade decoder   | tokens + speaker emb → mel         | ≈ 367 MB           | **converted, runs in-browser** |
| HiFT vocoder     | mel → 24 kHz waveform              | ≈ 85 MB            | **converted, runs in-browser** |
| Kanade clone enc | reference audio → 128-dim speaker  | 174 MB (WavLM)     | **converted, runs in-browser** |

The built-in speakers (`tor, ida, liv, ask, kaj`) ship as precomputed embeddings,
so only the Kanade **decoder** is needed for built-in voices. **Voice cloning**
(Phase 5) additionally exports the Kanade global encoder (WavLM frontend +
GlobalEncoder) to derive a speaker embedding from reference audio entirely in the
browser — see `Engine.cloneVoice()` and the ElevenLabs Instant Voice Cloning
adapter.

## Repository layout

```
conversion/   Python: export the 3 models to ONNX, precompute speaker embeddings,
              and produce "golden" reference outputs for parity testing.
web/          Vite + TypeScript app running the pipeline on onnxruntime-web.
              The full pipeline (LM decode loop, decoder, vocoder, clone encoder),
              the provider-neutral engine, adapters, audio, and model caching are
              implemented and tested; ONNX stages load from web/public/models/.
              phase0/phase1/phase5/bench .html are the in-browser gates.
docs/         PLAN.md (phased plan + risks), ARCHITECTURE.md (data flow detail),
              INTERFACE.md (engine contract + OpenAI/ElevenLabs adapters), and
              EXTENSION.md (Chrome MV3 offscreen-document hosting).
```

## Status / where to start

### Implemented and unit-tested today (no model files required)

These run and are covered by `npm test` (vitest) in `web/`:

- **Danish text normalization** + sentence splitting, matching the reference.
- **Danish number-to-words**, reproducing `num2words(da)` exactly (locked by a
  2037-case golden fixture).
- **Autoregressive sampling** (temperature / top-k / top-p, seeded RNG).
- **Provider-neutral `Engine`** — streaming `synthesize()` + buffered
  `synthesizeToPcm()`, `listVoices()`, `AbortSignal` cancellation — over an
  injectable `SpeechModel` seam (`web/src/engine/`).
- **Audio encoders** — raw 16-bit PCM, WAV, and MP3 (pure-JS lamejs), plus a
  band-limited windowed-sinc resampler for non-24 kHz output and a
  pitch-preserving WSOLA time-stretch for playback speed (`web/src/audio/`).
- **OpenAI & ElevenLabs drop-in adapters** over the engine, each defaulting to
  the provider's own default format (OpenAI `mp3`, ElevenLabs `mp3_44100_128`)
  so they are drop-in out of the box (`web/src/adapters/`).

### Browser-validated in-browser (WASM + WebGPU)

Each stage is gated by a real in-browser test (Playwright-driven `*.html`
harnesses), independent of the gated LM weights:

- **Phase 0 — decoder + vocoder** reproduce the PyTorch reference to float
  precision (mel ≈ 1e-5, wav ≈ 1e-7). Needed a real-valued (i)STFT rewrite of the
  vocoder (`conversion/hift_onnx.py`). `web/phase0.html`.
- **Phase 1 — LM decode loop.** The hand-rolled KV-cache decode loop
  (`web/src/pipeline/lm.ts`) is proven by unit tests + a genuine **toy LM**
  (`conversion/gen_toy_lm.py`) built to the real export contract: 30/30 golden
  ids on both backends (`web/phase1.html`). The gated real export is written to
  that contract (`conversion/export_lm.py`).
- **Phase 5 — voice cloning.** The Kanade clone encoder runs under ORT-Web
  reproducing the torch embedding (cosine 1.0) on both backends
  (`web/phase5.html`). `Engine.cloneVoice()` + ElevenLabs `voices.add` mapping.
- **Phase 4 — performance.** Decoder+vocoder real-time factor ≈ **3.3× on WASM**
  and **18.6× on WebGPU** for ~2 s of audio (`web/bench.html`).

### The one remaining human step (gated weights)

The Plapre LM weights live in `syvai/plapre-pico`, which is license-gated. The
browser loop and the export scripts are ready; producing the real LM artifacts
(and the end-to-end Danish audio + quality A/B) needs:

```bash
huggingface-cli login   # after clicking "Agree" on https://huggingface.co/syvai/plapre-pico
cd conversion && source .venv/bin/activate
python fetch_tokenizer.py && python export_lm.py && python precompute_speakers.py && python smoke_reference.py
```

The model files in `web/public/models/` are produced by the conversion scripts
and are not checked in. See [docs/PLAN.md](docs/PLAN.md).

## Getting started

```bash
# Web app + tests (run today; the pipeline reports which model files are missing)
cd web
npm install
npm test        # vitest: normalization, num2words, sampling, engine, adapters, LM loop, cloning, cache
npm run dev     # serves the harnesses: phase0/phase1/phase5/bench .html

# Model conversion (Python). Public stages need no credentials:
cd conversion
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python export_kanade_decoder.py && python export_hift_vocoder.py && python gen_phase0_golden.py  # Phase 0
python gen_toy_lm.py           # Phase 1 browser gate (toy LM)
python export_clone_encoder.py # Phase 5 clone encoder
# then open web/phase0.html / phase1.html / phase5.html (and ?backend=webgpu).
# Gated Phase 1 LM export + full recipe: conversion/README.md
```

## License / attribution

Plapre is released under **CC-BY** by [syv.ai](https://syv.ai/produkter/plapre)
([model](https://huggingface.co/syvai/plapre-pico),
[code](https://github.com/syv-ai/plapre)). The Kanade tokenizer is from
[frothywater/kanade-tokenizer](https://github.com/frothywater/kanade-tokenizer).
Converted ONNX weights redistributed from this project must keep attribution.
