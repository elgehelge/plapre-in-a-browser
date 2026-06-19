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
| Plapre LM        | SmolLM2/LLaMA, autoregressive      | 135M (q8 ≈ 121 MB) | to convert |
| Kanade decoder   | tokens + speaker emb → mel         | part of ~140M codec| to convert |
| HiFT vocoder     | mel → 24 kHz waveform              | small              | to convert (highest ONNX risk) |

We only need the **decoder** side of Kanade. The built-in speakers
(`tor, ida, liv, ask, kaj`) ship as precomputed embeddings in the model repo, so the
Kanade *encoder* / WavLM frontend (needed only for voice cloning) is **out of scope**.

## Repository layout

```
conversion/   Python: export the 3 models to ONNX, precompute speaker embeddings,
              and produce "golden" reference outputs for parity testing.
web/          Vite + TypeScript app running the pipeline on onnxruntime-web.
              Pure-JS, testable-now pieces (text normalization, Danish number words,
              sampling) are implemented; ONNX-dependent stages are wired with clear
              interfaces and load model files from web/public/models/.
docs/         PLAN.md (phased plan + risks), ARCHITECTURE.md (data flow detail),
              and INTERFACE.md (public engine contract + OpenAI/ElevenLabs adapters).
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
  Catmull-Rom resampler for non-24 kHz output (`web/src/audio/`).
- **OpenAI & ElevenLabs drop-in adapters** over the engine, each defaulting to
  the provider's own default format (OpenAI `mp3`, ElevenLabs `mp3_44100_128`)
  so they are drop-in out of the box (`web/src/adapters/`).

### Phase 0 cleared — decoder + vocoder run in the browser

The make-or-break gate is done: the **Kanade decoder** and **HiFT vocoder** both
export to ONNX and run under `onnxruntime-web` on **WASM and WebGPU**,
reproducing the PyTorch reference to float precision (mel ≈ 1e-5, wav ≈ 1e-7).
This required a real-valued (i)STFT rewrite of the vocoder (`conversion/
hift_onnx.py`) since HiFT's `torch.istft`/complex ops have no ONNX support. See
[docs/PLAN.md](docs/PLAN.md) for the full gate write-up (and the WebGPU
`SkipLayerNormalization` caveat). Reproduce with `conversion/
gen_phase0_golden.py` + `web/phase0.html`.

### Not yet working (Phase 1+)

The **LM generation loop** (text → audio tokens) is the remaining core piece;
its export scaffolding lives in `conversion/export_lm.py`. The model files in
`web/public/models/` are produced by the Python conversion scripts and are not
checked in. See [docs/PLAN.md](docs/PLAN.md).

## Getting started

```bash
# Web app (runs today; pipeline reports which model files are still missing)
cd web
npm install
npm run dev
npm test        # vitest: normalization, num2words, sampling, engine, adapters

# Model conversion (Python; produces the ONNX files the web app needs)
cd conversion
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python export_kanade_decoder.py && python export_hift_vocoder.py && python gen_phase0_golden.py
# then open web/phase0.html (and ?backend=webgpu) to verify in-browser parity.
# Full recipe + what "slim"/"deterministic" mean: conversion/README.md
```

## License / attribution

Plapre is released under **CC-BY** by [syv.ai](https://syv.ai/produkter/plapre)
([model](https://huggingface.co/syvai/plapre-pico),
[code](https://github.com/syv-ai/plapre)). The Kanade tokenizer is from
[frothywater/kanade-tokenizer](https://github.com/frothywater/kanade-tokenizer).
Converted ONNX weights redistributed from this project must keep attribution.
