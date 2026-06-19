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
- **Audio encoders** — raw 16-bit PCM and WAV (`web/src/audio/`).
- **OpenAI & ElevenLabs drop-in adapters** over the engine (`web/src/adapters/`).

### Not yet working (needs the converted models)

The ONNX-dependent stages (LM generation loop, Kanade decoder, HiFT vocoder)
load model files from `web/public/models/`, which don't exist until the Python
conversion runs. The first such milestone is **Phase 0: de-risk the vocoder** —
get the HiFT vocoder + Kanade decoder running under `onnxruntime-web` and
reproduce a known sentence's audio in the browser. If the vocoder will not
export/run on the web runtime, the whole approach needs rethinking, so this is
gated first. See [docs/PLAN.md](docs/PLAN.md).

## Getting started

```bash
# Web app (runs today; pipeline reports which model files are still missing)
cd web
npm install
npm run dev
npm test        # vitest: normalization, num2words, sampling, engine, adapters

# Model conversion (Python; produces the ONNX files the web app needs)
cd conversion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# then follow conversion/README.md
```

## License / attribution

Plapre is released under **CC-BY** by [syv.ai](https://syv.ai/produkter/plapre)
([model](https://huggingface.co/syvai/plapre-pico),
[code](https://github.com/syv-ai/plapre)). The Kanade tokenizer is from
[frothywater/kanade-tokenizer](https://github.com/frothywater/kanade-tokenizer).
Converted ONNX weights redistributed from this project must keep attribution.
