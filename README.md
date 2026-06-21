# plapre-in-a-browser

[![npm](https://img.shields.io/npm/v/plapre-in-a-browser.svg)](https://www.npmjs.com/package/plapre-in-a-browser)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/demo-GitHub%20Pages-blue.svg)](https://elgehelge.github.io/plapre-in-a-browser/)

**Danish text-to-speech that runs entirely in the browser.** No server, no cloud,
no API key — the speech is synthesized on the user's own machine via WebGPU (with
a WebAssembly fallback). It ships as a small TypeScript library with a
provider-neutral engine and **drop-in OpenAI / ElevenLabs adapters**, so you can
swap a hosted TTS API for local inference with a one-line change.

```ts
import { loadPlapreEngine } from "plapre-in-a-browser";

const engine = await loadPlapreEngine();
const { samples, sampleRate } = await engine.synthesizeToPcm({
  text: "Hej, hvordan har du det i dag?",
  voice: "ida",
});
// → mono Float32 PCM @ 24 kHz, ready to play or encode.
```

**[▶ Try the live demo](https://elgehelge.github.io/plapre-in-a-browser/)** — type
Danish, pick a voice, generate, download, or clone a voice from a clip. Everything
runs client-side; the model weights stream from Hugging Face on first use.

## What is Plapre, and what does this add?

[**Plapre**](https://syv.ai/produkter/plapre) is an open-source (CC BY 4.0) Danish
TTS model from [syv.ai](https://syv.ai/): natural Danish speech and voice cloning
from a short clip. Upstream it runs in **Python** (PyTorch / GGUF) on a CPU or GPU.

This project makes that same model run **in the browser**, with no Python and no
backend:

- **A real port, not a wrapper.** Plapre is a 3-stage neural pipeline; all three
  stages are converted to ONNX and the autoregressive generation loop is
  reimplemented in TypeScript (see [below](#how-it-works)). None of these had
  published web builds.
- **Verified faithful.** Each stage is checked in-browser against the PyTorch
  reference to float precision ([parity table](#verified-correctness)).
- **A clean library API.** A provider-neutral `Engine` plus OpenAI / ElevenLabs
  adapters, local voice cloning, streaming, cancellation, and WAV/MP3 encoding.
- **Private by design.** Audio (including voice-cloning reference clips) never
  leaves the device.

Typical uses: privacy-sensitive apps, offline/edge tools, browser extensions (a
Chrome MV3 offscreen-document TTS provider), and dropping local Danish TTS into an
app that already speaks the OpenAI or ElevenLabs API.

## Install & use

```bash
npm install plapre-in-a-browser
```

```ts
import { loadPlapreEngine } from "plapre-in-a-browser";

const engine = await loadPlapreEngine({
  backend: "webgpu",                 // auto-falls back to "wasm"
  // Where the converted ONNX artifacts are served from. Defaults to "/models".
  modelsBaseUrl: "https://huggingface.co/elgehelge/plapre-onnx-web/resolve/main",
  cache: { onProgress: (loaded, total) => console.log(loaded / total) },
});

for (const v of engine.listVoices()) console.log(v.id); // tor ida liv ask kaj

// Stream sentence-by-sentence, or buffer the whole utterance:
const pcm = await engine.synthesizeToPcm({ text: "Hej med dig.", voice: "tor" });
```

Drop-in adapters and local voice cloning:

```ts
import { createOpenAISpeech, createElevenLabsClient } from "plapre-in-a-browser";

const openai = createOpenAISpeech(engine);
const res = await openai.create({ model: "tts-1", input: "Hej!", voice: "alloy" });
const mp3 = await res.arrayBuffer();          // same shape as the OpenAI SDK

if (engine.canCloneVoice()) {
  const voice = await engine.cloneVoice(refSamples, 24000, { displayName: "Mit" });
  await engine.synthesizeToPcm({ text: "Min klonede stemme.", voice: voice.id });
}
```

Full library docs (adapters, formats, caching, and the cross-origin-isolation
requirement for the WASM backend) are in **[web/README.md](web/README.md)**; the
engine contract and adapter mapping are in [docs/INTERFACE.md](docs/INTERFACE.md).

## Features

- **Provider-neutral engine** — `synthesize()` (streaming `AsyncIterable`) +
  `synthesizeToPcm()` (buffered), `listVoices()`, `AbortSignal` cancellation,
  pitch-preserving playback speed.
- **Drop-in adapters** — OpenAI `audio.speech` and ElevenLabs `text-to-speech`,
  each defaulting to the provider's own default format so they work out of the box.
- **Voice cloning, locally** — derive a speaker from a reference clip in-browser;
  the ElevenLabs adapter maps Instant Voice Cloning onto it.
- **Audio output** — raw PCM, WAV, and MP3 (pure-JS, no native deps), a
  band-limited resampler for non-24 kHz output, and WSOLA time-stretch for speed.
- **Caching** — large weights are fetched cache-first (Cache API) for instant,
  offline reloads, with download progress.
- **Tested** — 120 unit tests plus real in-browser parity gates on WASM + WebGPU.

## How it works

Plapre is **not** a single model — it is a 3-stage autoregressive LLM + neural
codec pipeline:

```
Danish text
  │  (1) normalize + BPE tokenize → prompt: [<text>] ids [<audio>]
  ▼
Plapre LM (SmolLM2 / LLaMA, ~118M)        speaker identity is injected as the
  │  (2) autoregressively sample           FIRST input embedding (128-dim vec
  │      ~25 audio tokens / second         → Linear → prepended to inputs_embeds)
  ▼
Kanade decoder   (3a) audio tokens + speaker emb → mel spectrogram
  ▼
HiFT vocoder     (3b) mel → 24 kHz waveform
  ▼
PCM audio (24 kHz, mono, float32)
```

| Component         | Role                                | Size (Pico) |
| ----------------- | ----------------------------------- | ----------- |
| Plapre LM         | SmolLM2/LLaMA, autoregressive       | ~500 MB f32 |
| Kanade decoder    | audio tokens + speaker emb → mel    | ~367 MB     |
| HiFT vocoder      | mel → 24 kHz waveform               | ~85 MB      |
| Kanade clone enc. | reference audio → 128-dim speaker   | ~174 MB     |

Built-in voices (`tor, ida, liv, ask, kaj`) ship as precomputed embeddings, so
only the decoder is needed for them; the clone encoder is loaded on demand.

### Making it run as ONNX in the browser

Most of the work was getting these models to export and run correctly under
`onnxruntime-web`. The notable hurdles and fixes:

- **Autoregressive loop in JS.** There is no off-the-shelf generate loop for ORT
  Web, so the KV-cache decode loop (temperature / top-k / top-p, seeded RNG) is
  hand-rolled and validated bit-for-bit against the reference.
- **Complex-tensor RoPE (decoder).** Kanade's rotary embeddings use complex
  tensors, which the legacy TorchScript exporter rejects. The TorchDynamo exporter
  decomposes them into real ops; inference-time attention dropout is zeroed for
  determinism.
- **STFT in the vocoder.** HiFT uses `torch.stft`/`istft` on complex tensors (no
  ONNX op) plus a random sine source. Both are rewritten with a real-valued
  (i)STFT and a deterministic source (`conversion/hift_onnx.py`).
- **Data-dependent control flow.** The LM (variadic KV-cache inputs) and the WavLM
  clone frontend export via the legacy exporter, where Dynamo trips.
- **External-data sidecars.** Large graphs split weights into `.onnx.data`, mounted
  explicitly through ORT-Web's `externalData`.
- **WebGPU fusion bug.** A `SkipLayerNormalization` fusion rejects our LayerNorm
  bias, so the WebGPU EP uses `graphOptimizationLevel: "basic"`.
- **Exact text frontend.** Danish normalization and `num2words(da)` are
  reimplemented to match the reference exactly (locked by a 2037-case fixture).

Every export script self-checks ORT-CPU parity vs PyTorch and refuses to write a
graph that drifts past tolerance.

### Verified correctness

Each stage is gated by a real in-browser test (Playwright-driven harnesses) that
compares against a PyTorch "golden" reference:

| Stage             | Check                                   | Result (WASM / WebGPU) |
| ----------------- | --------------------------------------- | ---------------------- |
| Kanade decoder    | mel `max\|diff\|` vs PyTorch            | ≈ 1.6e-5               |
| HiFT vocoder      | waveform `max\|diff\|` vs PyTorch       | ≈ 1.1e-7               |
| LM decode loop    | golden token ids (greedy)               | 30 / 30 exact          |
| Clone encoder     | speaker-embedding cosine vs PyTorch     | 1.000                  |

**Performance** (Pico, ~2 s utterance, **warm**, real-time factor = audio ÷ wall):

| Stage                          | WASM† | WebGPU |
| ------------------------------ | ----- | ------ |
| LM decode (autoregressive)     | 2.7×  | 1.1×   |
| decoder + vocoder              | 3.6×  | 18.6×  |
| **end-to-end**                 | ~1.5× | ~1.0×  |

The LM is the bottleneck, and because it runs **one token at a time** it benefits
less from WebGPU (per-dispatch overhead) than the fully parallel decoder/vocoder —
on this machine threaded WASM is actually faster for the LM. Measure your own with
`web/bench.html?backend=webgpu&iters=5` (`window.__bench`).

† WASM figures assume a **cross-origin-isolated** page (threaded SharedArrayBuffer);
single-threaded WASM is slower.

## Model artifacts

The library is **code only**; the converted ONNX weights (~1 GB) are hosted
separately and fetched at runtime from `modelsBaseUrl`. Three ways to get them:

1. **Hugging Face (default for the demo):**
   [`elgehelge/plapre-onnx-web`](https://huggingface.co/elgehelge/plapre-onnx-web)
   — CORS-enabled per-file fetch, the simplest option for the browser.
2. **Produce them yourself** with the conversion toolchain (`conversion/`); the
   decoder, vocoder, and clone encoder are public, the Plapre LM is license-gated
   on Hugging Face.
3. **A GitHub Release bundle** for self-hosting (`scripts/models.sh`).

> The Plapre weights are CC BY 4.0, so the converted artifacts are redistributed
> here under the same license with attribution (see [NOTICE](NOTICE)).

## Repository layout

```
web/          Publishable TypeScript library (web/src/index.ts) + the interactive
              demo (web/src/demo/), running on onnxruntime-web. Engine, adapters,
              audio, cloning, caching — implemented and tested.
conversion/   Python (uv): export the 3 models to ONNX, precompute speaker
              embeddings, and produce golden reference outputs. prepare_artifacts.py
              runs the stages in dependency order.
scripts/      models.sh (GitHub-Release bundle) + hf-model-card.md.
docs/         INTERFACE.md (engine + adapter contract), ARCHITECTURE.md (data flow),
              PLAN.md (build log + recorded numbers), EXTENSION.md (Chrome MV3).
LICENSE       CC BY 4.0 (matches upstream Plapre). NOTICE — third-party attribution.
```

## Getting started (development)

```bash
# Library + demo + tests
npm run setup       # installs web/ deps
npm test            # 120 vitest unit tests
npm run dev         # demo at http://localhost:5173 (uses local /models)
npm run build:lib   # build the publishable library (dist/ + types)

# Produce the model artifacts (Python via uv)
cd conversion
uv sync
uv run python prepare_artifacts.py            # public stages (decoder, vocoder, clone)
uv run python prepare_artifacts.py --gated    # + Plapre LM (needs `hf auth login`)
```

`prepare_artifacts.py --list` shows every stage; the full recipe (incl. gated
weights) is in [conversion/README.md](conversion/README.md).

## License & attribution

Licensed under **CC BY 4.0** ([LICENSE](LICENSE)), matching upstream
[Plapre](https://syv.ai/produkter/plapre) ([model](https://huggingface.co/syvai/plapre-pico),
[code](https://github.com/syv-ai/plapre)) by [syv.ai](https://syv.ai/). Use, modify,
and redistribute it — including commercially — with appropriate credit.

[NOTICE](NOTICE) lists every incorporated work, including the
[Kanade tokenizer](https://github.com/frothywater/kanade-tokenizer), the HiFT
vocoder (from CosyVoice 2), WavLM, and the bundled `lamejs` MP3 encoder.
Redistributed ONNX artifacts must keep that attribution.
