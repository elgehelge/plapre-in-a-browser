# plapre-in-a-browser

Open-source **Danish text-to-speech** ([Plapre](https://huggingface.co/syvai/plapre-pico))
running **fully in the browser** — no server, no cloud, no API key. Inference runs
on [`onnxruntime-web`](https://onnxruntime.ai/docs/tutorials/web/) (WebGPU, with a
WASM fallback).

It ships a **provider-neutral engine** plus thin **drop-in adapters** for the
OpenAI (`audio.speech`) and ElevenLabs (`text-to-speech`) APIs, and supports
**local voice cloning** from a reference clip.

> Full project (model conversion toolchain, architecture, phased plan):
> https://github.com/elgehelge/plapre-in-a-browser

## Install

```bash
npm install plapre-in-a-browser
```

`onnxruntime-web`, `@huggingface/transformers`, and `@breezystack/lamejs` are
installed as dependencies and left external in the bundle, so your bundler
controls and dedupes them.

## Model artifacts

The library is code only — the converted ONNX models (~0.5 GB) are **not**
bundled. You host them yourself and point the engine at them with
`modelsBaseUrl` (default `/models`, i.e. served next to your app).

Get the artifacts one of three ways:

- **Fetch from Hugging Face (easiest).** A prebuilt set is hosted at
  [`elgehelge/plapre-onnx-web`](https://huggingface.co/elgehelge/plapre-onnx-web)
  with CORS-enabled per-file access:

  ```ts
  const engine = await loadPlapreEngine({
    modelsBaseUrl: "https://huggingface.co/elgehelge/plapre-onnx-web/resolve/main",
  });
  ```

- **Download a bundle** from
  [GitHub Releases](https://github.com/elgehelge/plapre-in-a-browser/releases)
  and serve its contents (preserving the directory layout) from your
  `modelsBaseUrl`.
- **Produce them yourself** with the Python conversion toolchain (`conversion/`).
  The Plapre language-model weights are license-gated on Hugging Face; the decoder,
  vocoder, and clone encoder are public.

Probe what's reachable before loading:

```ts
import { setModelsBaseUrl, reportArtifacts } from "plapre-in-a-browser";

setModelsBaseUrl("https://huggingface.co/elgehelge/plapre-onnx-web/resolve/main");
console.log(await reportArtifacts()); // { lm: true, kanadeDecoder: true, ... }
```

## Quickstart

```ts
import { loadPlapreEngine } from "plapre-in-a-browser";

const engine = await loadPlapreEngine({
  backend_lm: "auto",                // "auto" | "webgpu" | "wasm"  (auto → threaded WASM)
  backend_codec: "auto",             // "auto" | "webgpu" | "wasm"  (auto → WebGPU)
  modelsBaseUrl: "/models",          // where you host the artifacts
  cache: { onProgress: (loaded, total) => console.log(loaded / total) },
});

const { samples, sampleRate } = await engine.synthesizeToPcm({
  text: "Hej, hvordan har du det i dag?",
  voice: "ida",                      // tor | ida | liv | ask | kaj
});

// `samples` is mono Float32Array @ 24 kHz — play it however you like.
```

Stream sentence-by-sentence instead of buffering:

```ts
for await (const chunk of engine.synthesize({ text, voice: "tor" })) {
  // chunk.samples (Float32Array), chunk.sampleRate (24000), chunk.startSec
}
```

`listVoices()`, an `AbortSignal` (`request.signal`), pitch-preserving playback
speed (`request.rate`), and sampling knobs (`request.generation`) are all
supported. See [docs/INTERFACE.md](https://github.com/elgehelge/plapre-in-a-browser/blob/main/docs/INTERFACE.md).

## Drop-in API adapters

Swap a hosted TTS API for local inference with the same request shapes:

```ts
import { loadPlapreEngine, createOpenAISpeech, createElevenLabsClient } from "plapre-in-a-browser";

const engine = await loadPlapreEngine();

// OpenAI: mirrors `openai.audio.speech.create(...)`, returns a web Response.
const openai = createOpenAISpeech(engine);
const res = await openai.create({ model: "tts-1", input: "Hej!", voice: "alloy" });
const mp3 = await res.arrayBuffer();

// ElevenLabs: mirrors `client.textToSpeech.convert(voiceId, {...})`.
const eleven = createElevenLabsClient(engine);
const audio = await (await eleven.textToSpeech.convert("ida", { text: "Hej!" })).arrayBuffer();
```

Each adapter returns a standard `Response` and defaults to its provider's own
default format (OpenAI `mp3`, ElevenLabs `mp3_44100_128`), so existing call sites
(`await res.arrayBuffer()` / streaming via `res.body`) work unchanged. MP3 is
encoded in pure JS; raw `pcm` passes through unencoded.

## Voice cloning (local)

```ts
if (engine.canCloneVoice()) {
  const voice = await engine.cloneVoice(referenceSamples, 24000, { displayName: "My voice" });
  const out = await engine.synthesizeToPcm({ text: "Hej fra min klonede stemme.", voice: voice.id });
}
```

The reference audio never leaves the browser. The ElevenLabs adapter maps
`voices.add({ name, files })` onto this.

## Cross-origin isolation

Threaded WASM (the fallback backend) needs `SharedArrayBuffer`, which requires
your page to be **cross-origin isolated**. Serve it with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

WebGPU does not require this, but the WASM fallback does. In a Chrome MV3
extension, mirror these in the manifest (see
[docs/EXTENSION.md](https://github.com/elgehelge/plapre-in-a-browser/blob/main/docs/EXTENSION.md)).

## License

[CC BY 4.0](https://github.com/elgehelge/plapre-in-a-browser/blob/main/LICENSE),
matching upstream Plapre. See
[NOTICE](https://github.com/elgehelge/plapre-in-a-browser/blob/main/NOTICE) for
attribution of all incorporated works.
