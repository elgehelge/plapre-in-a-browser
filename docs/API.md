# API

The public contract of the in-browser TTS module. The goal is a single
**provider-neutral engine** that emits canonical PCM, plus thin **adapters** that
make it a drop-in replacement for the OpenAI and ElevenLabs text-to-speech APIs.

Design rules:

- The engine emits **canonical audio**: mono `Float32Array` PCM at 24 kHz (the
  model's native rate). It never plays audio and never encodes container/codec
  formats — consumers own playback, adapters own format encoding.
- One **streaming** core (`AsyncIterable`) serves both streaming and buffered
  callers; buffered = collect the stream.
- Anything provider-specific (voice-id naming, format encoding, request/response
  shapes, auth) lives in an adapter, never in the engine.

## Engine (core)

```ts
const NATIVE_SAMPLE_RATE = 24000; // mono float32, the model's native rate

interface Voice {
  id: string;          // engine voice id, e.g. "ida"
  displayName: string;
  lang: string;        // BCP-47, e.g. "da-DK"
}

interface SynthesisRequest {
  text: string;
  voice: string;                          // engine voice id; adapters map their own ids → this
  rate?: number;                          // speed multiplier, 0.25..4, default 1
  signal?: AbortSignal;                   // cancellation
  generation?: Partial<GenerateOptions>;  // temperature / topK / topP / seed / maxTokens
}

interface PcmChunk {
  samples: Float32Array;  // mono
  sampleRate: number;     // == NATIVE_SAMPLE_RATE
  startSec: number;       // offset of this chunk within the utterance
}

interface LoadOptions {
  // Per-stage backend, each "webgpu" | "wasm" | "auto". "auto" (default) picks
  // LM→threaded-WASM and decoder+vocoder→WebGPU, with graceful fallback.
  backend_lm?: BackendChoice;     // autoregressive LM
  backend_codec?: BackendChoice;  // decoder + vocoder + clone encoder
  onProgress?: (p: { stage: string; loaded: number; total: number }) => void;
}

interface Engine {
  listVoices(): Voice[];
  synthesize(req: SynthesisRequest): AsyncIterable<PcmChunk>;   // streaming core
  synthesizeToPcm(                                              // buffered convenience
    req: SynthesisRequest,
  ): Promise<{ samples: Float32Array; sampleRate: number }>;
  canCloneVoice(): boolean;                                     // capability probe
  cloneVoice(                                                   // see Phase 5
    audio: Float32Array,
    sampleRate: number,
    opts?: { id?: string; displayName?: string; lang?: string },
  ): Promise<Voice>;                                            // CloningUnsupportedError if !canCloneVoice
}
```

Model loading + caching is configured on the concrete loader, not the engine:
`loadPlapreEngine({ model?, backend_lm?, backend_codec?, modelsBaseUrl?, generation?, cache?: { cacheName?, onProgress? } })`,
where `onProgress(loaded, total)` reports the cache-first model download. `model`
selects the variant (`"pico"` default | `"nano"`); it controls which LM-side
artifacts are fetched (the Kanade decoder/vocoder are shared) — see
`PLAPRE_MODELS` in `web/src/pipeline/models.ts`.

`GenerateOptions` is the existing engine-native sampling contract
(`temperature`, `topK`, `topP`, `maxTokens`, `seed`) from `web/src/pipeline`.
The current `Plapre` class in `plapre.ts` stays as the internal primitive; the
streaming `Engine` is a small wrapper over it.

### Timing / boundaries (optional capability)

We synthesize per sentence at a known 25 audio-tokens/second, so `startSec` per
chunk is cheap to produce. This optionally backs ElevenLabs timestamped streaming
and `chrome.ttsEngine` boundary events. It is **not** required for OpenAI /
ElevenLabs audio parity.

### Voice cloning (optional)

`cloneVoice` derives the same 128-dim speaker embedding the built-in voices ship
as, from arbitrary reference audio, and registers a `Voice` usable in
`synthesize()` like any other. It runs fully local — the audio never leaves the
browser. See [docs/ARCHITECTURE.md](ARCHITECTURE.md) for the encoder pipeline; the
ElevenLabs adapter maps Instant Voice Cloning onto it, OpenAI has no analogue.

## Adapters

Each adapter is a thin mapping: translate the provider's request → `SynthesisRequest`,
map the provider's voice id → an engine voice id, run the PCM stream through a
format encoder, and shape the response like the provider's SDK. Encoding is an
adapter concern; PCM passes through unencoded.

**Supported output formats** (each provider's *default* works out of the box):

- OpenAI `response_format`: `mp3` (default), `wav`, `pcm`. `opus`/`aac`/`flac`
  are a typed gap (`UnsupportedFormatError`) — they need a heavier codec.
- ElevenLabs `output_format`: `mp3_{22050,44100}_{kbps}` (default
  `mp3_44100_128`) and `pcm_{8000,16000,22050,24000,44100}`. Non-24 kHz rates are
  produced by resampling the engine's 24 kHz PCM (`audio/resample.ts`,
  band-limited windowed-sinc / Lanczos with anti-aliasing). `ulaw_*` is a gap.

MP3 is encoded with the pure-JS lamejs encoder (no native deps; runs in the
browser / a worker / an MV3 extension). Only native 24 kHz raw PCM streams
chunk-by-chunk; resampled / MP3 output is buffered then encoded.

### Common mapping

| Concern    | OpenAI `/v1/audio/speech` | ElevenLabs `/v1/text-to-speech/{id}` | Engine                       |
| ---------- | ------------------------- | ------------------------------------ | ---------------------------- |
| text       | `input`                   | `text`                               | `text`                       |
| voice      | `voice`                   | `{voice_id}` + `GET /v1/voices`      | `voice` + `listVoices()`     |
| speed      | `speed` (0.25–4)          | `voice_settings.speed`               | `rate`                       |
| streaming  | chunked / `stream_format` | `/stream` endpoint                   | `synthesize()` iterable      |
| buffered   | default                   | `convert`                            | `synthesizeToPcm()`          |
| out format | `response_format`         | `output_format`                      | adapter-side encoder         |
| model knobs| `instructions`            | `voice_settings` (stability, …)      | `generation`                 |
| cancel     | HTTP abort                | HTTP abort                           | `signal`                     |

Both providers natively support raw 24 kHz mono PCM (OpenAI `pcm`, ElevenLabs
`pcm_24000`), which matches `NATIVE_SAMPLE_RATE` exactly — so the PCM path needs
no resampling, only the lossy formats need an encoder.

### OpenAI adapter

```ts
// openai.audio.speech.create({ model, input, voice, speed?, response_format? })
//   → engine.synthesize({ text: input, voice: mapVoice(voice), rate: speed })
//   → encode(response_format ?? "mp3")   // "pcm" passes through
```

### ElevenLabs adapter

```ts
// elevenlabs.textToSpeech.convert(voiceId, { text, voice_settings?, output_format? })
//   → engine.synthesize({ text, voice: mapVoice(voiceId),
//                         rate: voice_settings?.speed })
//   → encode(output_format ?? "mp3_44100_128")   // "pcm_24000" passes through
// .stream(...) yields encoded chunks straight off the iterable.
```

Only `voice_settings.speed` has an engine analogue (`rate`). The other
ElevenLabs knobs (`stability`, `similarityBoost`, `style`, …) are accepted for
API compatibility but **not applied** — the local model has no analogue. Neither
wire protocol carries a sampling **temperature**, so to tune it under an adapter
set `generation` once at engine construction; see [`TUNING.md`](./TUNING.md).

#### Instant Voice Cloning

`createElevenLabsVoices(engine)` / `createElevenLabsClient(engine).voices` map
`voices.add({ name, files })` onto `engine.cloneVoice()`: each file is decoded
(via a pluggable `AudioDecoder`, WebAudio by default) and the clips are
concatenated into the reference waveform. Returns `{ voiceId, name }`; the id is
then usable as a normal `voice` in `convert`/`stream`. OpenAI has no cloning
analogue.
