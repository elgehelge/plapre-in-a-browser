# Interface

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
  backend?: Backend;      // "webgpu" | "wasm"
  onProgress?: (p: { stage: string; loaded: number; total: number }) => void;
}

interface Engine {
  listVoices(): Voice[];
  synthesize(req: SynthesisRequest): AsyncIterable<PcmChunk>;   // streaming core
  synthesizeToPcm(                                              // buffered convenience
    req: SynthesisRequest,
  ): Promise<{ samples: Float32Array; sampleRate: number }>;
}
```

`GenerateOptions` is the existing engine-native sampling contract
(`temperature`, `topK`, `topP`, `maxTokens`, `seed`) from `web/src/pipeline`.
The current `Plapre` class in `plapre.ts` stays as the internal primitive; the
streaming `Engine` is a small wrapper over it.

### Timing / boundaries (optional capability)

We synthesize per sentence at a known 25 audio-tokens/second, so `startSec` per
chunk is cheap to produce. This optionally backs ElevenLabs timestamped streaming
and `chrome.ttsEngine` boundary events. It is **not** required for OpenAI /
ElevenLabs audio parity.

## Adapters

Each adapter is a thin mapping: translate the provider's request → `SynthesisRequest`,
map the provider's voice id → an engine voice id, run the PCM stream through a
format encoder, and shape the response like the provider's SDK. Encoding
(`mp3`/`opus`/`wav`/`pcm`) is an adapter concern; PCM passes through unencoded.

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
//                         rate: voice_settings?.speed,
//                         generation: fromVoiceSettings(voice_settings) })
//   → encode(output_format ?? "mp3_44100_128")   // "pcm_24000" passes through
// .stream(...) yields encoded chunks straight off the iterable.
```

`fromVoiceSettings` maps the subset of ElevenLabs knobs that have an engine
analogue (e.g. expressiveness → `temperature`); unmapped settings are ignored.
