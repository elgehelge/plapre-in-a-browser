// Drop-in replacement for the ElevenLabs text-to-speech API
// (textToSpeech.convert / textToSpeech.stream), backed by the local engine.
// Mirrors the SDK's (voiceId, request) shape so existing call sites work
// unchanged against local, server-free synthesis. See docs/INTERFACE.md.

import { NATIVE_SAMPLE_RATE, type Engine, type SynthesisRequest } from "../engine/engine.js";
import { encodeMp3, pcmToInt16LE, UnsupportedFormatError } from "../audio/format.js";
import { resample } from "../audio/resample.js";
import { decodeWithWebAudio, type AudioDecoder, type DecodedAudio } from "../audio/decode.js";
import { mapProviderVoice, type VoiceMap } from "./voice-map.js";
import { pcmStream } from "./stream.js";
import { UnsupportedSpeedError } from "./errors.js";

// ElevenLabs output_format strings are `pcm_<rate>` or `mp3_<rate>_<kbps>`.
// We resample the engine's 24 kHz PCM to the requested rate (audio/resample.ts)
// and encode. Rates outside these sets, and other codecs (ulaw, …), are an
// explicit gap (UnsupportedFormatError).
const PCM_RATES = new Set([8000, 16000, 22050, 24000, 44100]);
const MP3_RATES = new Set([22050, 44100]);

interface OutputSpec {
  readonly codec: "pcm" | "mp3";
  readonly rate: number;
  readonly kbps?: number;
  readonly contentType: string;
}

function parseOutputFormat(raw: string): OutputSpec {
  const m = /^(pcm|mp3)_(\d+)(?:_(\d+))?$/.exec(raw);
  if (!m) throw new UnsupportedFormatError(raw);
  const codec = m[1] as "pcm" | "mp3";
  const rate = Number(m[2]);
  const kbps = m[3] !== undefined ? Number(m[3]) : undefined;

  if (codec === "pcm") {
    if (kbps !== undefined || !PCM_RATES.has(rate)) throw new UnsupportedFormatError(raw);
    return { codec, rate, contentType: "audio/pcm" };
  }
  if (kbps === undefined || !MP3_RATES.has(rate)) throw new UnsupportedFormatError(raw);
  return { codec, rate, kbps, contentType: "audio/mpeg" };
}

function render(pcm: Float32Array, spec: OutputSpec): Uint8Array<ArrayBuffer> {
  const resampled = resample(pcm, NATIVE_SAMPLE_RATE, spec.rate);
  return spec.codec === "mp3"
    ? encodeMp3(resampled, spec.rate, spec.kbps)
    : pcmToInt16LE(resampled);
}

export interface ElevenLabsVoiceSettings {
  /** Playback speed (pitch-preserving). ElevenLabs range 0.7–1.2; default 1. */
  speed?: number;
  // Accepted for compatibility but not applied — the local model has no analogue
  // for these ElevenLabs-specific knobs.
  stability?: number;
  similarityBoost?: number;
  style?: number;
  useSpeakerBoost?: boolean;
}

export interface ElevenLabsConvertRequest {
  text: string;
  /** Accepted for compatibility; the local engine has a single model. */
  modelId?: string;
  voiceSettings?: ElevenLabsVoiceSettings;
  /** ElevenLabs default, matched here: "mp3_44100_128". */
  outputFormat?: string;
}

export interface ElevenLabsRequestOptions {
  signal?: AbortSignal;
}

export interface ElevenLabsTextToSpeechAdapter {
  convert(
    voiceId: string,
    request: ElevenLabsConvertRequest,
    options?: ElevenLabsRequestOptions,
  ): Promise<Response>;
  stream(
    voiceId: string,
    request: ElevenLabsConvertRequest,
    options?: ElevenLabsRequestOptions,
  ): ReadableStream<Uint8Array>;
}

export interface ElevenLabsOptions {
  voiceMap?: VoiceMap;
  /** Audio decoder for voice cloning input; defaults to WebAudio. */
  decodeAudio?: AudioDecoder;
}

/** Reference audio for cloning: encoded bytes (decoded via the decoder) or PCM. */
export type CloneAudioInput = ArrayBuffer | Uint8Array | Blob | DecodedAudio;

export interface ElevenLabsAddVoiceRequest {
  name?: string;
  /** One or more reference clips (concatenated). Mirrors ElevenLabs `files`. */
  files: CloneAudioInput[];
  /** Accepted for compatibility; not used locally. */
  description?: string;
  labels?: Record<string, string>;
}

export interface ElevenLabsVoicesAdapter {
  /** ElevenLabs Instant Voice Cloning: POST /v1/voices/add -> local cloneVoice. */
  add(request: ElevenLabsAddVoiceRequest): Promise<{ voiceId: string; name: string }>;
}

/** A small mirror of the ElevenLabs client: `.textToSpeech` + `.voices`. */
export interface ElevenLabsClient {
  textToSpeech: ElevenLabsTextToSpeechAdapter;
  voices: ElevenLabsVoicesAdapter;
}

export function createElevenLabsTextToSpeech(
  engine: Engine,
  options: ElevenLabsOptions = {},
): ElevenLabsTextToSpeechAdapter {
  const voiceMap = { ...options.voiceMap };

  function plan(
    voiceId: string,
    request: ElevenLabsConvertRequest,
    options: ElevenLabsRequestOptions,
  ): { synthReq: SynthesisRequest; spec: OutputSpec } {
    const rate = request.voiceSettings?.speed ?? 1;
    if (rate < 0.7 || rate > 1.2) throw new UnsupportedSpeedError(rate, 0.7, 1.2);

    const spec = parseOutputFormat(request.outputFormat ?? "mp3_44100_128");
    const voice = mapProviderVoice(engine.listVoices(), voiceMap, voiceId);
    return { synthReq: { text: request.text, voice, rate, signal: options.signal }, spec };
  }

  // Native 24 kHz raw PCM can stream chunk-by-chunk; resampled / MP3 output
  // needs the whole utterance, so it is buffered then encoded.
  const isNativePcm = (spec: OutputSpec): boolean =>
    spec.codec === "pcm" && spec.rate === NATIVE_SAMPLE_RATE;

  async function renderResponse(
    synthReq: SynthesisRequest,
    spec: OutputSpec,
  ): Promise<Response> {
    const headers = { "content-type": spec.contentType };
    if (isNativePcm(spec)) return new Response(pcmStream(engine, synthReq), { headers });
    const { samples } = await engine.synthesizeToPcm(synthReq);
    return new Response(render(samples, spec), { headers });
  }

  return {
    async convert(voiceId, request, options = {}): Promise<Response> {
      const { synthReq, spec } = plan(voiceId, request, options);
      return renderResponse(synthReq, spec);
    },

    stream(voiceId, request, options = {}): ReadableStream<Uint8Array> {
      const { synthReq, spec } = plan(voiceId, request, options);
      if (isNativePcm(spec)) return pcmStream(engine, synthReq);
      return new ReadableStream<Uint8Array>({
        async start(controller) {
          try {
            const { samples } = await engine.synthesizeToPcm(synthReq);
            controller.enqueue(render(samples, spec));
            controller.close();
          } catch (err) {
            controller.error(err);
          }
        },
      });
    },
  };
}

function isDecoded(x: CloneAudioInput): x is DecodedAudio {
  return typeof x === "object" && x !== null && "pcm" in x && "sampleRate" in x;
}

async function toArrayBuffer(x: ArrayBuffer | Uint8Array | Blob): Promise<ArrayBuffer> {
  if (x instanceof ArrayBuffer) return x;
  if (x instanceof Uint8Array) {
    const out = new ArrayBuffer(x.byteLength);
    new Uint8Array(out).set(x);
    return out;
  }
  return x.arrayBuffer();
}

/**
 * ElevenLabs `voices` sub-client. Only Instant Voice Cloning is mapped (onto the
 * engine's local cloneVoice); the rest of the ElevenLabs voice CRUD has no local
 * analogue. Cloning runs fully in-browser — reference audio never leaves it.
 */
export function createElevenLabsVoices(
  engine: Engine,
  options: ElevenLabsOptions = {},
): ElevenLabsVoicesAdapter {
  const decodeAudio = options.decodeAudio ?? decodeWithWebAudio;

  return {
    async add(request): Promise<{ voiceId: string; name: string }> {
      if (!engine.canCloneVoice()) throw new UnsupportedFormatError("voice cloning");
      if (!request.files?.length) throw new Error("voices.add requires at least one file");

      const decoded: DecodedAudio[] = [];
      for (const file of request.files) {
        decoded.push(isDecoded(file) ? file : await decodeAudio(await toArrayBuffer(file)));
      }
      // Concatenate clips at a common rate (the first clip's) — more reference
      // audio yields a more robust embedding.
      const rate = decoded[0].sampleRate;
      const parts = decoded.map((d) =>
        d.sampleRate === rate ? d.pcm : resample(d.pcm, d.sampleRate, rate),
      );
      let total = 0;
      for (const p of parts) total += p.length;
      const merged = new Float32Array(total);
      let off = 0;
      for (const p of parts) {
        merged.set(p, off);
        off += p.length;
      }

      const voice = await engine.cloneVoice(merged, rate, { displayName: request.name });
      return { voiceId: voice.id, name: voice.displayName };
    },
  };
}

/** Mirror of the ElevenLabs client surface this adapter supports. */
export function createElevenLabsClient(
  engine: Engine,
  options: ElevenLabsOptions = {},
): ElevenLabsClient {
  return {
    textToSpeech: createElevenLabsTextToSpeech(engine, options),
    voices: createElevenLabsVoices(engine, options),
  };
}
