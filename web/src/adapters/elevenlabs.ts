// Drop-in replacement for the ElevenLabs text-to-speech API
// (textToSpeech.convert / textToSpeech.stream), backed by the local engine.
// Mirrors the SDK's (voiceId, request) shape so existing call sites work
// unchanged against local, server-free synthesis. See docs/INTERFACE.md.

import type { Engine, SynthesisRequest } from "../engine/engine.js";
import { type AudioFormat, UnsupportedFormatError } from "../audio/format.js";
import { mapProviderVoice, type VoiceMap } from "./voice-map.js";
import { pcmStream } from "./stream.js";
import { UnsupportedSpeedError } from "./errors.js";

// ElevenLabs output_format strings → engine formats. Only 24 kHz PCM matches the
// engine's native rate; other rates need resampling and lossy formats need an
// encoder, so they are an explicit gap (UnsupportedFormatError).
const FORMATS: Record<string, AudioFormat> = { pcm_24000: "pcm" };
const CONTENT_TYPE: Record<AudioFormat, string> = { pcm: "audio/pcm", wav: "audio/wav" };

export interface ElevenLabsVoiceSettings {
  /** Only 1 is supported (see UnsupportedSpeedError). */
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
  /** ElevenLabs default is "mp3_44100_128"; here it defaults to "pcm_24000". */
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
}

export function createElevenLabsTextToSpeech(
  engine: Engine,
  options: ElevenLabsOptions = {},
): ElevenLabsTextToSpeechAdapter {
  const voiceMap = { ...options.voiceMap };

  function buildRequest(
    voiceId: string,
    request: ElevenLabsConvertRequest,
    options: ElevenLabsRequestOptions,
  ): { synthReq: SynthesisRequest; format: AudioFormat } {
    const speed = request.voiceSettings?.speed;
    if (speed !== undefined && speed !== 1) throw new UnsupportedSpeedError(speed);

    const format = FORMATS[request.outputFormat ?? "pcm_24000"];
    if (!format) throw new UnsupportedFormatError(request.outputFormat ?? "");

    const voice = mapProviderVoice(engine.listVoices(), voiceMap, voiceId);
    return { synthReq: { text: request.text, voice, signal: options.signal }, format };
  }

  return {
    async convert(voiceId, request, options = {}): Promise<Response> {
      const { synthReq, format } = buildRequest(voiceId, request, options);
      return new Response(pcmStream(engine, synthReq), {
        headers: { "content-type": CONTENT_TYPE[format] },
      });
    },

    stream(voiceId, request, options = {}): ReadableStream<Uint8Array> {
      const { synthReq } = buildRequest(voiceId, request, options);
      return pcmStream(engine, synthReq);
    },
  };
}
