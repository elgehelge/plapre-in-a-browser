// Drop-in replacement for the OpenAI text-to-speech API (audio.speech.create),
// backed by the local engine. Mirrors the request body and returns a web
// Response, so existing call sites — `await client.audio.speech.create(body)`
// then `response.arrayBuffer()` / streaming via `response.body` — work unchanged
// against local, server-free synthesis. See docs/API.md.

import type { Engine, SynthesisRequest } from "../engine/engine.js";
import { type AudioFormat, encodeAudio, UnsupportedFormatError } from "../audio/format.js";
import { mapProviderVoice, type VoiceMap } from "./voice-map.js";
import { pcmStream } from "./stream.js";
import { UnsupportedSpeedError } from "./errors.js";

// OpenAI voice names mapped onto the built-in Danish speakers. Override per
// deployment via OpenAISpeechOptions.voiceMap; unknown names pass through and
// are validated against the engine (so a native speaker id also works).
const DEFAULT_VOICE_MAP: VoiceMap = {
  alloy: "ida",
  ash: "tor",
  ballad: "ask",
  coral: "liv",
  echo: "tor",
  fable: "ask",
  onyx: "kaj",
  nova: "liv",
  sage: "kaj",
  shimmer: "ida",
  verse: "tor",
};

const FORMATS: Record<string, AudioFormat> = { pcm: "pcm", wav: "wav", mp3: "mp3" };
const CONTENT_TYPE: Record<AudioFormat, string> = {
  pcm: "audio/pcm",
  wav: "audio/wav",
  mp3: "audio/mpeg",
};

export interface OpenAISpeechRequest {
  /** Accepted for compatibility; the local engine has a single model. */
  model?: string;
  voice: string;
  input: string;
  /** OpenAI's default; mp3/wav/pcm supported (opus/aac/flac are a typed gap). */
  response_format?: string;
  /** Playback speed (pitch-preserving). OpenAI range 0.25–4.0; default 1. */
  speed?: number;
}

export interface OpenAISpeechRequestOptions {
  signal?: AbortSignal;
}

export interface OpenAISpeechOptions {
  voiceMap?: VoiceMap;
}

export interface OpenAISpeechAdapter {
  create(req: OpenAISpeechRequest, options?: OpenAISpeechRequestOptions): Promise<Response>;
}

export function createOpenAISpeech(engine: Engine, options: OpenAISpeechOptions = {}): OpenAISpeechAdapter {
  const voiceMap = { ...DEFAULT_VOICE_MAP, ...options.voiceMap };

  return {
    async create(req, requestOptions = {}): Promise<Response> {
      const rate = req.speed ?? 1;
      if (rate < 0.25 || rate > 4) throw new UnsupportedSpeedError(rate, 0.25, 4);

      const format = FORMATS[req.response_format ?? "mp3"];
      if (!format) throw new UnsupportedFormatError(req.response_format ?? "");

      const voice = mapProviderVoice(engine.listVoices(), voiceMap, req.voice);
      const synthReq: SynthesisRequest = {
        text: req.input,
        voice,
        rate,
        signal: requestOptions.signal,
      };
      const headers = { "content-type": CONTENT_TYPE[format] };

      if (format === "pcm") {
        return new Response(pcmStream(engine, synthReq), { headers });
      }
      const { samples, sampleRate } = await engine.synthesizeToPcm(synthReq);
      return new Response(encodeAudio(samples, sampleRate, format), { headers });
    },
  };
}
