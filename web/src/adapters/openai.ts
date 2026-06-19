// Drop-in replacement for the OpenAI text-to-speech API (audio.speech.create),
// backed by the local engine. Mirrors the request body and returns a web
// Response, so existing call sites — `await client.audio.speech.create(body)`
// then `response.arrayBuffer()` / streaming via `response.body` — work unchanged
// against local, server-free synthesis. See docs/INTERFACE.md.

import type { Engine, SynthesisRequest } from "../engine/engine.js";
import { type AudioFormat, encodeAudio, pcmToInt16LE, UnsupportedFormatError } from "../audio/format.js";
import { mapProviderVoice, type VoiceMap } from "./voice-map.js";
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

const FORMATS: Record<string, AudioFormat> = { pcm: "pcm", wav: "wav" };
const CONTENT_TYPE: Record<AudioFormat, string> = { pcm: "audio/pcm", wav: "audio/wav" };

export interface OpenAISpeechRequest {
  /** Accepted for compatibility; the local engine has a single model. */
  model?: string;
  voice: string;
  input: string;
  /** OpenAI default is "mp3"; here it defaults to "wav" until an mp3 encoder exists. */
  response_format?: string;
  /** Only 1 is supported (see UnsupportedSpeedError). */
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
      if (req.speed !== undefined && req.speed !== 1) throw new UnsupportedSpeedError(req.speed);

      const format = FORMATS[req.response_format ?? "wav"];
      if (!format) throw new UnsupportedFormatError(req.response_format ?? "");

      const voice = mapProviderVoice(engine.listVoices(), voiceMap, req.voice);
      const synthReq: SynthesisRequest = { text: req.input, voice, signal: requestOptions.signal };
      const headers = { "content-type": CONTENT_TYPE[format] };

      if (format === "pcm") {
        return new Response(pcmStream(engine, synthReq), { headers });
      }
      const { samples, sampleRate } = await engine.synthesizeToPcm(synthReq);
      return new Response(encodeAudio(samples, sampleRate, format), { headers });
    },
  };
}

/** Stream raw 16-bit PCM chunk-by-chunk as the engine produces them. */
function pcmStream(engine: Engine, req: SynthesisRequest): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const chunk of engine.synthesize(req)) {
          controller.enqueue(pcmToInt16LE(chunk.samples));
        }
        controller.close();
      } catch (err) {
        controller.error(err);
      }
    },
  });
}
