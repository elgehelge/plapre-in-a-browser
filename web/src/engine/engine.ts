// The provider-neutral synthesis engine. It exposes one streaming primitive
// (`synthesize`) and a buffered convenience over it (`synthesizeToPcm`), emitting
// canonical mono PCM at NATIVE_SAMPLE_RATE. Provider-specific concerns (OpenAI /
// ElevenLabs request shapes, format encoding) live in adapters above this; the
// model runtime lives in the SpeechModel below it. See docs/INTERFACE.md.

import { SAMPLE_RATE } from "../pipeline/types.js";
import { splitSentences } from "../pipeline/normalize.js";
import type { GenerateOptions } from "../pipeline/lm.js";
import type { SpeechModel } from "./speech-model.js";
import { resolveVoice, type Voice } from "./voice.js";

export const NATIVE_SAMPLE_RATE = SAMPLE_RATE;

export interface PcmChunk {
  readonly samples: Float32Array; // mono
  readonly sampleRate: number; // == NATIVE_SAMPLE_RATE
  readonly startSec: number; // offset of this chunk within the utterance
}

export interface Pcm {
  readonly samples: Float32Array;
  readonly sampleRate: number;
}

export interface SynthesisRequest {
  readonly text: string;
  readonly voice: string; // adapters map their own voice ids onto an engine voice id
  readonly signal?: AbortSignal;
  readonly generation?: Partial<GenerateOptions>;
}

export interface Engine {
  listVoices(): readonly Voice[];
  synthesize(request: SynthesisRequest): AsyncIterable<PcmChunk>;
  synthesizeToPcm(request: SynthesisRequest): Promise<Pcm>;
}

// The reference pipeline's sampling defaults (plapre/inference.py). Expressed
// here as the engine's intent for an unconfigured request; callers override per
// engine (EngineOptions) or per request (SynthesisRequest.generation).
export const DEFAULT_GENERATION: GenerateOptions = {
  temperature: 0.8,
  topK: 50,
  topP: 0.95,
  maxTokens: 500,
  seed: 0,
};

export interface EngineOptions {
  readonly generation?: Partial<GenerateOptions>;
}

export function createEngine(model: SpeechModel, options: EngineOptions = {}): Engine {
  const baseGeneration: GenerateOptions = { ...DEFAULT_GENERATION, ...options.generation };

  async function* synthesize(request: SynthesisRequest): AsyncIterable<PcmChunk> {
    const voice = resolveVoice(model.voices(), request.voice);
    const generation: GenerateOptions = { ...baseGeneration, ...request.generation };

    let startSec = 0;
    for (const sentence of splitSentences(request.text)) {
      request.signal?.throwIfAborted();
      const samples = await model.synthesizeSentence({
        sentence,
        voice,
        generation,
        signal: request.signal,
      });
      if (samples.length === 0) continue; // a sentence may yield no audio tokens
      yield { samples, sampleRate: NATIVE_SAMPLE_RATE, startSec };
      startSec += samples.length / NATIVE_SAMPLE_RATE;
    }
  }

  async function synthesizeToPcm(request: SynthesisRequest): Promise<Pcm> {
    const chunks: Float32Array[] = [];
    for await (const chunk of synthesize(request)) chunks.push(chunk.samples);
    return { samples: concat(chunks), sampleRate: NATIVE_SAMPLE_RATE };
  }

  return {
    listVoices: () => model.voices(),
    synthesize,
    synthesizeToPcm,
  };
}

function concat(chunks: readonly Float32Array[]): Float32Array {
  let total = 0;
  for (const c of chunks) total += c.length;
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}
