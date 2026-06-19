// The narrow seam the Engine orchestrates over. A SpeechModel turns a single
// already-normalized sentence into mono PCM for a resolved voice; it owns the
// model weights, the voice embeddings, and the runtime backend. Everything the
// Engine adds on top (sentence splitting, streaming, chunk timing, cancellation,
// voice resolution) is expressed in terms of this seam, which keeps that
// orchestration testable without a model runtime.

import type { GenerateOptions } from "../pipeline/lm.js";
import type { Voice } from "./voice.js";

export interface SentenceRequest {
  /** A single sentence, already run through text normalization. */
  readonly sentence: string;
  /** A voice resolved from this model's own catalog. */
  readonly voice: Voice;
  readonly generation: GenerateOptions;
  readonly signal?: AbortSignal;
}

export interface SpeechModel {
  /** The voices this model can synthesize. */
  voices(): readonly Voice[];
  /** Synthesize one sentence to mono PCM at the model's native sample rate. */
  synthesizeSentence(request: SentenceRequest): Promise<Float32Array>;
}
