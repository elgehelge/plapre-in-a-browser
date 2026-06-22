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

export interface CloneVoiceOptions {
  /** Stable id for the cloned voice; auto-generated if omitted. */
  readonly id?: string;
  readonly displayName?: string;
  readonly lang?: string;
}

/**
 * Optional capability: derive a new voice from reference audio, fully locally.
 * Models that support cloning (PlapreSpeechModel) implement this; the Engine
 * exposes it and reports `CloningUnsupportedError` for models that don't.
 */
export interface VoiceCloner {
  cloneVoice(
    audio: Float32Array,
    sampleRate: number,
    opts?: CloneVoiceOptions,
  ): Promise<Voice>;
  /**
   * Optionally pre-load the cloning weights (e.g. the clone encoder) ahead of
   * the first {@link cloneVoice}, so that call is instant. Safe to call more
   * than once; a no-op if already loaded.
   */
  prepareCloning?(): Promise<void>;
}

export function supportsCloning(model: SpeechModel): model is SpeechModel & VoiceCloner {
  return typeof (model as Partial<VoiceCloner>).cloneVoice === "function";
}
