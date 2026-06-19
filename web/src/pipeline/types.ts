export type Backend = "webgpu" | "wasm";

export interface SpeakerData {
  /** Raw 128-dim Kanade global embedding (used by the decoder). */
  raw: number[];
  /** Precomputed speaker_proj(raw) hidden vector (prepended to LM inputs). */
  hidden: number[];
}

export type SpeakerTable = Record<string, SpeakerData>;

/** Audio constants from the reference pipeline. */
export const SAMPLE_RATE = 24000;
export const AUDIO_CODEBOOK = 12800; // <audio_0> .. <audio_12799>

/** PCM result: mono float32 at SAMPLE_RATE. */
export interface SynthResult {
  pcm: Float32Array;
  sampleRate: number;
}

/** Thrown when a required converted model artifact is not present yet. */
export class MissingModelError extends Error {
  constructor(
    public readonly artifact: string,
    public readonly producedBy: string,
  ) {
    super(`Missing model artifact "${artifact}". Produce it with: ${producedBy}`);
    this.name = "MissingModelError";
  }
}
