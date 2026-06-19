// A voice as seen by callers: an opaque identity plus presentation metadata.
// The embedding that actually drives synthesis is private to the SpeechModel
// that owns the voice — callers never see or pass it.

export interface Voice {
  readonly id: string;
  readonly displayName: string;
  readonly lang: string; // BCP-47, e.g. "da-DK"
}

export class UnknownVoiceError extends Error {
  constructor(
    readonly requested: string,
    readonly available: readonly string[],
  ) {
    super(`Unknown voice "${requested}". Available: ${available.join(", ") || "(none)"}`);
    this.name = "UnknownVoiceError";
  }
}

/**
 * Resolve a requested voice id to a Voice, or throw. Resolving up front means
 * the rest of the pipeline only ever handles a known Voice, not a raw string.
 */
export function resolveVoice(voices: readonly Voice[], requested: string): Voice {
  const found = voices.find((v) => v.id === requested);
  if (!found) {
    throw new UnknownVoiceError(
      requested,
      voices.map((v) => v.id),
    );
  }
  return found;
}
