// Map a provider's voice id onto an engine voice id. A provider voice can be
// remapped via the table; otherwise it is used as-is (so passing a native engine
// voice id works too). The result is validated against the engine's catalog.

import { UnknownVoiceError, type Voice } from "../engine/voice.js";

export type VoiceMap = Record<string, string>;

export function mapProviderVoice(
  voices: readonly Voice[],
  voiceMap: VoiceMap,
  requested: string,
): string {
  const target = voiceMap[requested] ?? requested;
  if (!voices.some((v) => v.id === target)) {
    throw new UnknownVoiceError(
      requested,
      voices.map((v) => v.id),
    );
  }
  return target;
}
