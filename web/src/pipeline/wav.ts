// Float32 PCM -> WAV Blob for playback in <audio>. Encoding lives in the audio
// layer; this just wraps it as a Blob for the demo UI.

import { encodeWav } from "../audio/format.js";

export function pcmToWavBlob(pcm: Float32Array, sampleRate: number): Blob {
  return new Blob([encodeWav(pcm, sampleRate)], { type: "audio/wav" });
}
