// Decode encoded audio (mp3/wav/…) into mono Float32 PCM + its sample rate.
//
// Voice cloning takes reference audio in whatever the caller has (an ElevenLabs
// client passes file Blobs). Decoding is environment-specific, so the cloning
// adapter takes an `AudioDecoder` it can inject; this is the browser default
// (WebAudio). Multi-channel input is downmixed to mono.

export interface DecodedAudio {
  pcm: Float32Array; // mono
  sampleRate: number;
}

export type AudioDecoder = (data: ArrayBuffer) => Promise<DecodedAudio>;

/** Downmix interleaved/per-channel buffers to a single mono channel (average). */
export function downmixToMono(channels: Float32Array[]): Float32Array {
  if (channels.length === 1) return channels[0];
  const length = channels[0].length;
  const out = new Float32Array(length);
  for (const ch of channels) for (let i = 0; i < length; i++) out[i] += ch[i];
  const inv = 1 / channels.length;
  for (let i = 0; i < length; i++) out[i] *= inv;
  return out;
}

/** Default decoder using the WebAudio API (browser/offscreen-document only). */
export async function decodeWithWebAudio(data: ArrayBuffer): Promise<DecodedAudio> {
  const Ctx =
    (globalThis as { AudioContext?: typeof AudioContext }).AudioContext ??
    (globalThis as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctx) throw new Error("No WebAudio AudioContext available to decode reference audio");

  const ctx = new Ctx();
  try {
    // decodeAudioData detaches its input, so hand it a copy.
    const buffer = await ctx.decodeAudioData(data.slice(0));
    const channels: Float32Array[] = [];
    for (let c = 0; c < buffer.numberOfChannels; c++) channels.push(buffer.getChannelData(c));
    return { pcm: downmixToMono(channels), sampleRate: buffer.sampleRate };
  } finally {
    if (typeof ctx.close === "function") void ctx.close();
  }
}
