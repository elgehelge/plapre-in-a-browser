// Audio serialization at the boundary: encode the engine's canonical mono
// Float32 PCM into the byte formats adapters return — raw 16-bit little-endian
// PCM, a WAV container, and MP3 (via the pure-JS lamejs encoder, so it works in
// the browser / a worker / an MV3 extension with no native deps).
//
// Sample-rate conversion is NOT done here: callers pass already-resampled PCM
// plus its rate (see audio/resample.ts). Formats still needing a heavier codec
// (opus, aac, flac) are an explicit, typed gap — UnsupportedFormatError.

import { Mp3Encoder } from "@breezystack/lamejs";

export type AudioFormat = "pcm" | "wav" | "mp3";

export class UnsupportedFormatError extends Error {
  constructor(readonly format: string) {
    super(`Audio format "${format}" is not supported yet (needs an encoder/resampler)`);
    this.name = "UnsupportedFormatError";
  }
}

/** Quantize float samples to clamped 16-bit integers (host order). */
function pcmToInt16(pcm: Float32Array): Int16Array {
  const out = new Int16Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

/** Quantize float samples to clamped 16-bit little-endian PCM bytes. */
export function pcmToInt16LE(pcm: Float32Array): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(pcm.length * 2);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < pcm.length; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return bytes;
}

const MP3_BLOCK = 1152; // one MPEG frame's worth of samples

/** Encode mono PCM to an MP3 byte stream at the given sample rate / bitrate. */
export function encodeMp3(pcm: Float32Array, sampleRate: number, kbps = 128): Uint8Array<ArrayBuffer> {
  const encoder = new Mp3Encoder(1, sampleRate, kbps);
  const samples = pcmToInt16(pcm);
  const parts: Uint8Array[] = [];
  for (let i = 0; i < samples.length; i += MP3_BLOCK) {
    const frame = encoder.encodeBuffer(samples.subarray(i, i + MP3_BLOCK));
    if (frame.length > 0) parts.push(frame);
  }
  const tail = encoder.flush();
  if (tail.length > 0) parts.push(tail);

  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

/** Wrap 16-bit PCM in a canonical mono WAV container. */
export function encodeWav(pcm: Float32Array, sampleRate: number): Uint8Array<ArrayBuffer> {
  const data = pcmToInt16LE(pcm);
  const out = new Uint8Array(44 + data.length);
  const view = new DataView(out.buffer);
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + data.length, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, "data");
  view.setUint32(40, data.length, true);
  out.set(data, 44);
  return out;
}

export function encodeAudio(
  pcm: Float32Array,
  sampleRate: number,
  format: AudioFormat,
): Uint8Array<ArrayBuffer> {
  switch (format) {
    case "pcm":
      return pcmToInt16LE(pcm);
    case "wav":
      return encodeWav(pcm, sampleRate);
    case "mp3":
      return encodeMp3(pcm, sampleRate);
  }
}
