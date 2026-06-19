// Audio serialization at the boundary: encode the engine's canonical mono
// Float32 PCM (NATIVE_SAMPLE_RATE) into the byte formats adapters return.
//
// Only formats that need no extra dependency or resampling are supported here:
// raw 16-bit little-endian PCM and a WAV container, both at the native rate.
// Lossy/resampled formats (mp3, opus, aac, …) require an encoder and are an
// explicit, typed gap — adapters surface them as UnsupportedFormatError.

export type AudioFormat = "pcm" | "wav";

export class UnsupportedFormatError extends Error {
  constructor(readonly format: string) {
    super(`Audio format "${format}" is not supported yet (needs an encoder/resampler)`);
    this.name = "UnsupportedFormatError";
  }
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
  }
}
