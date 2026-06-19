import { describe, it, expect } from "vitest";
import { encodeAudio, encodeWav, pcmToInt16LE } from "./format.js";

const SAMPLE_RATE = 24000;

function int16At(bytes: Uint8Array, sampleIndex: number, byteOffset = 0): number {
  return new DataView(bytes.buffer).getInt16(byteOffset + sampleIndex * 2, true);
}

describe("pcmToInt16LE", () => {
  it("quantizes and clamps samples to 16-bit", () => {
    const bytes = pcmToInt16LE(new Float32Array([0, 1, -1, 0.5, 2, -2]));
    expect(bytes.length).toBe(6 * 2);
    expect(int16At(bytes, 0)).toBe(0);
    expect(int16At(bytes, 1)).toBe(32767); // +1.0
    expect(int16At(bytes, 2)).toBe(-32768); // -1.0
    expect(int16At(bytes, 3)).toBe(16383); // 0.5 truncated
    expect(int16At(bytes, 4)).toBe(32767); // clamped from +2
    expect(int16At(bytes, 5)).toBe(-32768); // clamped from -2
  });

  it("writes samples little-endian", () => {
    const bytes = pcmToInt16LE(new Float32Array([1])); // 32767 = 0x7FFF
    expect(bytes[0]).toBe(0xff);
    expect(bytes[1]).toBe(0x7f);
  });
});

describe("encodeWav", () => {
  it("prepends a 44-byte header and embeds the PCM data", () => {
    const pcm = new Float32Array([0.25, -0.25]);
    const wav = encodeWav(pcm, SAMPLE_RATE);
    const view = new DataView(wav.buffer);
    const ascii = (o: number, n: number) =>
      Array.from({ length: n }, (_, i) => String.fromCharCode(view.getUint8(o + i))).join("");

    expect(wav.length).toBe(44 + pcm.length * 2);
    expect(ascii(0, 4)).toBe("RIFF");
    expect(ascii(8, 4)).toBe("WAVE");
    expect(ascii(36, 4)).toBe("data");
    expect(view.getUint32(24, true)).toBe(SAMPLE_RATE);
    // data section equals the standalone PCM encoding
    expect(int16At(wav, 0, 44)).toBe(int16At(pcmToInt16LE(pcm), 0));
  });
});

describe("encodeAudio", () => {
  const pcm = new Float32Array([0.1, -0.2, 0.3]);

  it("dispatches 'pcm' to raw 16-bit PCM", () => {
    expect(encodeAudio(pcm, SAMPLE_RATE, "pcm")).toEqual(pcmToInt16LE(pcm));
  });

  it("dispatches 'wav' to the WAV container", () => {
    expect(encodeAudio(pcm, SAMPLE_RATE, "wav")).toEqual(encodeWav(pcm, SAMPLE_RATE));
  });
});
