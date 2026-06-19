import { describe, it, expect } from "vitest";
import { pcmToWavBlob } from "./wav.js";

const SAMPLE_RATE = 24000;

async function bytesOf(blob: Blob): Promise<DataView> {
  return new DataView(await blob.arrayBuffer());
}

function ascii(view: DataView, offset: number, length: number): string {
  let s = "";
  for (let i = 0; i < length; i++) s += String.fromCharCode(view.getUint8(offset + i));
  return s;
}

describe("pcmToWavBlob", () => {
  it("produces a 44-byte header plus 16-bit samples", async () => {
    const pcm = new Float32Array([0, 0, 0]);
    const blob = pcmToWavBlob(pcm, SAMPLE_RATE);
    expect(blob.type).toBe("audio/wav");
    expect(blob.size).toBe(44 + pcm.length * 2);
  });

  it("writes a canonical mono 16-bit PCM WAV header", async () => {
    const view = await bytesOf(pcmToWavBlob(new Float32Array([0]), SAMPLE_RATE));
    expect(ascii(view, 0, 4)).toBe("RIFF");
    expect(ascii(view, 8, 4)).toBe("WAVE");
    expect(ascii(view, 12, 4)).toBe("fmt ");
    expect(view.getUint16(20, true)).toBe(1); // PCM
    expect(view.getUint16(22, true)).toBe(1); // mono
    expect(view.getUint32(24, true)).toBe(SAMPLE_RATE);
    expect(view.getUint32(28, true)).toBe(SAMPLE_RATE * 2); // byte rate
    expect(view.getUint16(32, true)).toBe(2); // block align
    expect(view.getUint16(34, true)).toBe(16); // bits/sample
    expect(ascii(view, 36, 4)).toBe("data");
  });

  it("quantizes float samples to int16 and clamps out-of-range values", async () => {
    const pcm = new Float32Array([0, 1, -1, 0.5, 2, -2]);
    const view = await bytesOf(pcmToWavBlob(pcm, SAMPLE_RATE));
    const sampleAt = (i: number) => view.getInt16(44 + i * 2, true);
    expect(sampleAt(0)).toBe(0);
    expect(sampleAt(1)).toBe(32767); // +1.0 full scale
    expect(sampleAt(2)).toBe(-32768); // -1.0 full scale
    expect(sampleAt(3)).toBe(16383); // 0.5 * 32767, truncated
    expect(sampleAt(4)).toBe(32767); // clamped from +2.0
    expect(sampleAt(5)).toBe(-32768); // clamped from -2.0
  });
});
