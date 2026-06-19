import { describe, it, expect } from "vitest";
import { resample } from "./resample.js";

describe("resample", () => {
  it("is a no-op when the rates match", () => {
    const pcm = new Float32Array([0.1, 0.2, 0.3]);
    expect(resample(pcm, 24000, 24000)).toBe(pcm);
  });

  it("scales the length by the rate ratio", () => {
    const pcm = new Float32Array(2400).fill(0.5);
    expect(resample(pcm, 24000, 44100).length).toBe(Math.round(2400 * (44100 / 24000)));
    expect(resample(pcm, 24000, 22050).length).toBe(Math.round(2400 * (22050 / 24000)));
  });

  it("preserves a constant signal exactly", () => {
    const pcm = new Float32Array(100).fill(0.42);
    for (const v of resample(pcm, 24000, 44100)) expect(v).toBeCloseTo(0.42, 6);
  });

  it("reproduces a linear ramp (Catmull-Rom is exact on collinear points)", () => {
    const n = 200;
    const pcm = new Float32Array(n);
    for (let i = 0; i < n; i++) pcm[i] = i / (n - 1); // 0 .. 1
    const up = resample(pcm, 24000, 48000);
    // Interior samples should lie on the same line (skip the clamped edges).
    for (let j = 10; j < up.length - 10; j++) {
      const x = j / (up.length - 1);
      expect(up[j]).toBeCloseTo(x, 2);
    }
  });

  it("handles empty input", () => {
    expect(resample(new Float32Array(0), 24000, 44100).length).toBe(0);
  });
});
