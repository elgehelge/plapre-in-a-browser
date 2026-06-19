import { describe, it, expect } from "vitest";
import { timeStretch } from "./time-stretch.js";

const SR = 24000;

function sine(freqHz: number, seconds: number): Float32Array {
  const n = Math.round(seconds * SR);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.sin((2 * Math.PI * freqHz * i) / SR);
  return out;
}

/** Crude pitch estimate via zero-crossing rate (good enough for a clean sine). */
function dominantFreq(pcm: Float32Array): number {
  let crossings = 0;
  for (let i = 1; i < pcm.length; i++) {
    if (pcm[i - 1] <= 0 && pcm[i] > 0) crossings++;
  }
  return (crossings * SR) / pcm.length;
}

describe("timeStretch", () => {
  it("is a no-op for speed ~ 1", () => {
    const x = sine(200, 0.5);
    const y = timeStretch(x, 1);
    expect(y.length).toBe(x.length);
    expect(Array.from(y.subarray(0, 8))).toEqual(Array.from(x.subarray(0, 8)));
  });

  // Duration scales by ~1/speed. Bounds (not exact) absorb the one partial
  // analysis frame WSOLA drops at the tail.
  it("shortens audio for speed > 1", () => {
    const x = sine(200, 1.0);
    const ratio = timeStretch(x, 2).length / x.length;
    expect(ratio).toBeGreaterThan(0.45);
    expect(ratio).toBeLessThan(0.55);
  });

  it("lengthens audio for speed < 1", () => {
    const x = sine(200, 1.0);
    const ratio = timeStretch(x, 0.5).length / x.length;
    expect(ratio).toBeGreaterThan(1.9);
    expect(ratio).toBeLessThan(2.1);
  });

  it("preserves pitch when stretching", () => {
    const x = sine(220, 1.0);
    for (const speed of [0.7, 1.5, 2.0]) {
      const y = timeStretch(x, speed);
      // Pitch should stay ~220 Hz regardless of speed (within ~10%).
      expect(dominantFreq(y)).toBeGreaterThan(220 * 0.9);
      expect(dominantFreq(y)).toBeLessThan(220 * 1.1);
    }
  });

  it("rejects non-positive speed", () => {
    expect(() => timeStretch(sine(200, 0.1), 0)).toThrow();
    expect(() => timeStretch(sine(200, 0.1), -1)).toThrow();
  });
});
