import { describe, it, expect } from "vitest";
import { argmax, makeRng, sample, type SamplingParams } from "./sampling.js";

const params = (overrides: Partial<SamplingParams> = {}): SamplingParams => ({
  temperature: 1,
  topK: 0,
  topP: 1,
  ...overrides,
});

describe("argmax", () => {
  it("returns the index of the largest logit", () => {
    expect(argmax([0.1, 0.9, 0.3])).toBe(1);
  });

  it("returns the first index on ties", () => {
    expect(argmax([2, 2, 1])).toBe(0);
  });
});

describe("makeRng", () => {
  it("is deterministic for a given seed", () => {
    const a = makeRng(42);
    const b = makeRng(42);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
  });

  it("produces values in [0, 1)", () => {
    const rng = makeRng(7);
    for (let i = 0; i < 100; i++) {
      const v = rng();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it("differs across seeds", () => {
    expect(makeRng(1)()).not.toBe(makeRng(2)());
  });
});

describe("sample", () => {
  const logits = [1, 5, 2, 0]; // argmax at index 1

  it("is greedy when temperature is 0, ignoring the rng", () => {
    const alwaysLast = () => 0.999999;
    expect(sample(logits, params({ temperature: 0 }), alwaysLast)).toBe(1);
  });

  it("returns the top token when top-k is 1, for any rng draw", () => {
    expect(sample(logits, params({ topK: 1 }), () => 0)).toBe(1);
    expect(sample(logits, params({ topK: 1 }), () => 0.999999)).toBe(1);
  });

  it("picks the highest-probability token when the rng draws 0", () => {
    expect(sample(logits, params(), () => 0)).toBe(1);
  });

  it("collapses to the dominant token under a tight nucleus (top-p)", () => {
    // One logit dominates, so its softmax mass already exceeds topP: the nucleus
    // keeps only it, and every rng draw must return it.
    const peaked = [12, 0, 0, 0];
    expect(sample(peaked, params({ topP: 0.5 }), () => 0)).toBe(0);
    expect(sample(peaked, params({ topP: 0.5 }), () => 0.999999)).toBe(0);
  });

  it("is deterministic given the same seed and parameters", () => {
    const draw = () => sample(logits, params({ temperature: 0.8, topK: 3, topP: 0.95 }), makeRng(123));
    expect(draw()).toBe(draw());
  });

  it("only ever returns an in-range token id", () => {
    const rng = makeRng(99);
    for (let i = 0; i < 200; i++) {
      const id = sample(logits, params({ temperature: 0.7, topK: 2, topP: 0.9 }), rng);
      expect(id).toBeGreaterThanOrEqual(0);
      expect(id).toBeLessThan(logits.length);
    }
  });
});
