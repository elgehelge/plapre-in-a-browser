import { describe, it, expect } from "vitest";
import { PlapreLM, type KvCache, type LmGraph, type LmStep, type PromptTokenizer } from "./lm.js";
import type { GenerateOptions } from "./lm.js";

const HIDDEN = 4;
const VOCAB = 12;
const EOS = 11;

const tokenizer: PromptTokenizer = {
  // [<text>=0] + ids + [<audio>=1]; the ids are the text bytes for determinism.
  buildPrompt: (text) => [0, ...Array.from(text, (c) => c.charCodeAt(0) % 10), 1],
  special: { eos: EOS },
};

interface ForwardCall {
  inputIds: number[];
  prefixLen: number;
  prefixEmbeds: Float32Array;
  past: KvCache;
}

/**
 * Fake graph whose argmax at step n is `peaks[n]` (clamped to the last entry).
 * Each call returns a fresh cache object so KV feedback can be checked by identity.
 */
function fakeGraph(peaks: number[]): { graph: LmGraph; calls: ForwardCall[] } {
  const calls: ForwardCall[] = [];
  const graph: LmGraph = {
    hidden: HIDDEN,
    emptyCache: () => new Map(),
    async forward(inputIds, prefixEmbeds, prefixLen, past): Promise<LmStep> {
      calls.push({ inputIds, prefixLen, prefixEmbeds, past });
      const logits = new Float32Array(VOCAB); // all zero
      const peak = peaks[Math.min(calls.length - 1, peaks.length - 1)];
      logits[peak] = 10; // dominates argmax and softmax
      return { logits, present: new Map() };
    },
  };
  return { graph, calls };
}

const GREEDY: GenerateOptions = { temperature: 0, topK: 0, topP: 1, maxTokens: 50, seed: 0 };

describe("PlapreLM.generate decode loop", () => {
  it("emits the argmax sequence and stops at EOS (greedy)", async () => {
    const { graph } = fakeGraph([5, 6, 7, EOS]);
    const lm = PlapreLM.withGraph(graph, tokenizer);
    const ids = await lm.generate("hej", new Float32Array(HIDDEN), GREEDY);
    expect(ids).toEqual([5, 6, 7]); // EOS terminates, not included
  });

  it("prepends the speaker hidden only on prefill, then feeds tokens back", async () => {
    const speaker = new Float32Array([0.5, -0.5, 0.25, -0.25]); // exact in float32
    const { graph, calls } = fakeGraph([5, 6, EOS]);
    const lm = PlapreLM.withGraph(graph, tokenizer);
    await lm.generate("a", speaker, GREEDY);

    // Prefill: prompt ids, prefixLen 1, the speaker vector.
    expect(calls[0].prefixLen).toBe(1);
    expect(Array.from(calls[0].prefixEmbeds)).toEqual([0.5, -0.5, 0.25, -0.25]);
    expect(calls[0].inputIds).toEqual(tokenizer.buildPrompt("a"));
    // Decode steps: single previous token, no prefix.
    expect(calls[1].prefixLen).toBe(0);
    expect(calls[1].inputIds).toEqual([5]);
    expect(calls[2].inputIds).toEqual([6]);
  });

  it("feeds each step's present cache as the next step's past", async () => {
    const { graph, calls } = fakeGraph([5, 6, 7, EOS]);
    // Wrap forward to expose the present it returns.
    const presents: KvCache[] = [];
    const orig = graph.forward.bind(graph);
    graph.forward = async (...args) => {
      const step = await orig(...args);
      presents.push(step.present);
      return step;
    };
    const lm = PlapreLM.withGraph(graph, tokenizer);
    await lm.generate("hi", new Float32Array(HIDDEN), GREEDY);

    // calls[n].past === presents[n-1] (the previous step's output cache).
    for (let n = 1; n < calls.length; n++) {
      expect(calls[n].past).toBe(presents[n - 1]);
    }
  });

  it("respects maxTokens when EOS never comes", async () => {
    const { graph } = fakeGraph([5]); // always argmax 5, never EOS
    const lm = PlapreLM.withGraph(graph, tokenizer);
    const ids = await lm.generate("x", new Float32Array(HIDDEN), { ...GREEDY, maxTokens: 4 });
    expect(ids).toEqual([5, 5, 5, 5]);
  });

  it("is deterministic across runs with the same seed (sampled)", async () => {
    const make = () => PlapreLM.withGraph(fakeGraph([5, 6, 7, EOS]).graph, tokenizer);
    const opts: GenerateOptions = { temperature: 0.8, topK: 5, topP: 0.95, maxTokens: 10, seed: 42 };
    const a = await make().generate("hej", new Float32Array(HIDDEN), opts);
    const b = await make().generate("hej", new Float32Array(HIDDEN), opts);
    expect(a).toEqual(b);
  });

  it("rejects a speaker vector whose length != model hidden", async () => {
    const { graph } = fakeGraph([EOS]);
    const lm = PlapreLM.withGraph(graph, tokenizer);
    await expect(lm.generate("hej", new Float32Array(HIDDEN + 1), GREEDY)).rejects.toThrow(
      /hidden/,
    );
  });
});
