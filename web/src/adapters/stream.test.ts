import { describe, it, expect } from "vitest";
import { createEngine, NATIVE_SAMPLE_RATE } from "../engine/engine.js";
import type { SentenceRequest, SpeechModel } from "../engine/speech-model.js";
import type { Voice } from "../engine/voice.js";
import { pcmStream } from "./stream.js";

const VOICES: Voice[] = [{ id: "ida", displayName: "Ida", lang: "da-DK" }];

function countingModel(): { model: SpeechModel; count: () => number } {
  let count = 0;
  const model: SpeechModel = {
    voices: () => VOICES,
    async synthesizeSentence(_req: SentenceRequest) {
      count++;
      await new Promise((r) => setTimeout(r, 10));
      return new Float32Array(NATIVE_SAMPLE_RATE / 100).fill(0.1);
    },
  };
  return { model, count: () => count };
}

describe("pcmStream", () => {
  it("emits one encoded chunk per sentence", async () => {
    const { model } = countingModel();
    const stream = pcmStream(createEngine(model), { text: "En. To.", voice: "ida" });
    const reader = stream.getReader();
    const chunks: number[] = [];
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      chunks.push(value.byteLength);
    }
    // Two sentences, each (NATIVE_SAMPLE_RATE/100 samples * 2 bytes).
    expect(chunks).toHaveLength(2);
    expect(chunks[0]).toBe((NATIVE_SAMPLE_RATE / 100) * 2);
  });

  it("stops generating when the consumer cancels the stream", async () => {
    const { model, count } = countingModel();
    const stream = pcmStream(createEngine(model), {
      text: "En. To. Tre. Fire. Fem.",
      voice: "ida",
    });
    const reader = stream.getReader();
    await reader.read(); // pull the first sentence's chunk
    await reader.cancel();
    await new Promise((r) => setTimeout(r, 60)); // let any in-flight work settle

    // Without cancellation all 5 sentences would be synthesized; cancelling
    // mid-stream must stop the engine well short of that.
    expect(count()).toBeLessThan(5);
  });

  it("propagates an already-aborted caller signal as no output", async () => {
    const { model, count } = countingModel();
    const stream = pcmStream(createEngine(model), {
      text: "En. To.",
      voice: "ida",
      signal: AbortSignal.abort(),
    });
    const reader = stream.getReader();
    await expect(reader.read()).rejects.toBeDefined();
    expect(count()).toBe(0);
  });
});
