import { describe, it, expect } from "vitest";
import { SpeakerProjection, VoiceClonerImpl, type CloneEmbedder } from "./clone.js";

describe("SpeakerProjection", () => {
  it("applies y = W·x + b", () => {
    // out=2, in=3
    const proj = SpeakerProjection.fromJSON({
      in: 3,
      out: 2,
      weight: [
        [1, 0, 0],
        [0, 2, 1],
      ],
      bias: [10, -1],
    });
    const y = proj.apply(Float32Array.from([5, 3, 4]));
    expect(Array.from(y)).toEqual([15, 9]); // [5+10, 6+4-1]
  });

  it("rejects a wrong-sized input", () => {
    const proj = SpeakerProjection.fromJSON({ in: 2, out: 1, weight: [[1, 1]], bias: [0] });
    expect(() => proj.apply(Float32Array.from([1, 2, 3]))).toThrow(/projection input/);
  });
});

describe("VoiceClonerImpl", () => {
  const fixedEmbedder = (raw: number[]): CloneEmbedder => ({
    embed: async () => Float32Array.from(raw),
  });

  it("produces raw + projected hidden for a SpeakerData", async () => {
    const proj = SpeakerProjection.fromJSON({
      in: 3,
      out: 2,
      weight: [
        [1, 0, 0],
        [0, 0, 1],
      ],
      bias: [0, 0],
    });
    const cloner = VoiceClonerImpl.withParts(fixedEmbedder([7, 8, 9]), proj);
    const speaker = await cloner.embedSpeaker(new Float32Array([0.1, 0.2]), 16000);
    expect(speaker.raw).toEqual([7, 8, 9]);
    expect(speaker.hidden).toEqual([7, 9]);
  });

  it("rejects empty reference audio", async () => {
    const proj = SpeakerProjection.fromJSON({ in: 1, out: 1, weight: [[1]], bias: [0] });
    const cloner = VoiceClonerImpl.withParts(fixedEmbedder([1]), proj);
    await expect(cloner.embedSpeaker(new Float32Array(0), 16000)).rejects.toThrow(/empty/);
  });

  it("passes the audio + sample rate through to the embedder", async () => {
    let seen: { len: number; sr: number } | null = null;
    const embedder: CloneEmbedder = {
      embed: async (audio, sr) => {
        seen = { len: audio.length, sr };
        return Float32Array.from([1, 1, 1]);
      },
    };
    const proj = SpeakerProjection.fromJSON({
      in: 3,
      out: 1,
      weight: [[1, 1, 1]],
      bias: [0],
    });
    const cloner = VoiceClonerImpl.withParts(embedder, proj);
    await cloner.embedSpeaker(new Float32Array(48000), 48000);
    expect(seen).toEqual({ len: 48000, sr: 48000 });
  });
});
