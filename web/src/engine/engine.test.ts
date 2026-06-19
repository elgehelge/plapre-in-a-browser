import { describe, it, expect } from "vitest";
import {
  createEngine,
  CloningUnsupportedError,
  NATIVE_SAMPLE_RATE,
  type PcmChunk,
} from "./engine.js";
import type {
  CloneVoiceOptions,
  SentenceRequest,
  SpeechModel,
  VoiceCloner,
} from "./speech-model.js";
import { UnknownVoiceError, type Voice } from "./voice.js";

const VOICES: Voice[] = [
  { id: "ida", displayName: "Ida", lang: "da-DK" },
  { id: "tor", displayName: "Tor", lang: "da-DK" },
];

/**
 * In-memory SpeechModel for behavior tests. We own this seam, so this is a real
 * implementation (not a mock of foreign code): it records the requests it
 * receives and renders deterministic PCM via an injected function.
 */
class FakeSpeechModel implements SpeechModel {
  readonly calls: SentenceRequest[] = [];
  constructor(private readonly render: (req: SentenceRequest) => Float32Array) {}
  voices(): readonly Voice[] {
    return VOICES;
  }
  async synthesizeSentence(request: SentenceRequest): Promise<Float32Array> {
    this.calls.push(request);
    request.signal?.throwIfAborted();
    return this.render(request);
  }
}

/** One second of PCM filled with `marker`, so chunk order is verifiable. */
function second(marker: number): Float32Array {
  return new Float32Array(NATIVE_SAMPLE_RATE).fill(marker);
}

async function collect(stream: AsyncIterable<PcmChunk>): Promise<PcmChunk[]> {
  const out: PcmChunk[] = [];
  for await (const chunk of stream) out.push(chunk);
  return out;
}

describe("createEngine — voices", () => {
  it("exposes the model's voices", () => {
    const engine = createEngine(new FakeSpeechModel(() => second(1)));
    expect(engine.listVoices()).toEqual(VOICES);
  });

  it("rejects an unknown voice id with the available ids", async () => {
    const model = new FakeSpeechModel(() => second(1));
    const engine = createEngine(model);
    await expect(
      engine.synthesizeToPcm({ text: "Hej.", voice: "nobody" }),
    ).rejects.toBeInstanceOf(UnknownVoiceError);
    expect(model.calls).toHaveLength(0);
  });
});

describe("createEngine — streaming", () => {
  it("yields one chunk per sentence with cumulative start times", async () => {
    const engine = createEngine(new FakeSpeechModel((req) => second(req.sentence.length)));
    const chunks = await collect(engine.synthesize({ text: "En. To. Tre.", voice: "ida" }));

    expect(chunks).toHaveLength(3);
    expect(chunks.map((c) => c.startSec)).toEqual([0, 1, 2]);
    expect(chunks.every((c) => c.sampleRate === NATIVE_SAMPLE_RATE)).toBe(true);
  });

  it("skips sentences that produce no audio", async () => {
    const render = (req: SentenceRequest) =>
      req.sentence.includes("tom") ? new Float32Array(0) : second(1);
    const engine = createEngine(new FakeSpeechModel(render));

    const chunks = await collect(engine.synthesize({ text: "Hej. tom her. Dav.", voice: "ida" }));
    expect(chunks).toHaveLength(2);
    expect(chunks.map((c) => c.startSec)).toEqual([0, 1]);
  });
});

describe("createEngine — buffered", () => {
  it("concatenates sentence chunks in order", async () => {
    const engine = createEngine(new FakeSpeechModel((req) => second(req.sentence.length)));
    const { samples, sampleRate } = await engine.synthesizeToPcm({ text: "En. To.", voice: "ida" });

    expect(sampleRate).toBe(NATIVE_SAMPLE_RATE);
    expect(samples).toHaveLength(2 * NATIVE_SAMPLE_RATE);
    expect(samples[0]).toBe(3); // "En." has length 3
    expect(samples[NATIVE_SAMPLE_RATE]).toBe(3); // "To." also length 3
  });

  it("returns empty PCM for text with no sentences", async () => {
    const engine = createEngine(new FakeSpeechModel(() => second(1)));
    const { samples } = await engine.synthesizeToPcm({ text: "   ", voice: "ida" });
    expect(samples).toHaveLength(0);
  });
});

describe("createEngine — inter-sentence silence", () => {
  it("inserts silence between sentences but not before the first or after the last", async () => {
    const engine = createEngine(new FakeSpeechModel(() => second(1)), {
      interSentenceSilenceSec: 0.5,
    });
    const chunks = await collect(engine.synthesize({ text: "En. To. Tre.", voice: "ida" }));
    const half = NATIVE_SAMPLE_RATE / 2;

    expect(chunks).toHaveLength(3);
    expect(chunks[0].samples.length).toBe(NATIVE_SAMPLE_RATE); // no leading silence
    expect(chunks[1].samples.length).toBe(half + NATIVE_SAMPLE_RATE); // gap prepended
    expect(chunks[2].samples.length).toBe(half + NATIVE_SAMPLE_RATE);
    // start times account for the inserted gaps
    expect(chunks.map((c) => c.startSec)).toEqual([0, 1, 2.5]);
  });

  it("defaults to no silence (contiguous chunks)", async () => {
    const engine = createEngine(new FakeSpeechModel(() => second(1)));
    const chunks = await collect(engine.synthesize({ text: "En. To.", voice: "ida" }));
    expect(chunks.every((c) => c.samples.length === NATIVE_SAMPLE_RATE)).toBe(true);
  });
});

describe("createEngine — generation options", () => {
  it("merges request generation over engine defaults", async () => {
    const model = new FakeSpeechModel(() => second(1));
    const engine = createEngine(model, { generation: { temperature: 0.5 } });

    await engine.synthesizeToPcm({ text: "Hej.", voice: "ida", generation: { topK: 10 } });

    const gen = model.calls[0].generation;
    expect(gen.temperature).toBe(0.5); // from engine options
    expect(gen.topK).toBe(10); // from request
    expect(gen.maxTokens).toBe(500); // from built-in defaults
  });
});

describe("createEngine — cancellation", () => {
  it("does not synthesize anything when already aborted", async () => {
    const model = new FakeSpeechModel(() => second(1));
    const engine = createEngine(model);
    const controller = new AbortController();
    controller.abort();

    await expect(
      collect(engine.synthesize({ text: "En. To.", voice: "ida", signal: controller.signal })),
    ).rejects.toThrow();
    expect(model.calls).toHaveLength(0);
  });

  it("stops before the next sentence once aborted mid-stream", async () => {
    const controller = new AbortController();
    const model = new FakeSpeechModel(() => {
      controller.abort(); // abort while producing the first sentence
      return second(1);
    });
    const engine = createEngine(model);

    const produced: PcmChunk[] = [];
    await expect(
      (async () => {
        for await (const chunk of engine.synthesize({
          text: "En. To. Tre.",
          voice: "ida",
          signal: controller.signal,
        })) {
          produced.push(chunk);
        }
      })(),
    ).rejects.toThrow();

    expect(produced).toHaveLength(1); // first chunk emitted, no further sentences
    expect(model.calls).toHaveLength(1);
  });

  it("forwards the abort signal to the model", async () => {
    const model = new FakeSpeechModel(() => second(1));
    const engine = createEngine(model);
    const controller = new AbortController();

    await engine.synthesizeToPcm({ text: "Hej.", voice: "ida", signal: controller.signal });
    expect(model.calls[0].signal).toBe(controller.signal);
  });
});

/** A model that supports cloning: registers cloned voices into its own catalog. */
class CloningFakeModel extends FakeSpeechModel implements VoiceCloner {
  private readonly cloned: Voice[] = [];
  cloneCalls: { audio: Float32Array; sampleRate: number }[] = [];
  constructor() {
    super(() => second(1));
  }
  voices(): readonly Voice[] {
    return [...VOICES, ...this.cloned];
  }
  async cloneVoice(
    audio: Float32Array,
    sampleRate: number,
    opts: CloneVoiceOptions = {},
  ): Promise<Voice> {
    this.cloneCalls.push({ audio, sampleRate });
    const voice = { id: opts.id ?? "cloned-1", displayName: opts.displayName ?? "Cloned", lang: "da-DK" };
    this.cloned.push(voice);
    return voice;
  }
}

describe("createEngine — voice cloning", () => {
  it("reports cloning unsupported for a plain model", async () => {
    const engine = createEngine(new FakeSpeechModel(() => second(1)));
    expect(engine.canCloneVoice()).toBe(false);
    await expect(engine.cloneVoice(new Float32Array(8), 16000)).rejects.toBeInstanceOf(
      CloningUnsupportedError,
    );
  });

  it("delegates to a cloning-capable model and makes the voice usable", async () => {
    const model = new CloningFakeModel();
    const engine = createEngine(model);
    expect(engine.canCloneVoice()).toBe(true);

    const audio = new Float32Array(16000).fill(0.3);
    const voice = await engine.cloneVoice(audio, 16000, { id: "my-voice" });
    expect(voice.id).toBe("my-voice");
    expect(model.cloneCalls).toHaveLength(1);
    expect(model.cloneCalls[0].sampleRate).toBe(16000);

    // The cloned voice is now resolvable and synthesizable.
    expect(engine.listVoices().some((v) => v.id === "my-voice")).toBe(true);
    const pcm = await engine.synthesizeToPcm({ text: "Hej.", voice: "my-voice" });
    expect(pcm.samples.length).toBeGreaterThan(0);
  });
});
