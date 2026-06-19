import { describe, it, expect } from "vitest";
import { createOpenAISpeech } from "./openai.js";
import { UnsupportedFormatError } from "../audio/format.js";
import { UnsupportedSpeedError } from "./errors.js";
import { UnknownVoiceError, type Voice } from "../engine/voice.js";
import { NATIVE_SAMPLE_RATE, type Engine, type SynthesisRequest } from "../engine/engine.js";

const VOICES: Voice[] = [
  { id: "ida", displayName: "Ida", lang: "da-DK" },
  { id: "tor", displayName: "Tor", lang: "da-DK" },
];

/** Recording fake engine that returns a fixed PCM buffer. */
function fakeEngine(pcm = new Float32Array([0.5, -0.5])) {
  const requests: SynthesisRequest[] = [];
  const engine: Engine = {
    listVoices: () => VOICES,
    async *synthesize(req) {
      requests.push(req);
      yield { samples: pcm, sampleRate: NATIVE_SAMPLE_RATE, startSec: 0 };
    },
    async synthesizeToPcm(req) {
      requests.push(req);
      return { samples: pcm, sampleRate: NATIVE_SAMPLE_RATE };
    },
  };
  return { engine, requests };
}

describe("createOpenAISpeech — output", () => {
  it("returns an MP3 Response by default (matching OpenAI)", async () => {
    const { engine } = fakeEngine(new Float32Array(2048).fill(0.2));
    const tts = createOpenAISpeech(engine);
    const res = await tts.create({ voice: "alloy", input: "Hej." });

    expect(res.headers.get("content-type")).toBe("audio/mpeg");
    const bytes = new Uint8Array(await res.arrayBuffer());
    expect(bytes.length).toBeGreaterThan(0);
    expect(bytes[0]).toBe(0xff); // MP3 frame sync
  });

  it("returns a WAV Response for response_format 'wav'", async () => {
    const { engine } = fakeEngine();
    const tts = createOpenAISpeech(engine);
    const res = await tts.create({ voice: "alloy", input: "Hej.", response_format: "wav" });

    expect(res.headers.get("content-type")).toBe("audio/wav");
    const bytes = new Uint8Array(await res.arrayBuffer());
    expect(String.fromCharCode(...bytes.slice(0, 4))).toBe("RIFF");
  });

  it("returns raw PCM bytes for response_format 'pcm'", async () => {
    const { engine } = fakeEngine(new Float32Array([1, -1]));
    const tts = createOpenAISpeech(engine);
    const res = await tts.create({ voice: "alloy", input: "Hej.", response_format: "pcm" });

    expect(res.headers.get("content-type")).toBe("audio/pcm");
    const view = new DataView(await res.arrayBuffer());
    expect(view.getInt16(0, true)).toBe(32767); // +1.0
    expect(view.getInt16(2, true)).toBe(-32768); // -1.0
  });
});

describe("createOpenAISpeech — mapping", () => {
  it("maps OpenAI voice names to engine voices", async () => {
    const { engine, requests } = fakeEngine();
    await createOpenAISpeech(engine).create({ voice: "alloy", input: "Hej." });
    expect(requests[0].voice).toBe("ida"); // alloy -> ida
  });

  it("passes a native engine voice id through", async () => {
    const { engine, requests } = fakeEngine();
    await createOpenAISpeech(engine).create({ voice: "tor", input: "Hej." });
    expect(requests[0].voice).toBe("tor");
  });

  it("honors a custom voice map override", async () => {
    const { engine, requests } = fakeEngine();
    await createOpenAISpeech(engine, { voiceMap: { alloy: "tor" } }).create({
      voice: "alloy",
      input: "Hej.",
    });
    expect(requests[0].voice).toBe("tor");
  });

  it("forwards the input text", async () => {
    const { engine, requests } = fakeEngine();
    await createOpenAISpeech(engine).create({ voice: "alloy", input: "Goddag verden." });
    expect(requests[0].text).toBe("Goddag verden.");
  });

  it("forwards the abort signal", async () => {
    const { engine, requests } = fakeEngine();
    const controller = new AbortController();
    await createOpenAISpeech(engine).create({ voice: "alloy", input: "Hej." }, { signal: controller.signal });
    expect(requests[0].signal).toBe(controller.signal);
  });
});

describe("createOpenAISpeech — rejected requests", () => {
  it("rejects an unknown voice", async () => {
    const { engine } = fakeEngine();
    await expect(
      createOpenAISpeech(engine).create({ voice: "nonexistent-speaker", input: "Hej." }),
    ).rejects.toBeInstanceOf(UnknownVoiceError);
  });

  it("rejects unsupported formats (opus/aac/flac)", async () => {
    const { engine } = fakeEngine();
    await expect(
      createOpenAISpeech(engine).create({ voice: "alloy", input: "Hej.", response_format: "opus" }),
    ).rejects.toBeInstanceOf(UnsupportedFormatError);
  });

  it("rejects non-unit speed but accepts 1", async () => {
    const { engine } = fakeEngine();
    const tts = createOpenAISpeech(engine);
    await expect(tts.create({ voice: "alloy", input: "Hej.", speed: 1.5 })).rejects.toBeInstanceOf(
      UnsupportedSpeedError,
    );
    await expect(tts.create({ voice: "alloy", input: "Hej.", speed: 1 })).resolves.toBeInstanceOf(Response);
  });
});
