import { describe, it, expect } from "vitest";
import { createElevenLabsTextToSpeech } from "./elevenlabs.js";
import { UnsupportedFormatError } from "../audio/format.js";
import { UnsupportedSpeedError } from "./errors.js";
import { UnknownVoiceError, type Voice } from "../engine/voice.js";
import { NATIVE_SAMPLE_RATE, type Engine, type SynthesisRequest } from "../engine/engine.js";

const VOICES: Voice[] = [
  { id: "ida", displayName: "Ida", lang: "da-DK" },
  { id: "tor", displayName: "Tor", lang: "da-DK" },
];

function fakeEngine(pcm = new Float32Array([1, -1])) {
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

async function bytesOf(stream: ReadableStream<Uint8Array>): Promise<Uint8Array> {
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

describe("createElevenLabsTextToSpeech — convert", () => {
  it("returns MP3 by default (matching ElevenLabs' mp3_44100_128)", async () => {
    const { engine } = fakeEngine(new Float32Array(2048).fill(0.2));
    const res = await createElevenLabsTextToSpeech(engine).convert("ida", { text: "Hej." });

    expect(res.headers.get("content-type")).toBe("audio/mpeg");
    const bytes = new Uint8Array(await res.arrayBuffer());
    expect(bytes.length).toBeGreaterThan(0);
    expect(bytes[0]).toBe(0xff); // MP3 frame sync
  });

  it("returns raw PCM for outputFormat 'pcm_24000'", async () => {
    const { engine } = fakeEngine(new Float32Array([1, -1]));
    const res = await createElevenLabsTextToSpeech(engine).convert("ida", {
      text: "Hej.",
      outputFormat: "pcm_24000",
    });

    expect(res.headers.get("content-type")).toBe("audio/pcm");
    const view = new DataView(await res.arrayBuffer());
    expect(view.getInt16(0, true)).toBe(32767);
    expect(view.getInt16(2, true)).toBe(-32768);
  });

  it("resamples raw PCM to the requested rate", async () => {
    const { engine } = fakeEngine(new Float32Array(2400).fill(0.1)); // 0.1 s @ 24 kHz
    const res = await createElevenLabsTextToSpeech(engine).convert("ida", {
      text: "Hej.",
      outputFormat: "pcm_44100",
    });
    const samples = (await res.arrayBuffer()).byteLength / 2;
    expect(samples).toBe(Math.round(2400 * (44100 / 24000))); // 4410
  });

  it("uses the voice id directly as the engine voice", async () => {
    const { engine, requests } = fakeEngine();
    await createElevenLabsTextToSpeech(engine).convert("tor", { text: "Hej." });
    expect(requests[0].voice).toBe("tor");
  });

  it("honors a voice map override", async () => {
    const { engine, requests } = fakeEngine();
    await createElevenLabsTextToSpeech(engine, { voiceMap: { "my-clone": "ida" } }).convert(
      "my-clone",
      { text: "Hej." },
    );
    expect(requests[0].voice).toBe("ida");
  });

  it("forwards text and abort signal", async () => {
    const { engine, requests } = fakeEngine();
    const controller = new AbortController();
    await createElevenLabsTextToSpeech(engine).convert(
      "ida",
      { text: "Goddag." },
      { signal: controller.signal },
    );
    expect(requests[0].text).toBe("Goddag.");
    expect(requests[0].signal).toBe(controller.signal);
  });

  it("accepts (but ignores) non-speed voice settings", async () => {
    const { engine } = fakeEngine();
    await expect(
      createElevenLabsTextToSpeech(engine).convert("ida", {
        text: "Hej.",
        voiceSettings: { stability: 0.7, similarityBoost: 0.8, style: 0.1, useSpeakerBoost: true },
      }),
    ).resolves.toBeInstanceOf(Response);
  });
});

describe("createElevenLabsTextToSpeech — stream", () => {
  it("streams raw PCM bytes for pcm_24000", async () => {
    const { engine } = fakeEngine(new Float32Array([1, -1]));
    const stream = createElevenLabsTextToSpeech(engine).stream("ida", {
      text: "Hej.",
      outputFormat: "pcm_24000",
    });
    const bytes = await bytesOf(stream);
    expect(bytes.length).toBe(4); // 2 samples * 2 bytes
  });

  it("streams MP3 bytes by default", async () => {
    const { engine } = fakeEngine(new Float32Array(2048).fill(0.2));
    const stream = createElevenLabsTextToSpeech(engine).stream("ida", { text: "Hej." });
    const bytes = await bytesOf(stream);
    expect(bytes.length).toBeGreaterThan(0);
    expect(bytes[0]).toBe(0xff);
  });

  it("validates eagerly: an unknown voice throws synchronously", () => {
    const { engine } = fakeEngine();
    expect(() => createElevenLabsTextToSpeech(engine).stream("missing", { text: "Hej." })).toThrow(
      UnknownVoiceError,
    );
  });
});

describe("createElevenLabsTextToSpeech — rejected requests", () => {
  it("rejects an unknown voice", async () => {
    const { engine } = fakeEngine();
    await expect(
      createElevenLabsTextToSpeech(engine).convert("missing", { text: "Hej." }),
    ).rejects.toBeInstanceOf(UnknownVoiceError);
  });

  it("rejects unsupported output formats (codec or rate)", async () => {
    const { engine } = fakeEngine();
    const tts = createElevenLabsTextToSpeech(engine);
    await expect(
      tts.convert("ida", { text: "Hej.", outputFormat: "ulaw_8000" }),
    ).rejects.toBeInstanceOf(UnsupportedFormatError);
    await expect(
      tts.convert("ida", { text: "Hej.", outputFormat: "mp3_48000_128" }), // unsupported mp3 rate
    ).rejects.toBeInstanceOf(UnsupportedFormatError);
  });

  it("rejects non-unit speed but accepts 1", async () => {
    const { engine } = fakeEngine();
    const tts = createElevenLabsTextToSpeech(engine);
    await expect(
      tts.convert("ida", { text: "Hej.", voiceSettings: { speed: 0.8 } }),
    ).rejects.toBeInstanceOf(UnsupportedSpeedError);
    await expect(
      tts.convert("ida", { text: "Hej.", voiceSettings: { speed: 1 } }),
    ).resolves.toBeInstanceOf(Response);
  });
});
