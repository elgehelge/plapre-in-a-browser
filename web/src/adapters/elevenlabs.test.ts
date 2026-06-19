import { describe, it, expect } from "vitest";
import {
  createElevenLabsTextToSpeech,
  createElevenLabsVoices,
  createElevenLabsClient,
} from "./elevenlabs.js";
import type { DecodedAudio } from "../audio/decode.js";
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
  const clones: { audio: Float32Array; sampleRate: number; opts?: unknown }[] = [];
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
    canCloneVoice: () => true,
    async cloneVoice(audio, sampleRate, opts) {
      clones.push({ audio, sampleRate, opts });
      const id = opts?.id ?? `cloned-${clones.length}`;
      return { id, displayName: opts?.displayName ?? id, lang: "da-DK" };
    },
  };
  return { engine, requests, clones };
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

describe("createElevenLabsVoices — Instant Voice Cloning", () => {
  const fakeDecoder = (pcm: Float32Array, sampleRate: number) => async (): Promise<DecodedAudio> => ({
    pcm,
    sampleRate,
  });

  it("decodes reference files and clones onto the engine", async () => {
    const { engine, clones } = fakeEngine();
    const voices = createElevenLabsVoices(engine, {
      decodeAudio: fakeDecoder(new Float32Array(16000).fill(0.2), 16000),
    });
    const res = await voices.add({ name: "Custom", files: [new Uint8Array([1, 2, 3]).buffer] });

    expect(res.name).toBe("Custom");
    expect(res.voiceId).toBeTruthy();
    expect(clones).toHaveLength(1);
    expect(clones[0].sampleRate).toBe(16000);
    expect(clones[0].audio.length).toBe(16000);
  });

  it("accepts pre-decoded PCM and concatenates multiple clips", async () => {
    const { engine, clones } = fakeEngine();
    const voices = createElevenLabsVoices(engine); // no decoder needed for PCM inputs
    await voices.add({
      files: [
        { pcm: new Float32Array(100).fill(0.1), sampleRate: 24000 },
        { pcm: new Float32Array(50).fill(0.2), sampleRate: 24000 },
      ],
    });
    expect(clones[0].audio.length).toBe(150);
    expect(clones[0].sampleRate).toBe(24000);
  });

  it("rejects cloning when the engine doesn't support it", async () => {
    const { engine } = fakeEngine();
    const noClone = { ...engine, canCloneVoice: () => false };
    const voices = createElevenLabsVoices(noClone);
    await expect(
      voices.add({ files: [{ pcm: new Float32Array(10), sampleRate: 24000 }] }),
    ).rejects.toThrow();
  });

  it("createElevenLabsClient exposes textToSpeech + voices", () => {
    const { engine } = fakeEngine();
    const client = createElevenLabsClient(engine);
    expect(typeof client.textToSpeech.convert).toBe("function");
    expect(typeof client.voices.add).toBe("function");
  });
});
