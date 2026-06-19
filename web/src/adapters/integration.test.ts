import { describe, it, expect } from "vitest";
import { createEngine, NATIVE_SAMPLE_RATE } from "../engine/engine.js";
import type { SentenceRequest, SpeechModel } from "../engine/speech-model.js";
import type { Voice } from "../engine/voice.js";
import { createOpenAISpeech } from "./openai.js";
import { createElevenLabsTextToSpeech } from "./elevenlabs.js";

// End-to-end composition through the *real* engine (sentence splitting,
// streaming, concatenation) and a fake SpeechModel — verifying the layers
// (adapter -> engine -> encoder) interact correctly, not just in isolation.

const VOICES: Voice[] = [{ id: "ida", displayName: "Ida", lang: "da-DK" }];
const PER_SENTENCE = NATIVE_SAMPLE_RATE / 4; // 0.25 s of audio per sentence

const model: SpeechModel = {
  voices: () => VOICES,
  async synthesizeSentence(_req: SentenceRequest) {
    return new Float32Array(PER_SENTENCE).fill(0.5);
  },
};

async function byteLength(res: Response): Promise<number> {
  return (await res.arrayBuffer()).byteLength;
}

describe("OpenAI adapter over the real engine", () => {
  it("synthesizes each sentence and concatenates into one WAV", async () => {
    const tts = createOpenAISpeech(createEngine(model));
    const res = await tts.create({ voice: "ida", input: "En. To. Tre.", response_format: "wav" });
    // 3 sentences * PER_SENTENCE samples * 2 bytes + 44-byte WAV header
    expect(await byteLength(res)).toBe(3 * PER_SENTENCE * 2 + 44);
  });

  it("streams raw PCM for every sentence", async () => {
    const tts = createOpenAISpeech(createEngine(model));
    const res = await tts.create({ voice: "ida", input: "En. To.", response_format: "pcm" });
    expect(await byteLength(res)).toBe(2 * PER_SENTENCE * 2);
  });
});

describe("ElevenLabs adapter over the real engine", () => {
  it("converts multi-sentence text to raw PCM (pcm_24000)", async () => {
    const tts = createElevenLabsTextToSpeech(createEngine(model));
    const res = await tts.convert("ida", {
      text: "En. To. Tre. Fire.",
      outputFormat: "pcm_24000",
    });
    expect(await byteLength(res)).toBe(4 * PER_SENTENCE * 2);
  });

  it("converts multi-sentence text to MP3 by default", async () => {
    const tts = createElevenLabsTextToSpeech(createEngine(model));
    const res = await tts.convert("ida", { text: "En. To. Tre. Fire." });
    expect(res.headers.get("content-type")).toBe("audio/mpeg");
    expect(await byteLength(res)).toBeGreaterThan(0);
  });
});
