// Orchestrator: ties the four stages together into a single synthesize() call.
//
//   text -> normalize -> LM (audio tokens) -> Kanade decode (mel) -> vocode (PCM)

import { normalizeText, splitSentences } from "./normalize.js";
import { PlapreTokenizer } from "./tokenizer.js";
import { PlapreLM, type GenerateOptions } from "./lm.js";
import { KanadeDecoder } from "./decoder.js";
import { HiftVocoder } from "./vocoder.js";
import { loadSpeakers } from "./speakers.js";
import { pickBackend } from "./ort.js";
import {
  SAMPLE_RATE,
  type Backend,
  type SpeakerTable,
  type SynthResult,
} from "./types.js";

export interface PlapreConfig {
  backend?: Backend;
  generate?: Partial<GenerateOptions>;
}

const DEFAULT_GENERATE: GenerateOptions = {
  temperature: 0.8,
  topK: 50,
  topP: 0.95,
  maxTokens: 500,
  seed: 0,
};

export class Plapre {
  private constructor(
    readonly backend: Backend,
    private readonly lm: PlapreLM,
    private readonly decoder: KanadeDecoder,
    private readonly vocoder: HiftVocoder,
    private readonly speakers: SpeakerTable,
    private readonly genDefaults: GenerateOptions,
  ) {}

  static async load(config: PlapreConfig = {}): Promise<Plapre> {
    const backend = await pickBackend(config.backend ?? "webgpu");
    const tokenizer = await PlapreTokenizer.load();
    const [lm, speakers] = await Promise.all([
      PlapreLM.load(tokenizer, backend),
      loadSpeakers(),
    ]);
    const { audioTokenStart, audioTokenEnd } = tokenizer.special;
    const [decoder, vocoder] = await Promise.all([
      KanadeDecoder.load(audioTokenStart, audioTokenEnd, backend),
      HiftVocoder.load(backend),
    ]);
    return new Plapre(backend, lm, decoder, vocoder, speakers, {
      ...DEFAULT_GENERATE,
      ...config.generate,
    });
  }

  listSpeakers(): string[] {
    return Object.keys(this.speakers);
  }

  async synthesize(
    text: string,
    speaker: string,
    splitIntoSentences = false,
  ): Promise<SynthResult> {
    const spk = this.speakers[speaker];
    if (!spk) {
      throw new Error(`Unknown speaker "${speaker}". Available: ${this.listSpeakers()}`);
    }
    const hidden = Float32Array.from(spk.hidden);
    const rawEmb = Float32Array.from(spk.raw);

    const sentences = splitIntoSentences
      ? splitSentences(text)
      : [normalizeText(text)];

    const chunks: Float32Array[] = [];
    for (const sentence of sentences) {
      const audioTokens = await this.lm.generate(sentence, hidden, this.genDefaults);
      const kanadeIdx = this.decoder.toKanadeIndices(audioTokens);
      if (kanadeIdx.length === 0) continue;
      const mel = await this.decoder.decode(kanadeIdx, rawEmb);
      const pcm = await this.vocoder.vocode(mel.data, mel.dims);
      chunks.push(pcm);
    }

    return { pcm: concatFloat32(chunks), sampleRate: SAMPLE_RATE };
  }
}

function concatFloat32(chunks: Float32Array[]): Float32Array {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}
