// Plapre as a SpeechModel: it owns the model runtime (LM + Kanade decoder + HiFT
// vocoder), the tokenizer, and the built-in speaker embeddings, and turns one
// normalized sentence into mono PCM. Text orchestration (splitting, streaming,
// cancellation) lives in the provider-neutral Engine above this; see
// docs/INTERFACE.md and src/engine/.

import { PlapreTokenizer } from "./tokenizer.js";
import { PlapreLM } from "./lm.js";
import { KanadeDecoder } from "./decoder.js";
import { HiftVocoder } from "./vocoder.js";
import { loadSpeakers } from "./speakers.js";
import { VoiceClonerImpl } from "./clone.js";
import {
  resolveBackends,
  type BackendChoice,
  type LoadOptions,
  type ResolvedBackends,
} from "./ort.js";
import { setModelsBaseUrl } from "./assets.js";
import type { Backend, SpeakerTable } from "./types.js";
import { createEngine, type Engine, type EngineOptions } from "../engine/engine.js";
import type {
  CloneVoiceOptions,
  SentenceRequest,
  SpeechModel,
  VoiceCloner,
} from "../engine/speech-model.js";
import type { Voice } from "../engine/voice.js";

const DANISH = "da-DK";

export interface PlapreConfig extends EngineOptions, LoadOptions {
  /**
   * ONNX Runtime backend for the autoregressive LM. "auto" (default) runs it on
   * threaded WASM when the page is cross-origin isolated (the LM's one-token-at-
   * a-time loop beats WebGPU's per-dispatch overhead), else WebGPU, else
   * single-threaded WASM.
   */
  backend_lm?: BackendChoice;
  /**
   * ONNX Runtime backend for the decoder + vocoder (and clone encoder). "auto"
   * (default) uses WebGPU when available — these stages are large and parallel —
   * else WASM.
   */
  backend_codec?: BackendChoice;
  /**
   * Base URL the converted artifacts are served from. Defaults to `/models`
   * (artifacts hosted next to the app). Point it at a CDN or GitHub Release to
   * consume the library without copying artifacts into your own `public/`.
   */
  modelsBaseUrl?: string;
}

export class PlapreSpeechModel implements SpeechModel, VoiceCloner {
  private cloner: VoiceClonerImpl | null = null;
  private cloneCounter = 0;
  private readonly displayNames = new Map<string, string>();

  private constructor(
    private readonly lm: PlapreLM,
    private readonly decoder: KanadeDecoder,
    private readonly vocoder: HiftVocoder,
    private readonly speakers: SpeakerTable,
    private readonly codecBackend: Backend,
    private readonly loadOpts: LoadOptions,
  ) {}

  static async load(
    backends: ResolvedBackends,
    opts: LoadOptions = {},
  ): Promise<PlapreSpeechModel> {
    const tokenizer = await PlapreTokenizer.load();
    const [lm, speakers] = await Promise.all([
      PlapreLM.load(tokenizer, backends.lm, opts),
      loadSpeakers(),
    ]);
    const { audioTokenStart, audioTokenEnd } = tokenizer.special;
    const [decoder, vocoder] = await Promise.all([
      KanadeDecoder.load(audioTokenStart, audioTokenEnd, backends.codec, opts),
      HiftVocoder.load(backends.codec, opts),
    ]);
    return new PlapreSpeechModel(lm, decoder, vocoder, speakers, backends.codec, opts);
  }

  voices(): readonly Voice[] {
    return Object.keys(this.speakers).map((id) => ({
      id,
      displayName: this.displayNames.get(id) ?? id,
      lang: DANISH,
    }));
  }

  /** Clone a voice from reference audio and register it in the speaker table. */
  async cloneVoice(
    audio: Float32Array,
    sampleRate: number,
    opts: CloneVoiceOptions = {},
  ): Promise<Voice> {
    this.cloner ??= await VoiceClonerImpl.load(this.codecBackend, this.loadOpts);
    const id = opts.id ?? `cloned-${++this.cloneCounter}`;
    if (this.speakers[id]) throw new Error(`voice id "${id}" already exists`);
    this.speakers[id] = await this.cloner.embedSpeaker(audio, sampleRate);
    const displayName = opts.displayName ?? id;
    this.displayNames.set(id, displayName);
    return { id, displayName, lang: opts.lang ?? DANISH };
  }

  async synthesizeSentence({
    sentence,
    voice,
    generation,
    signal,
  }: SentenceRequest): Promise<Float32Array> {
    const speaker = this.speakers[voice.id];
    if (!speaker) {
      // The Engine resolves voices against voices() before calling, so this is
      // an internal invariant violation rather than a user error.
      throw new Error(`Speaker embedding missing for resolved voice "${voice.id}"`);
    }
    const audioTokens = await this.lm.generate(
      sentence,
      Float32Array.from(speaker.hidden),
      generation,
      signal,
    );
    const kanadeIndices = this.decoder.toKanadeIndices(audioTokens);
    if (kanadeIndices.length === 0) return new Float32Array(0);
    const mel = await this.decoder.decode(kanadeIndices, Float32Array.from(speaker.raw));
    return this.vocoder.vocode(mel.data, mel.dims);
  }
}

/** Load the full Plapre pipeline and expose it as a provider-neutral Engine. */
export async function loadPlapreEngine(config: PlapreConfig = {}): Promise<Engine> {
  if (config.modelsBaseUrl) setModelsBaseUrl(config.modelsBaseUrl);
  const backends = await resolveBackends({
    lm: config.backend_lm,
    codec: config.backend_codec,
  });
  const model = await PlapreSpeechModel.load(backends, { cache: config.cache });
  return createEngine(model, {
    generation: config.generation,
    interSentenceSilenceSec: config.interSentenceSilenceSec,
  });
}
