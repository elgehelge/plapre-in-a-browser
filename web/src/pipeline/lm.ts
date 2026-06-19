// Stage 2: autoregressive LM generation on onnxruntime-web.
//
// The Plapre LM is conditioned by prepending a precomputed speaker hidden vector
// as the FIRST input embedding, then sampling audio-token ids (see
// docs/ARCHITECTURE.md, Stage 2). This replaces the reference vLLM path with a
// hand-rolled ONNX loop (token embedding + KV cache + JS sampling).
//
// The exact input/output names and the inputs_embeds wiring depend on how the
// LM is exported (conversion/export_lm.py). Those spots are marked TODO; the
// surrounding structure (prompt build, sampling, stop conditions) is final.

import type { PlapreTokenizer } from "./tokenizer.js";
import { type SamplingParams, sample, makeRng } from "./sampling.js";
import { createSession, ort } from "./ort.js";
import { artifactUrl } from "./assets.js";
import type { Backend } from "./types.js";

export interface GenerateOptions extends SamplingParams {
  maxTokens: number;
  seed: number;
}

export class PlapreLM {
  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly tokenizer: PlapreTokenizer,
  ) {}

  static async load(tokenizer: PlapreTokenizer, backend: Backend): Promise<PlapreLM> {
    const session = await createSession(artifactUrl("lm"), backend);
    return new PlapreLM(session, tokenizer);
  }

  /**
   * Generate audio-token ids for one normalized sentence.
   *
   * @param normalizedText already run through normalizeText()
   * @param speakerHidden  precomputed speaker_proj(emb) hidden vector
   */
  async generate(
    normalizedText: string,
    speakerHidden: Float32Array,
    opts: GenerateOptions,
  ): Promise<number[]> {
    const promptIds = this.tokenizer.buildPrompt(normalizedText);
    const rng = makeRng(opts.seed);
    const { audioTokenStart, audioTokenEnd, eos } = this.tokenizer.special;

    void this.session;
    void promptIds;
    void speakerHidden;
    void rng;
    void sample;
    void audioTokenStart;
    void audioTokenEnd;
    void eos;

    // TODO (Phase 1): implement the decode loop:
    //   1. inputs_embeds = concat([speakerHidden, embed_tokens(promptIds)])
    //      - either export the LM to take inputs_embeds, or run an embed_tokens
    //        gather ONNX and concat here.
    //   2. prime the session, read logits for the last position.
    //   3. loop: sample() next id with `opts`; stop on eos or maxTokens or when
    //      the id leaves the [audioTokenStart, audioTokenEnd] range; feed the id
    //      back with the returned KV cache (present.* -> past.*).
    //   4. collect ids within the audio range and return them.
    throw new Error(
      "PlapreLM.generate not wired yet — complete the ONNX decode loop (Phase 1). " +
        "See conversion/export_lm.py for the export contract.",
    );
  }
}
