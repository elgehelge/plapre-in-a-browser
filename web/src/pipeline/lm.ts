// Stage 2: autoregressive LM generation on onnxruntime-web.
//
// The Plapre LM is conditioned by prepending a precomputed speaker hidden vector
// as the FIRST input embedding, then sampling audio-token ids (see
// docs/ARCHITECTURE.md, Stage 2). This replaces the reference vLLM path with a
// hand-rolled ONNX decode loop (token embedding + KV cache + JS sampling).
//
// The decode loop talks to the model through the narrow `LmGraph` seam, so the
// loop logic (prompt build, speaker prepend, sampling, KV-cache feedback, stop
// conditions) is unit-tested with a fake graph, independent of any ONNX export.
// `OrtLmGraph` is the production implementation over the exported lm/model.onnx.
//
// Export contract (conversion/export_lm.py): a single graph that embeds the
// token ids internally and prepends an optional prefix-embedding row, so the
// embedding matrix never has to leave the graph.
//
//   inputs : input_ids            int64   [seq]
//            prefix_embeds         float32 [k, hidden]   (k=1 prefill, k=0 decode)
//            past_key_values.{i}.{key,value} float32 [1, kvHeads, past, headDim]
//   outputs: logits               float32 [1, k+seq, vocab]
//            present.{i}.{key,value}         float32 [1, kvHeads, past+k+seq, headDim]

import { type SamplingParams, sample, makeRng } from "./sampling.js";
import { createSession, ort } from "./ort.js";
import { artifactUrl, ARTIFACTS } from "./assets.js";
import { MissingModelError, type Backend } from "./types.js";

export interface GenerateOptions extends SamplingParams {
  maxTokens: number;
  seed: number;
}

/** Opaque KV cache carried between decode steps. */
export type KvCache = Map<string, ort.Tensor>;

export interface LmStep {
  /** Logits for the LAST position only: [vocab]. */
  readonly logits: Float32Array;
  readonly present: KvCache;
}

/**
 * Narrow seam over the LM graph. The decode loop depends only on this, so it can
 * be tested with a fake; `OrtLmGraph` is the onnxruntime-web implementation.
 */
export interface LmGraph {
  readonly hidden: number;
  emptyCache(): KvCache;
  /**
   * Run one chunk. `prefixEmbeds` is `prefixLen * hidden` floats prepended before
   * the embedded `inputIds` (prefixLen is 1 on prefill to inject the speaker, 0
   * on decode steps). Returns the last-position logits + updated cache.
   */
  forward(
    inputIds: number[],
    prefixEmbeds: Float32Array,
    prefixLen: number,
    past: KvCache,
  ): Promise<LmStep>;
}

interface LmMeta {
  numLayers: number;
  kvHeads: number;
  headDim: number;
  hidden: number;
}

/** The slice of the tokenizer the decode loop needs (PlapreTokenizer satisfies it). */
export interface PromptTokenizer {
  buildPrompt(text: string): number[];
  readonly special: { readonly eos: number };
}

export class PlapreLM {
  private constructor(
    private readonly graph: LmGraph,
    private readonly tokenizer: PromptTokenizer,
  ) {}

  static async load(tokenizer: PromptTokenizer, backend: Backend): Promise<PlapreLM> {
    const graph = await OrtLmGraph.load(backend);
    return new PlapreLM(graph, tokenizer);
  }

  /** Construct over an injected graph — used by tests and alternative runtimes. */
  static withGraph(graph: LmGraph, tokenizer: PromptTokenizer): PlapreLM {
    return new PlapreLM(graph, tokenizer);
  }

  /**
   * Generate audio-token ids for one normalized sentence.
   *
   * @param normalizedText already run through normalizeText()
   * @param speakerHidden  precomputed speaker_proj(emb) hidden vector [hidden]
   */
  async generate(
    normalizedText: string,
    speakerHidden: Float32Array,
    opts: GenerateOptions,
    signal?: AbortSignal,
  ): Promise<number[]> {
    if (speakerHidden.length !== this.graph.hidden) {
      throw new Error(
        `speaker hidden length ${speakerHidden.length} != model hidden ${this.graph.hidden}`,
      );
    }
    const promptIds = this.tokenizer.buildPrompt(normalizedText);
    const { eos } = this.tokenizer.special;
    const rng = makeRng(opts.seed);
    const noPrefix = new Float32Array(0);

    signal?.throwIfAborted();
    // Prefill: embed the prompt and prepend the speaker hidden vector (prefixLen=1).
    let step = await this.graph.forward(promptIds, speakerHidden, 1, this.graph.emptyCache());

    const ids: number[] = [];
    for (let n = 0; n < opts.maxTokens; n++) {
      // Honor cancellation mid-generation so long sentences interrupt promptly,
      // not only between sentences.
      signal?.throwIfAborted();
      const id = sample(step.logits, opts, rng);
      if (id === eos) break;
      ids.push(id);
      // Decode step: feed the single new token back, no prefix (prefixLen=0).
      step = await this.graph.forward([id], noPrefix, 0, step.present);
    }
    return ids;
  }
}

/** onnxruntime-web implementation of the LM graph (KV cache + last-logits slice). */
export class OrtLmGraph implements LmGraph {
  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly meta: LmMeta,
    private readonly layers: number[],
  ) {}

  get hidden(): number {
    return this.meta.hidden;
  }

  static async load(backend: Backend): Promise<OrtLmGraph> {
    return OrtLmGraph.fromUrls(artifactUrl("lm"), artifactUrl("lmMeta"), backend, () =>
      new MissingModelError("lm/meta.json", ARTIFACTS.lmMeta.producedBy),
    );
  }

  /** Load from explicit URLs (used by the toy harness as well as load()). */
  static async fromUrls(
    modelUrl: string,
    metaUrl: string,
    backend: Backend,
    onMissingMeta: () => Error = () => new Error(`missing LM meta: ${metaUrl}`),
  ): Promise<OrtLmGraph> {
    const metaRes = await fetch(metaUrl);
    if (!metaRes.ok) throw onMissingMeta();
    const meta = (await metaRes.json()) as LmMeta;
    const session = await createSession(modelUrl, backend);
    const layers = Array.from({ length: meta.numLayers }, (_, i) => i);
    return new OrtLmGraph(session, meta, layers);
  }

  emptyCache(): KvCache {
    const cache: KvCache = new Map();
    const dims = [1, this.meta.kvHeads, 0, this.meta.headDim];
    for (const i of this.layers) {
      cache.set(`past_key_values.${i}.key`, new ort.Tensor("float32", new Float32Array(0), dims));
      cache.set(`past_key_values.${i}.value`, new ort.Tensor("float32", new Float32Array(0), dims));
    }
    return cache;
  }

  async forward(
    inputIds: number[],
    prefixEmbeds: Float32Array,
    prefixLen: number,
    past: KvCache,
  ): Promise<LmStep> {
    const feeds: Record<string, ort.Tensor> = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(inputIds.map(BigInt)), [
        inputIds.length,
      ]),
      prefix_embeds: new ort.Tensor("float32", prefixEmbeds, [prefixLen, this.meta.hidden]),
    };
    for (const [name, tensor] of past) feeds[name] = tensor;

    const out = await this.session.run(feeds);

    const logitsTensor = out.logits ?? out[this.session.outputNames[0]];
    const vocab = logitsTensor.dims[logitsTensor.dims.length - 1];
    const seq = logitsTensor.dims[logitsTensor.dims.length - 2];
    const all = logitsTensor.data as Float32Array;
    const logits = all.slice((seq - 1) * vocab, seq * vocab); // last position

    // Roll present.* outputs into the next step's past_key_values.* inputs.
    const present: KvCache = new Map();
    for (const i of this.layers) {
      present.set(`past_key_values.${i}.key`, out[`present.${i}.key`]);
      present.set(`past_key_values.${i}.value`, out[`present.${i}.value`]);
    }
    return { logits, present };
  }
}
