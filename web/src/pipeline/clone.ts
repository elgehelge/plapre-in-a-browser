// Phase 5: voice cloning. Derive the same 128-dim Kanade global embedding that
// built-in speakers ship as, plus its projected LM hidden vector, from a piece
// of reference audio — entirely in the browser (the audio never leaves it).
//
//   waveform -> resample to model rate -> clone_encoder.onnx -> raw 128-dim
//            -> speaker_proj (128 -> hidden, applied in JS) -> hidden
//
// The ONNX encoder runs the WavLM frontend + GlobalEncoder (the op-support risk,
// retired by the Phase 5 gate). speaker_proj is a single Linear shipped as JSON
// and applied here, so the projection math is unit-tested without any session.

import { createSession, ort } from "./ort.js";
import { artifactUrl, ARTIFACTS } from "./assets.js";
import { resample } from "../audio/resample.js";
import { MissingModelError, SAMPLE_RATE, type Backend, type SpeakerData } from "./types.js";

/** A single Linear (128 -> hidden) applied in JS: y = W·x + b. */
export class SpeakerProjection {
  constructor(
    private readonly weight: Float32Array, // row-major [out][in]
    private readonly bias: Float32Array, // [out]
    readonly inDim: number,
    readonly outDim: number,
  ) {}

  static fromJSON(json: {
    in: number;
    out: number;
    weight: number[][];
    bias: number[];
  }): SpeakerProjection {
    const weight = new Float32Array(json.out * json.in);
    for (let r = 0; r < json.out; r++) weight.set(json.weight[r], r * json.in);
    return new SpeakerProjection(weight, Float32Array.from(json.bias), json.in, json.out);
  }

  apply(x: Float32Array): Float32Array {
    if (x.length !== this.inDim) {
      throw new Error(`projection input ${x.length} != ${this.inDim}`);
    }
    const out = new Float32Array(this.outDim);
    for (let r = 0; r < this.outDim; r++) {
      let acc = this.bias[r];
      const base = r * this.inDim;
      for (let c = 0; c < this.inDim; c++) acc += this.weight[base + c] * x[c];
      out[r] = acc;
    }
    return out;
  }
}

/** The encoder seam: reference audio -> raw 128-dim embedding. */
export interface CloneEmbedder {
  /** Embed a mono waveform (at any sample rate) into the raw 128-dim embedding. */
  embed(audio: Float32Array, sampleRate: number): Promise<Float32Array>;
}

export class OrtCloneEmbedder implements CloneEmbedder {
  constructor(
    private readonly session: ort.InferenceSession,
    private readonly modelSampleRate: number = SAMPLE_RATE,
  ) {}

  static async load(backend: Backend): Promise<OrtCloneEmbedder> {
    const session = await createSession(artifactUrl("cloneEncoder"), backend);
    return new OrtCloneEmbedder(session);
  }

  async embed(audio: Float32Array, sampleRate: number): Promise<Float32Array> {
    const wav = sampleRate === this.modelSampleRate ? audio : resample(audio, sampleRate, this.modelSampleRate);
    const tensor = new ort.Tensor("float32", wav, [1, wav.length]);
    const out = await this.session.run({ waveform: tensor });
    const emb = out.embedding ?? out[this.session.outputNames[0]];
    return emb.data as Float32Array;
  }
}

/**
 * Cloner: turns reference audio into a `SpeakerData` (raw + hidden) ready to
 * register as a runtime voice. Both halves are kept so the decoder gets the raw
 * 128-dim emb and the LM gets the projected hidden vector — exactly like a
 * built-in speaker.
 */
export class VoiceClonerImpl {
  constructor(
    private readonly embedder: CloneEmbedder,
    private readonly projection: SpeakerProjection,
  ) {}

  static async load(backend: Backend): Promise<VoiceClonerImpl> {
    const res = await fetch(artifactUrl("speakerProj"));
    if (!res.ok) {
      throw new MissingModelError("speaker_proj.json", ARTIFACTS.speakerProj.producedBy);
    }
    const projection = SpeakerProjection.fromJSON(await res.json());
    const embedder = await OrtCloneEmbedder.load(backend);
    return new VoiceClonerImpl(embedder, projection);
  }

  static withParts(embedder: CloneEmbedder, projection: SpeakerProjection): VoiceClonerImpl {
    return new VoiceClonerImpl(embedder, projection);
  }

  async embedSpeaker(audio: Float32Array, sampleRate: number): Promise<SpeakerData> {
    if (audio.length === 0) throw new Error("cannot clone from empty audio");
    const raw = await this.embedder.embed(audio, sampleRate);
    const hidden = this.projection.apply(raw);
    return { raw: Array.from(raw), hidden: Array.from(hidden) };
  }
}
