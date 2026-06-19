// Stage 3a: Kanade decoder. content-token indices + 128-dim speaker emb -> mel.
//
// Maps audio-token ids to Kanade content-token indices (subtract <audio_0>),
// then runs the exported kanade_decoder.onnx.
//
// I/O names follow conversion/export_kanade_decoder.py
// (content_token_indices, global_embedding -> mel).

import { createSession, ort } from "./ort.js";
import { artifactUrl } from "./assets.js";
import type { Backend } from "./types.js";

export class KanadeDecoder {
  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly audioTokenStart: number,
    private readonly audioTokenEnd: number,
  ) {}

  static async load(
    audioTokenStart: number,
    audioTokenEnd: number,
    backend: Backend,
  ): Promise<KanadeDecoder> {
    const session = await createSession(artifactUrl("kanadeDecoder"), backend);
    return new KanadeDecoder(session, audioTokenStart, audioTokenEnd);
  }

  /** audio-token ids (from the LM) -> Kanade content-token indices. */
  toKanadeIndices(audioTokenIds: number[]): number[] {
    const out: number[] = [];
    for (const id of audioTokenIds) {
      if (id >= this.audioTokenStart && id <= this.audioTokenEnd) {
        out.push(id - this.audioTokenStart);
      }
    }
    return out;
  }

  /** Decode Kanade indices + raw 128-dim speaker emb to a mel spectrogram. */
  async decode(
    kanadeIndices: number[],
    speakerEmb: Float32Array,
  ): Promise<{ data: Float32Array; dims: readonly number[] }> {
    const tokens = new ort.Tensor(
      "int64",
      BigInt64Array.from(kanadeIndices.map((i) => BigInt(i))),
      [kanadeIndices.length],
    );
    const emb = new ort.Tensor("float32", speakerEmb, [speakerEmb.length]);

    const result = await this.session.run({
      content_token_indices: tokens,
      global_embedding: emb,
    });
    const mel = result.mel ?? Object.values(result)[0];
    return { data: mel.data as Float32Array, dims: mel.dims };
  }
}
