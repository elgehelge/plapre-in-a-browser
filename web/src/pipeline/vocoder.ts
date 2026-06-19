// Stage 3b: HiFT vocoder. mel spectrogram -> 24 kHz waveform.
//
// I/O names follow conversion/export_hift_vocoder.py (mel -> waveform).
// This is the Phase 0 gate: if hift_vocoder.onnx will not run under
// onnxruntime-web, the whole in-browser approach must be reconsidered.

import { createSession, ort } from "./ort.js";
import { artifactUrl } from "./assets.js";
import type { Backend } from "./types.js";

export class HiftVocoder {
  private constructor(private readonly session: ort.InferenceSession) {}

  static async load(backend: Backend): Promise<HiftVocoder> {
    const session = await createSession(artifactUrl("vocoder"), backend, {
      dataFile: "hift_vocoder.onnx.data",
    });
    return new HiftVocoder(session);
  }

  /**
   * @param mel  flat mel data
   * @param dims mel dims as produced by the decoder, e.g. [n_mels, T] or [1, n_mels, T]
   */
  async vocode(mel: Float32Array, dims: readonly number[]): Promise<Float32Array> {
    // Ensure a batch dim: model expects (B, n_mels, T).
    const batched = dims.length === 2 ? [1, dims[0], dims[1]] : [...dims];
    const melTensor = new ort.Tensor("float32", mel, batched);

    const result = await this.session.run({ mel: melTensor });
    // export_hift_vocoder.py names the output "wav".
    const wav = result.wav ?? Object.values(result)[0];
    return wav.data as Float32Array;
  }
}
