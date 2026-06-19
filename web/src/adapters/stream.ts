// Bridge the engine's PCM chunk stream to a ReadableStream of encoded bytes,
// shared by the provider adapters. Raw 16-bit PCM is emitted chunk-by-chunk as
// the engine produces it, so adapters can stream audio with low latency.

import type { Engine, SynthesisRequest } from "../engine/engine.js";
import { pcmToInt16LE } from "../audio/format.js";

export function pcmStream(engine: Engine, req: SynthesisRequest): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const chunk of engine.synthesize(req)) {
          controller.enqueue(pcmToInt16LE(chunk.samples));
        }
        controller.close();
      } catch (err) {
        controller.error(err);
      }
    },
  });
}
