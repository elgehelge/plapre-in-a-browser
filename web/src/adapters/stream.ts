// Bridge the engine's PCM chunk stream to a ReadableStream of encoded bytes,
// shared by the provider adapters. Raw 16-bit PCM is emitted chunk-by-chunk as
// the engine produces it, so adapters can stream audio with low latency.
//
// Cancellation flows both ways: the caller's AbortSignal aborts the stream, and
// the consumer cancelling the stream (e.g. dropping `response.body`) aborts the
// engine so it stops generating. Granularity is per-sentence (the engine checks
// the signal between sentences).

import type { Engine, SynthesisRequest } from "../engine/engine.js";
import { pcmToInt16LE } from "../audio/format.js";

export function pcmStream(engine: Engine, req: SynthesisRequest): ReadableStream<Uint8Array> {
  const abort = new AbortController();
  if (req.signal) {
    if (req.signal.aborted) abort.abort(req.signal.reason);
    else req.signal.addEventListener("abort", () => abort.abort(req.signal?.reason), { once: true });
  }
  const request: SynthesisRequest = { ...req, signal: abort.signal };
  let cancelled = false;

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const chunk of engine.synthesize(request)) {
          controller.enqueue(pcmToInt16LE(chunk.samples));
        }
        controller.close();
      } catch (err) {
        // After a consumer cancel() the stream is already closed; re-erroring it
        // would throw. Only surface genuine synthesis errors.
        if (!cancelled) controller.error(err);
      }
    },
    cancel(reason) {
      cancelled = true;
      abort.abort(reason);
    },
  });
}
