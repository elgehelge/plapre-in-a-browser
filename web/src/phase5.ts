// Phase 5 in-browser gate: run the clone encoder (WavLM frontend + GlobalEncoder)
// under onnxruntime-web and check the 128-dim embedding matches the ORT-CPU/torch
// golden (cosine). This retires the WavLM-under-ORT-Web op-support risk.
// Backend via ?backend=wasm|webgpu; result on window.__phase5.

import { OrtCloneEmbedder } from "./pipeline/clone.js";
import type { Backend } from "./pipeline/types.js";
import { createSession } from "./pipeline/ort.js";

function log(msg: string): void {
  const el = document.getElementById("out");
  if (el) el.textContent += msg + "\n";
}

function cosine(a: Float32Array, b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

interface CloneGolden {
  sampleRate: number;
  waveform: number[];
  embedding: number[];
}

async function run(): Promise<void> {
  const backend = (new URLSearchParams(location.search).get("backend") ?? "wasm") as Backend;
  log(`backend: ${backend}`);
  try {
    const golden = (await (await fetch("/models/clone_golden.json")).json()) as CloneGolden;

    const t0 = performance.now();
    const session = await createSession("/models/clone_encoder.onnx", backend);
    const tLoadMs = performance.now() - t0;
    const embedder = new OrtCloneEmbedder(session, golden.sampleRate);

    const waveform = Float32Array.from(golden.waveform);
    const t1 = performance.now();
    const emb = await embedder.embed(waveform, golden.sampleRate);
    const tRunMs = performance.now() - t1;

    const cos = cosine(emb, golden.embedding);
    const ok = cos >= 0.999;
    const result = {
      backend,
      ok,
      dim: emb.length,
      cosine: Number(cos.toFixed(6)),
      tLoadMs: Math.round(tLoadMs),
      tRunMs: Math.round(tRunMs),
    };
    (window as unknown as { __phase5: unknown }).__phase5 = result;
    log(JSON.stringify(result, null, 2));
  } catch (err) {
    (window as unknown as { __phase5: unknown }).__phase5 = { backend, ok: false, error: String(err) };
    log("ERROR: " + String(err));
  }
}

void run();
