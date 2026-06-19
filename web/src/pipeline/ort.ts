// Thin wrapper around onnxruntime-web session creation with backend selection.

import * as ort from "onnxruntime-web";
import type { Backend } from "./types.js";

let configured = false;

function configureOrt(): void {
  if (configured) return;
  // Serve the ORT wasm binaries from a CDN to avoid bundler asset wiring during
  // the PoC. The version MUST match the installed onnxruntime-web package, or
  // the JS glue and wasm mismatch (e.g. "e.getValue is not a function"). For an
  // extension build these must be packaged locally instead.
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
  configured = true;
}

export async function isWebGpuAvailable(): Promise<boolean> {
  const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
  if (!gpu) return false;
  try {
    const adapter = await gpu.requestAdapter();
    return adapter != null;
  } catch {
    return false;
  }
}

export async function pickBackend(preferred: Backend = "webgpu"): Promise<Backend> {
  if (preferred === "webgpu" && (await isWebGpuAvailable())) return "webgpu";
  return "wasm";
}

export async function createSession(
  modelUrl: string,
  backend: Backend,
): Promise<ort.InferenceSession> {
  configureOrt();
  return ort.InferenceSession.create(modelUrl, {
    executionProviders: [backend],
    // WebGPU EP: the extended-fusion pass produces a SkipLayerNormalization
    // kernel that rejects our LayerNorm bias ("Beta must be 1D"). "basic" keeps
    // cheap optimizations and skips that fusion; it was also fastest in the
    // Phase 0 gate. WASM has no such issue and runs "all".
    graphOptimizationLevel: backend === "webgpu" ? "basic" : "all",
  });
}

export { ort };
