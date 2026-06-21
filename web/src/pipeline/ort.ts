// Thin wrapper around onnxruntime-web session creation with backend selection.

import * as ort from "onnxruntime-web";
import type { Backend } from "./types.js";
import { fetchCached, type ProgressFn } from "./model-cache.js";

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

/**
 * Whether multi-threaded WASM is available. ORT-Web's threaded WASM uses
 * SharedArrayBuffer, which the browser only exposes on a cross-origin-isolated
 * page (COOP/COEP). Single-threaded WASM still works without it, just slower.
 */
export function isThreadedWasmAvailable(): boolean {
  return (
    typeof SharedArrayBuffer !== "undefined" &&
    (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true
  );
}

/** Per-stage backend preference. "auto" picks the best for the environment. */
export type BackendChoice = "webgpu" | "wasm" | "auto";

export interface BackendConfig {
  /** Backend for the autoregressive LM. Default "auto". */
  lm?: BackendChoice;
  /** Backend for the decoder + vocoder + clone encoder. Default "auto". */
  codec?: BackendChoice;
}

export interface ResolvedBackends {
  lm: Backend;
  codec: Backend;
}

type Stage = "lm" | "codec";

async function resolveStage(choice: BackendChoice, stage: Stage): Promise<Backend> {
  if (choice === "wasm") return "wasm";
  if (choice === "webgpu") return (await isWebGpuAvailable()) ? "webgpu" : "wasm";
  // "auto": the LM is sequential (one token at a time) and is fastest on
  // threaded WASM — WebGPU's per-dispatch overhead dominates those tiny steps.
  // The codec (decoder + vocoder) is large and parallel, so WebGPU wins. Each
  // stage degrades gracefully to whatever the environment offers.
  const webgpu = await isWebGpuAvailable();
  if (stage === "lm" && isThreadedWasmAvailable()) return "wasm";
  return webgpu ? "webgpu" : "wasm";
}

/** Resolve a per-stage config into concrete backends (each defaults to "auto"). */
export async function resolveBackends(config: BackendConfig = {}): Promise<ResolvedBackends> {
  const [lm, codec] = await Promise.all([
    resolveStage(config.lm ?? "auto", "lm"),
    resolveStage(config.codec ?? "auto", "codec"),
  ]);
  return { lm, codec };
}

export interface SessionConfig {
  /**
   * Filename of the external-data sidecar (e.g. "kanade_decoder.onnx.data"),
   * relative to the model URL. Required for models exported with external data
   * (>2 GB tensors, or any model torch/optimum split): ORT-Web cannot fetch it
   * implicitly — it must be mounted via `externalData`. The `path` must match
   * the location string baked into the .onnx (the bare filename here).
   */
  dataFile?: string;
  /**
   * When set, the model (and its sidecar) are fetched cache-first via the Cache
   * API and handed to ORT as bytes, so repeat loads are offline/instant. Off by
   * default to preserve ORT's own URL fetching. `onProgress` reports the model
   * download for a UI progress bar.
   */
  cache?: { cacheName?: string; onProgress?: ProgressFn };
}

/** Loader-level options forwarded to createSession (caching/progress). */
export type LoadOptions = Pick<SessionConfig, "cache">;

export async function createSession(
  modelUrl: string,
  backend: Backend,
  config: SessionConfig = {},
): Promise<ort.InferenceSession> {
  configureOrt();
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: [backend],
    // WebGPU EP: the extended-fusion pass produces a SkipLayerNormalization
    // kernel that rejects our LayerNorm bias ("Beta must be 1D"). "basic" keeps
    // cheap optimizations and skips that fusion; it was also fastest in the
    // Phase 0 gate. WASM has no such issue and runs "all".
    graphOptimizationLevel: backend === "webgpu" ? "basic" : "all",
  };
  const base = modelUrl.slice(0, modelUrl.lastIndexOf("/") + 1);

  if (config.cache) {
    // Cache-first: fetch bytes ourselves and pass them (plus any sidecar) to ORT.
    const model = new Uint8Array(
      await fetchCached(modelUrl, { ...config.cache, onProgress: config.cache.onProgress }),
    );
    if (config.dataFile) {
      const data = new Uint8Array(await fetchCached(`${base}${config.dataFile}`, config.cache));
      options.externalData = [{ data, path: config.dataFile }];
    }
    return ort.InferenceSession.create(model, options);
  }

  if (config.dataFile) {
    options.externalData = [{ data: `${base}${config.dataFile}`, path: config.dataFile }];
  }
  return ort.InferenceSession.create(modelUrl, options);
}

export { ort };
