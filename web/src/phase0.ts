// Phase 0 in-browser gate: load the exported decoder + HiFT vocoder under
// onnxruntime-web, run decoder -> vocoder on fixed inputs, and check the output
// matches the PyTorch golden (conversion/gen_phase0_golden.py). Backend is
// chosen via ?backend=wasm|webgpu. Result is mirrored to window.__phase0 for
// Playwright to read.

import * as ort from "onnxruntime-web";

ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;

type Backend = "wasm" | "webgpu";

interface Golden {
  tokens: number[];
  emb: number[];
  melDims: [number, number];
  mel: number[];
  wavLen: number;
  wav: number[];
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Number.POSITIVE_INFINITY;
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

function log(msg: string): void {
  const el = document.getElementById("out");
  if (el) el.textContent += msg + "\n";
}

async function run(): Promise<void> {
  const qs = new URLSearchParams(location.search);
  const backend = (qs.get("backend") ?? "wasm") as Backend;
  // Default mirrors createSession(): "basic" on WebGPU dodges the buggy
  // SkipLayerNormalization fusion; override with ?opt= to probe.
  const opt = (qs.get("opt") ?? (backend === "webgpu" ? "basic" : "all")) as
    ort.InferenceSession.SessionOptions["graphOptimizationLevel"];
  log(`backend: ${backend}, opt: ${opt}`);
  try {
    const golden = (await (await fetch("/models/phase0_golden.json")).json()) as Golden;

    const t0 = performance.now();
    const sessionOpts = (dataFile: string): ort.InferenceSession.SessionOptions => ({
      executionProviders: [backend],
      graphOptimizationLevel: opt,
      externalData: [{ data: `/models/${dataFile}`, path: dataFile }],
    });
    const decoder = await ort.InferenceSession.create(
      "/models/kanade_decoder.onnx",
      sessionOpts("kanade_decoder.onnx.data"),
    );
    const vocoder = await ort.InferenceSession.create(
      "/models/hift_vocoder.onnx",
      sessionOpts("hift_vocoder.onnx.data"),
    );
    const tLoadMs = performance.now() - t0;
    log(`models loaded in ${tLoadMs.toFixed(0)} ms`);

    const tokens = new ort.Tensor(
      "int64",
      BigInt64Array.from(golden.tokens.map((t) => BigInt(t))),
      [golden.tokens.length],
    );
    const emb = new ort.Tensor("float32", Float32Array.from(golden.emb), [golden.emb.length]);

    const t1 = performance.now();
    const decOut = await decoder.run({ content_token_indices: tokens, global_embedding: emb });
    const mel = decOut.mel ?? Object.values(decOut)[0];
    const [nMels, frames] = mel.dims as number[];

    const melBatched = new ort.Tensor("float32", mel.data as Float32Array, [1, nMels, frames]);
    const vocOut = await vocoder.run({ mel: melBatched });
    const wav = vocOut.wav ?? Object.values(vocOut)[0];
    const tRunMs = performance.now() - t1;

    const melDiff = maxAbsDiff(mel.data as Float32Array, golden.mel);
    const wavDiff = maxAbsDiff(wav.data as Float32Array, golden.wav);

    const result = {
      backend,
      ok: melDiff < 0.05 && wavDiff < 0.05,
      melDims: mel.dims,
      wavLen: (wav.data as Float32Array).length,
      melDiff,
      wavDiff,
      tLoadMs: Math.round(tLoadMs),
      tRunMs: Math.round(tRunMs),
    };
    (window as unknown as { __phase0: unknown }).__phase0 = result;
    log(JSON.stringify(result, null, 2));
  } catch (err) {
    const result = { backend, ok: false, error: String(err) };
    (window as unknown as { __phase0: unknown }).__phase0 = result;
    log("ERROR: " + String(err));
  }
}

void run();
