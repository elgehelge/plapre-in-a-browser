// Phase 1 in-browser gate: drive the real OrtLmGraph + PlapreLM decode loop
// against the TOY LM (conversion/gen_toy_lm.py) under onnxruntime-web, and check
// the greedy token ids match the PyTorch/ORT-CPU golden. This validates the
// KV-cache wiring (present.* -> past_key_values.*) and the inputs_embeds /
// prefix-embed contract for real — independent of the gated Plapre weights.
// Backend via ?backend=wasm|webgpu; result on window.__phase1.

import { OrtLmGraph, PlapreLM, type PromptTokenizer, type GenerateOptions } from "./pipeline/lm.js";
import type { Backend } from "./pipeline/types.js";

function log(msg: string): void {
  const el = document.getElementById("out");
  if (el) el.textContent += msg + "\n";
}

interface ToyGolden {
  prompt: number[];
  eos: number;
  ids: number[];
}

async function run(): Promise<void> {
  const backend = (new URLSearchParams(location.search).get("backend") ?? "wasm") as Backend;
  log(`backend: ${backend}`);
  try {
    const golden = (await (await fetch("/models/phase1_toy_golden.json")).json()) as ToyGolden;

    const t0 = performance.now();
    const graph = await OrtLmGraph.fromUrls(
      "/models/lm_toy/model.onnx",
      "/models/lm_toy/meta.json",
      backend,
    );
    const tLoadMs = performance.now() - t0;

    // Fake tokenizer: replay the golden prompt + eos so the loop is exercised
    // exactly as in the reference greedy decode.
    const tokenizer: PromptTokenizer = {
      buildPrompt: () => golden.prompt,
      special: { eos: golden.eos },
    };
    const lm = PlapreLM.withGraph(graph, tokenizer);

    const opts: GenerateOptions = { temperature: 0, topK: 0, topP: 1, maxTokens: 30, seed: 0 };
    const t1 = performance.now();
    const ids = await lm.generate("", new Float32Array(graph.hidden), opts); // zero speaker prefix
    const tRunMs = performance.now() - t1;

    const ok = ids.length === golden.ids.length && ids.every((v, i) => v === golden.ids[i]);
    const result = {
      backend,
      ok,
      generated: ids.length,
      expected: golden.ids.length,
      mismatchAt: ok ? -1 : ids.findIndex((v, i) => v !== golden.ids[i]),
      tLoadMs: Math.round(tLoadMs),
      tRunMs: Math.round(tRunMs),
    };
    (window as unknown as { __phase1: unknown }).__phase1 = result;
    log(JSON.stringify(result, null, 2));
  } catch (err) {
    (window as unknown as { __phase1: unknown }).__phase1 = { backend, ok: false, error: String(err) };
    log("ERROR: " + String(err));
  }
}

void run();
