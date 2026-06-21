// Phase 4 validation harness: measure load time, per-iteration latency, and
// real-time factor (RTF = audio seconds produced / wall seconds) for the model
// stages that are present, on a selectable backend. Stages whose artifacts are
// missing are skipped. Covers the LM decode loop (warm), the decoder + vocoder
// chain, and the clone encoder.
//
// Query: ?backend=wasm|webgpu&iters=N&tokens=T. Result on window.__bench.

import { createSession, ort } from "./pipeline/ort.js";
import { hasArtifact, artifactUrl } from "./pipeline/assets.js";
import { SAMPLE_RATE, type Backend } from "./pipeline/types.js";
import { normalizeText } from "./pipeline/normalize.js";
import { PlapreTokenizer } from "./pipeline/tokenizer.js";
import { PlapreLM } from "./pipeline/lm.js";
import { loadSpeakers } from "./pipeline/speakers.js";

// Plapre emits ~25 audio tokens per second of speech, so an LM run that produces
// N audio tokens corresponds to N / 25 seconds of audio.
const AUDIO_TOKENS_PER_SEC = 25;

function log(msg: string): void {
  const el = document.getElementById("out");
  if (el) el.textContent += msg + "\n";
}

interface StageResult {
  stage: string;
  loadMs: number;
  meanMs: number;
  p50Ms: number;
  iters: number;
  rtf?: number; // only where the stage produces audio
  note?: string;
}

function stats(times: number[]): { meanMs: number; p50Ms: number } {
  const sorted = [...times].sort((a, b) => a - b);
  const mean = times.reduce((s, t) => s + t, 0) / times.length;
  return { meanMs: round(mean), p50Ms: round(sorted[Math.floor(sorted.length / 2)]) };
}

const round = (n: number): number => Math.round(n * 100) / 100;

async function time<T>(fn: () => Promise<T>): Promise<[T, number]> {
  const t0 = performance.now();
  const out = await fn();
  return [out, performance.now() - t0];
}

async function benchDecoderVocoder(
  backend: Backend,
  iters: number,
  tokens: number,
): Promise<StageResult[]> {
  const out: StageResult[] = [];
  const [decoder, decLoad] = await time(() =>
    createSession(artifactUrl("kanadeDecoder"), backend, { dataFile: "kanade_decoder.onnx.data" }),
  );
  const [vocoder, vocLoad] = await time(() =>
    createSession(artifactUrl("vocoder"), backend, { dataFile: "hift_vocoder.onnx.data" }),
  );

  const indices = new BigInt64Array(tokens);
  for (let i = 0; i < tokens; i++) indices[i] = BigInt(i % 4096);
  const emb = new ort.Tensor("float32", randn(128), [128]);

  // The decoder emits mel as [n_mels, T]; the vocoder wants [1, n_mels, T].
  const batchMel = (mel: ort.Tensor): ort.Tensor => {
    const [nMels, frames] = mel.dims as number[];
    return new ort.Tensor("float32", mel.data as Float32Array, [1, nMels, frames]);
  };
  const decodeOnce = () =>
    decoder.run({
      content_token_indices: new ort.Tensor("int64", indices, [tokens]),
      global_embedding: emb,
    });

  // Warm up.
  await vocoder.run({ mel: batchMel((await decodeOnce()).mel) });

  const decTimes: number[] = [];
  const vocTimes: number[] = [];
  let audioSeconds = 0;
  for (let n = 0; n < iters; n++) {
    const [{ mel }, dt] = await time(decodeOnce);
    decTimes.push(dt);
    const [res, vt] = await time(() => vocoder.run({ mel: batchMel(mel) }));
    vocTimes.push(vt);
    const wav = (res.wav ?? Object.values(res)[0]).data as Float32Array;
    audioSeconds = wav.length / SAMPLE_RATE;
  }

  const decS = stats(decTimes);
  const vocS = stats(vocTimes);
  out.push({ stage: "decoder", loadMs: round(decLoad), ...decS, iters });
  out.push({ stage: "vocoder", loadMs: round(vocLoad), ...vocS, iters });
  const chainMean = decS.meanMs + vocS.meanMs;
  out.push({
    stage: "decoder+vocoder (chain)",
    loadMs: round(decLoad + vocLoad),
    meanMs: round(chainMean),
    p50Ms: round(decS.p50Ms + vocS.p50Ms),
    iters,
    rtf: round(audioSeconds / (chainMean / 1000)),
    note: `${tokens} tokens -> ${round(audioSeconds)}s audio`,
  });
  return out;
}

// Warm RTF for the autoregressive LM — the pipeline's real bottleneck. Drives the
// production decode loop (KV cache + sampling) for one sentence after a warmup
// pass, so shader compilation / cache allocation is excluded.
async function benchLm(
  backend: Backend,
  iters: number,
  maxTokens: number,
): Promise<StageResult[]> {
  const tokenizer = await PlapreTokenizer.load();
  const [lm, loadMs] = await time(() => PlapreLM.load(tokenizer, backend));
  const speakers = await loadSpeakers();
  const speaker = Object.values(speakers)[0];
  if (!speaker) throw new Error("no speakers available to drive the LM benchmark");
  const hidden = Float32Array.from(speaker.hidden);
  const text = normalizeText("Hej, hvordan har du det i dag?");
  const gen = { temperature: 0.8, topK: 50, topP: 0.95, maxTokens, seed: 0 };

  await lm.generate(text, hidden, gen); // warm: compile shaders / allocate KV cache

  const times: number[] = [];
  let producedTokens = 0;
  for (let n = 0; n < iters; n++) {
    const [ids, t] = await time(() => lm.generate(text, hidden, gen));
    times.push(t);
    producedTokens = ids.length; // deterministic across runs (seed fixed)
  }
  const s = stats(times);
  const audioSeconds = producedTokens / AUDIO_TOKENS_PER_SEC;
  return [
    {
      stage: "lm decode (warm)",
      loadMs: round(loadMs),
      ...s,
      iters,
      rtf: round(audioSeconds / (s.meanMs / 1000)),
      note: `${producedTokens} audio tokens -> ${round(audioSeconds)}s audio @ ${AUDIO_TOKENS_PER_SEC} tok/s`,
    },
  ];
}

async function benchClone(backend: Backend, iters: number): Promise<StageResult[]> {
  const [session, loadMs] = await time(() => createSession(artifactUrl("cloneEncoder"), backend));
  const samples = 2 * SAMPLE_RATE; // 2 s reference
  const wav = new ort.Tensor("float32", randn(samples), [1, samples]);
  await session.run({ waveform: wav }); // warm
  const times: number[] = [];
  for (let n = 0; n < iters; n++) {
    const [, t] = await time(() => session.run({ waveform: wav }));
    times.push(t);
  }
  return [{ stage: "clone_encoder (2s ref)", loadMs: round(loadMs), ...stats(times), iters }];
}

function randn(n: number): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (Math.random() * 2 - 1) * 0.1;
  return a;
}

async function run(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const backend = (params.get("backend") ?? "wasm") as Backend;
  const iters = Number(params.get("iters") ?? 5);
  const tokens = Number(params.get("tokens") ?? 50);
  log(`backend=${backend} iters=${iters} tokens=${tokens}`);

  const results: StageResult[] = [];
  try {
    if (
      (await hasArtifact("lm")) &&
      (await hasArtifact("lmMeta")) &&
      (await hasArtifact("tokenizer")) &&
      (await hasArtifact("speakers"))
    ) {
      results.push(...(await benchLm(backend, iters, tokens)));
    } else {
      log("skip lm (artifacts missing)");
    }
    if ((await hasArtifact("kanadeDecoder")) && (await hasArtifact("vocoder"))) {
      results.push(...(await benchDecoderVocoder(backend, iters, tokens)));
    } else {
      log("skip decoder/vocoder (artifacts missing)");
    }
    if (await hasArtifact("cloneEncoder")) {
      results.push(...(await benchClone(backend, iters)));
    } else {
      log("skip clone encoder (artifact missing)");
    }

    const mem = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
    const summary = {
      backend,
      results,
      usedJSHeapMB: mem ? round(mem.usedJSHeapSize / 1e6) : null,
    };
    (window as unknown as { __bench: unknown }).__bench = summary;
    log(JSON.stringify(summary, null, 2));
  } catch (err) {
    (window as unknown as { __bench: unknown }).__bench = { backend, error: String(err) };
    log("ERROR: " + String(err));
  }
}

void run();
