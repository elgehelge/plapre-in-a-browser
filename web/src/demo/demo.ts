// Interactive demo for plapre-in-a-browser. Wires the public engine API to a
// small UI: detect environment + artifacts, load the engine (with download
// progress), synthesize/stream Danish speech, download WAV/MP3, and clone a
// voice locally. It deliberately uses only the published surface (../index.js)
// plus a couple of internal helpers for the env probe and normalization preview.

import {
  loadPlapreEngine,
  reportArtifacts,
  setModelsBaseUrl,
  setModel,
  ARTIFACTS,
  artifactUrl,
  encodeAudio,
  decodeWithWebAudio,
  resolveBackends,
  isWebGpuAvailable,
  type Engine,
  type Voice,
  type ArtifactKey,
  type BackendChoice,
  type PlapreModelId,
  type ResolvedBackends,
} from "../index.js";
import { normalizeText } from "../pipeline/normalize.js";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const els = {
  backendBadge: $("backend-badge"),
  isolatedBadge: $("isolated-badge"),
  backendLm: $<HTMLSelectElement>("backend-lm"),
  backendCodec: $<HTMLSelectElement>("backend-codec"),
  model: $<HTMLSelectElement>("model"),
  modelsBase: $<HTMLInputElement>("models-base"),
  artifacts: $("artifacts"),
  cloneArtifacts: $("clone-artifacts"),
  load: $<HTMLButtonElement>("load"),
  progress: $("progress"),
  progressBar: $("progress-bar"),
  progressLabel: $("progress-label"),
  synth: $("synth"),
  voice: $<HTMLSelectElement>("voice"),
  rate: $<HTMLInputElement>("rate"),
  rateOut: $<HTMLOutputElement>("rate-out"),
  temp: $<HTMLInputElement>("temp"),
  tempOut: $<HTMLOutputElement>("temp-out"),
  text: $<HTMLTextAreaElement>("text"),
  examples: $("examples"),
  norm: $("norm"),
  speak: $<HTMLButtonElement>("speak"),
  stop: $<HTMLButtonElement>("stop"),
  stopwatch: $("stopwatch"),
  dlWav: $<HTMLButtonElement>("dl-wav"),
  dlMp3: $<HTMLButtonElement>("dl-mp3"),
  player: $<HTMLAudioElement>("player"),
  meta: $("meta"),
  cloneFile: $<HTMLInputElement>("clone-file"),
  cloneName: $<HTMLInputElement>("clone-name"),
  cloneBtn: $<HTMLButtonElement>("clone-btn"),
  log: $("log"),
};

// Artifacts required for end-to-end synthesis (everything except the optional
// clone encoder).
const REQUIRED: readonly ArtifactKey[] = [
  "lm",
  "lmMeta",
  "kanadeDecoder",
  "vocoder",
  "tokenizer",
  "speakers",
  "speakerProj",
];

// Hosted ONNX artifacts (CORS-enabled, per-file). The deployed demo defaults to
// these; local `npm run dev` defaults to /models (served from public/).
const HF_MODELS_BASE =
  "https://huggingface.co/elgehelge/plapre-onnx-web/resolve/main";

function defaultModelsBase(): string {
  const host = location.hostname;
  const isLocal = host === "localhost" || host === "127.0.0.1" || host === "";
  return isLocal ? "/models" : HF_MODELS_BASE;
}

const EXAMPLES = [
  "Hej, hvordan har du det i dag?",
  "Klokken er kvart over ni, og det er 12 grader udenfor.",
  "Velkommen! Det her kører lokalt — uden server og uden API-nøgle. Jeg taler " +
    "dansk direkte i din browser, og alt bliver behandlet privat på din egen maskine.",
];

let engine: Engine | null = null;
let lastPcm: { samples: Float32Array; sampleRate: number } | null = null;
let controller: AbortController | null = null;
let resolvedBackends: ResolvedBackends | null = null;
// Artifact checklist rows, so dots can flip grey -> green once actually loaded.
let artifactRows: Partial<Record<ArtifactKey, HTMLElement>> = {};
// Total bytes that will stream during load (the ONNX models + their sidecars),
// and the per-file tally used to show a single aggregate progress bar — these
// files download concurrently, so progress is summed across them by URL.
let plannedBytes = 0;
const downloadedBytes = new Map<string, number>();
// True while an engine load is streaming, so an async size estimate (triggered
// by a model/base change) can't reset the live progress bar mid-download.
let loading = false;

function log(msg: string): void {
  els.log.textContent += `${msg}\n`;
  els.log.scrollTop = els.log.scrollHeight;
}

const fmtMB = (bytes: number): string => `${(bytes / 1e6).toFixed(0)} MB`;

// The files that actually stream download progress: each ONNX model plus its
// external-data sidecar (the bulk of the bytes). The tokenizer/speaker JSON are
// tiny and fetched without progress, so they're left out — keeping the summed
// progress equal to this planned total at completion.
async function plannedDownloadUrls(): Promise<string[]> {
  const urls = [artifactUrl("lm")];
  try {
    const meta = await (await fetch(artifactUrl("lmMeta"))).json();
    if (meta.externalData) {
      const lm = artifactUrl("lm");
      urls.push(lm.slice(0, lm.lastIndexOf("/") + 1) + meta.externalData);
    }
  } catch {
    // meta is optional for sizing; the LM graph itself is still counted.
  }
  for (const key of ["kanadeDecoder", "vocoder"] as const) {
    urls.push(artifactUrl(key), `${artifactUrl(key)}.data`);
  }
  return urls;
}

async function headSize(url: string): Promise<number> {
  try {
    const res = await fetch(url, { method: "HEAD" });
    return res.ok ? Number(res.headers.get("content-length") ?? 0) : 0;
  } catch {
    return 0;
  }
}

// Sum the download sizes up front so the bar can show "0 / N MB" before loading.
async function estimateDownload(): Promise<void> {
  const sizes = await Promise.all((await plannedDownloadUrls()).map(headSize));
  plannedBytes = sizes.reduce((a, b) => a + b, 0);
  // A load may have started while the HEAD requests were in flight; never reset
  // the live bar or the byte tally out from under it.
  if (loading) return;
  downloadedBytes.clear();
  els.progress.hidden = false;
  els.progressBar.style.width = "0%";
  els.progressLabel.textContent = plannedBytes ? `0 / ${fmtMB(plannedBytes)}` : "";
}

function renderProgress(): void {
  const done = [...downloadedBytes.values()].reduce((a, b) => a + b, 0);
  const denom = plannedBytes || done || 1;
  els.progressBar.style.width = `${Math.min(100, Math.round((done / denom) * 100))}%`;
  els.progressLabel.textContent = `${fmtMB(done)} / ${fmtMB(denom)}`;
}

// A live stopwatch shown beside the Synthesize button: it ticks from the click
// until the audio is ready for playback (or the run stops/errors), then freezes
// on the final elapsed time.
let stopwatchStart = 0;
let stopwatchRaf = 0;

const renderStopwatch = (seconds: number): void => {
  els.stopwatch.textContent = `${seconds.toFixed(2)}s`;
};

function startStopwatch(): void {
  cancelAnimationFrame(stopwatchRaf);
  stopwatchStart = performance.now();
  els.stopwatch.hidden = false;
  els.stopwatch.classList.remove("done");
  els.stopwatch.classList.add("running");
  const tick = (): void => {
    renderStopwatch((performance.now() - stopwatchStart) / 1000);
    stopwatchRaf = requestAnimationFrame(tick);
  };
  tick();
}

function stopStopwatch(): void {
  cancelAnimationFrame(stopwatchRaf);
  renderStopwatch((performance.now() - stopwatchStart) / 1000);
  els.stopwatch.classList.remove("running");
  els.stopwatch.classList.add("done");
}

function setBadge(el: HTMLElement, label: string, state: "ok" | "no" | "") {
  // Update only the text span so the trailing ⓘ tooltip hint survives.
  const text = el.querySelector<HTMLElement>(".badge__text") ?? el;
  text.textContent = label;
  el.classList.remove("ok", "no");
  if (state) el.classList.add(state);
}

async function checkEnvironment(): Promise<void> {
  const webgpu = await isWebGpuAvailable();
  setBadge(els.backendBadge, `WebGPU: ${webgpu ? "yes" : "no"}`, webgpu ? "ok" : "no");
  setBadge(
    els.isolatedBadge,
    `multi-threading: ${crossOriginIsolated ? "yes" : "no"}`,
    crossOriginIsolated ? "ok" : "no",
  );
  await refreshArtifacts();
}

function selectedModel(): PlapreModelId {
  return els.model.value as PlapreModelId;
}

async function refreshArtifacts(): Promise<boolean> {
  setModelsBaseUrl(els.modelsBase.value.trim() || "/models");
  setModel(selectedModel());
  els.artifacts.textContent = "Checking artifacts…";
  let present: Record<ArtifactKey, boolean>;
  try {
    present = await reportArtifacts();
  } catch {
    els.artifacts.textContent = "Could not reach the models base URL.";
    return false;
  }

  els.artifacts.innerHTML = "";
  els.cloneArtifacts.innerHTML = "";
  artifactRows = {};
  for (const key of Object.keys(ARTIFACTS) as ArtifactKey[]) {
    const ok = present[key];
    const row = document.createElement("div");
    // Grey by default (present but not loaded), red if missing; green once it's
    // actually loaded — required files on engine load, the clone encoder lazily
    // on the first cloneVoice().
    row.className = `art${ok ? "" : " missing"}`;
    row.innerHTML = `<span class="dot"></span><code>${ARTIFACTS[key].file}</code>`;
    // The clone encoder lives next to the file chooser it powers, not in the
    // engine-load checklist.
    (key === "cloneEncoder" ? els.cloneArtifacts : els.artifacts).appendChild(row);
    artifactRows[key] = row;
  }

  const ready = REQUIRED.every((k) => present[k]);
  els.load.disabled = !ready;
  els.load.textContent = ready ? "Load engine" : "Load engine (artifacts missing)";
  if (ready) {
    // Show the download size up front, so the bar starts at "0 / N MB".
    void estimateDownload();
  } else {
    els.progress.hidden = true;
    log(
      "Some required artifacts are missing. Produce them with the conversion " +
        "toolchain (see conversion/) or set the models base URL to a bundle.",
    );
  }
  return ready;
}

async function loadEngine(): Promise<void> {
  loading = true;
  els.load.disabled = true;
  // Freeze the variant/source while loading so a change event can't recompute
  // the planned total or kick off a competing estimate mid-download.
  els.model.disabled = true;
  els.modelsBase.disabled = true;
  downloadedBytes.clear();
  els.progress.hidden = false;
  els.progressBar.style.width = "0%";
  els.progressLabel.textContent = plannedBytes ? `0 / ${fmtMB(plannedBytes)}` : "Loading…";
  const backend_lm = els.backendLm.value as BackendChoice;
  const backend_codec = els.backendCodec.value as BackendChoice;
  resolvedBackends = await resolveBackends({ lm: backend_lm, codec: backend_codec });
  try {
    engine = await loadPlapreEngine({
      backend_lm,
      backend_codec,
      model: selectedModel(),
      modelsBaseUrl: els.modelsBase.value.trim() || "/models",
      cache: {
        onProgress: (loaded, _total, url) => {
          // Files download concurrently; track each by URL and show the sum.
          if (url) downloadedBytes.set(url, loaded);
          renderProgress();
        },
      },
    });
    for (const key of REQUIRED) artifactRows[key]?.classList.add("loaded");
    els.progressBar.style.width = "100%";
    if (plannedBytes) els.progressLabel.textContent = `${fmtMB(plannedBytes)} — loaded`;
    populateVoices(engine.listVoices());
    els.synth.setAttribute("aria-disabled", "false");
    els.speak.disabled = false;
    els.cloneBtn.disabled = !engine.canCloneVoice();
    log(
      `Engine loaded — ${selectedModel()} on LM ${resolvedBackends.lm}, ` +
        `decoder+vocoder on ${resolvedBackends.codec}. ` +
        `${engine.listVoices().length} voice(s) ready.`,
    );
  } catch (err) {
    log(`Load failed: ${err instanceof Error ? err.message : String(err)}`);
    els.load.disabled = false;
    els.progress.hidden = true;
  } finally {
    loading = false;
    els.model.disabled = false;
    els.modelsBase.disabled = false;
  }
}

const capitalize = (s: string): string =>
  s.replace(/\b\p{L}/gu, (c) => c.toUpperCase());

function populateVoices(voices: readonly Voice[]): void {
  const current = els.voice.value;
  els.voice.innerHTML = "";
  for (const v of voices) {
    const opt = document.createElement("option");
    opt.value = v.id;
    opt.textContent = capitalize(v.displayName);
    els.voice.appendChild(opt);
  }
  if (current && voices.some((v) => v.id === current)) els.voice.value = current;
}

async function synthesize(): Promise<void> {
  if (!engine) return;
  els.speak.disabled = true;
  els.stop.disabled = false;
  els.dlWav.disabled = els.dlMp3.disabled = true;
  els.norm.innerHTML = `<b>Normalized:</b> ${normalizeText(els.text.value)}`;
  els.norm.hidden = false;
  els.meta.textContent = "Synthesizing…";
  startStopwatch();
  controller = new AbortController();
  // Fresh seed each click so repeated runs with the same settings vary (unless
  // temperature is 0, which is greedy/deterministic regardless of seed).
  const seed = (Math.random() * 0x100000000) >>> 0;
  const started = performance.now();
  try {
    const pcm = await engine.synthesizeToPcm({
      text: els.text.value,
      voice: els.voice.value,
      rate: Number(els.rate.value),
      // Temperature is a runtime sampling knob (not baked into the artifacts);
      // 0 makes generation greedy/deterministic. See docs/TUNING.md.
      generation: { temperature: Number(els.temp.value), seed },
      signal: controller.signal,
    });
    lastPcm = pcm;
    stopStopwatch();
    const wall = (performance.now() - started) / 1000;
    const dur = pcm.samples.length / pcm.sampleRate;
    playPcm(pcm);
    const backendLabel = resolvedBackends
      ? `lm ${resolvedBackends.lm}, codec ${resolvedBackends.codec}`
      : "";
    els.meta.textContent = `${dur.toFixed(2)}s of audio in ${wall.toFixed(2)}s (${(
      dur / wall
    ).toFixed(1)}× realtime, ${backendLabel}).`;
    els.dlWav.disabled = els.dlMp3.disabled = false;
  } catch (err) {
    stopStopwatch();
    if (err instanceof DOMException && err.name === "AbortError") {
      els.meta.textContent = "Stopped.";
    } else {
      els.meta.textContent = `Error: ${err instanceof Error ? err.message : String(err)}`;
      log(els.meta.textContent);
    }
  } finally {
    els.speak.disabled = false;
    els.stop.disabled = true;
    controller = null;
  }
}

function playPcm(pcm: { samples: Float32Array; sampleRate: number }): void {
  const wav = encodeAudio(pcm.samples, pcm.sampleRate, "wav");
  const blob = new Blob([wav], { type: "audio/wav" });
  if (els.player.src) URL.revokeObjectURL(els.player.src);
  els.player.src = URL.createObjectURL(blob);
  void els.player.play().catch(() => {});
}

function download(format: "wav" | "mp3"): void {
  if (!lastPcm) return;
  const bytes = encodeAudio(lastPcm.samples, lastPcm.sampleRate, format);
  const mime = format === "wav" ? "audio/wav" : "audio/mpeg";
  const blob = new Blob([bytes], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `plapre-${els.voice.value}.${format}`;
  a.click();
  URL.revokeObjectURL(a.href);
}

async function cloneVoice(): Promise<void> {
  if (!engine || !engine.canCloneVoice()) return;
  const file = els.cloneFile.files?.[0];
  if (!file) {
    log("Pick a reference audio file first.");
    return;
  }
  els.cloneBtn.disabled = true;
  els.meta.textContent = "Cloning voice…";
  try {
    const decoded = await decodeWithWebAudio(await file.arrayBuffer());
    const name = els.cloneName.value.trim() || file.name.replace(/\.[^.]+$/, "");
    const voice = await engine.cloneVoice(decoded.pcm, decoded.sampleRate, { displayName: name });
    artifactRows.cloneEncoder?.classList.add("loaded");
    populateVoices(engine.listVoices());
    els.voice.value = voice.id;
    els.meta.textContent = `Cloned "${voice.displayName}" — selected as the active voice.`;
    log(`Cloned voice id=${voice.id} from ${file.name}`);
  } catch (err) {
    els.meta.textContent = `Clone failed: ${err instanceof Error ? err.message : String(err)}`;
    log(els.meta.textContent);
  } finally {
    els.cloneBtn.disabled = false;
  }
}

function buildExamples(): void {
  for (const ex of EXAMPLES) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.textContent = ex.length > 38 ? `${ex.slice(0, 36)}…` : ex;
    chip.title = ex;
    chip.addEventListener("click", () => {
      els.text.value = ex;
    });
    els.examples.appendChild(chip);
  }
}

els.rate.addEventListener("input", () => {
  els.rateOut.textContent = `${Number(els.rate.value).toFixed(2)}×`;
});
els.temp.addEventListener("input", () => {
  els.tempOut.textContent = Number(els.temp.value).toFixed(2);
});
els.model.addEventListener("change", () => void refreshArtifacts());
els.modelsBase.addEventListener("change", () => void refreshArtifacts());
els.load.addEventListener("click", () => void loadEngine());
els.speak.addEventListener("click", () => void synthesize());
els.stop.addEventListener("click", () => controller?.abort());
els.dlWav.addEventListener("click", () => download("wav"));
els.dlMp3.addEventListener("click", () => download("mp3"));
els.cloneBtn.addEventListener("click", () => void cloneVoice());

els.modelsBase.value = defaultModelsBase();
buildExamples();
void checkEnvironment();
