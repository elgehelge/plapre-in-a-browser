// Interactive demo for plapre-in-a-browser. Wires the public engine API to a
// small UI: detect environment + artifacts, load the engine (with download
// progress), synthesize/stream Danish speech, download WAV/MP3, and clone a
// voice locally. It deliberately uses only the published surface (../index.js)
// plus a couple of internal helpers for the env probe and normalization preview.

import {
  loadPlapreEngine,
  reportArtifacts,
  setModelsBaseUrl,
  ARTIFACTS,
  encodeAudio,
  decodeWithWebAudio,
  resolveBackends,
  isWebGpuAvailable,
  type Engine,
  type Voice,
  type ArtifactKey,
  type BackendChoice,
  type ResolvedBackends,
} from "../index.js";
import { normalizeText } from "../pipeline/normalize.js";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const els = {
  backendBadge: $("backend-badge"),
  isolatedBadge: $("isolated-badge"),
  backendLm: $<HTMLSelectElement>("backend-lm"),
  backendCodec: $<HTMLSelectElement>("backend-codec"),
  modelsBase: $<HTMLInputElement>("models-base"),
  artifacts: $("artifacts"),
  check: $<HTMLButtonElement>("check"),
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
  "Jeg hedder Ida, og jeg taler dansk i din browser.",
  "Velkommen! Det her kører lokalt — uden server og uden API-nøgle.",
];

let engine: Engine | null = null;
let lastPcm: { samples: Float32Array; sampleRate: number } | null = null;
let controller: AbortController | null = null;
let resolvedBackends: ResolvedBackends | null = null;
// Artifact checklist rows, so dots can flip grey -> green once actually loaded.
let artifactRows: Partial<Record<ArtifactKey, HTMLElement>> = {};

function log(msg: string): void {
  els.log.textContent += `${msg}\n`;
  els.log.scrollTop = els.log.scrollHeight;
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

async function refreshArtifacts(): Promise<boolean> {
  setModelsBaseUrl(els.modelsBase.value.trim() || "/models");
  els.artifacts.textContent = "Checking artifacts…";
  let present: Record<ArtifactKey, boolean>;
  try {
    present = await reportArtifacts();
  } catch {
    els.artifacts.textContent = "Could not reach the models base URL.";
    return false;
  }

  els.artifacts.innerHTML = "";
  artifactRows = {};
  for (const key of Object.keys(ARTIFACTS) as ArtifactKey[]) {
    const ok = present[key];
    const row = document.createElement("div");
    // Grey by default (present but not loaded), red if missing; green once the
    // engine actually loads it (see loadEngine / cloneVoice).
    row.className = `art${ok ? "" : " missing"}`;
    const optional = key === "cloneEncoder" ? " (optional)" : "";
    row.innerHTML = `<span class="dot"></span><code>${ARTIFACTS[key].file}</code>${optional}`;
    els.artifacts.appendChild(row);
    artifactRows[key] = row;
  }

  const ready = REQUIRED.every((k) => present[k]);
  els.load.disabled = !ready;
  els.load.textContent = ready ? "Load engine" : "Load engine (artifacts missing)";
  if (!ready) {
    log(
      "Some required artifacts are missing. Produce them with the conversion " +
        "toolchain (see conversion/) or set the models base URL to a bundle.",
    );
  }
  return ready;
}

function showProgress(on: boolean): void {
  els.progress.hidden = !on;
  if (!on) els.progressBar.style.width = "0%";
}

async function loadEngine(): Promise<void> {
  els.load.disabled = true;
  showProgress(true);
  els.progressLabel.textContent = "Loading…";
  const backend_lm = els.backendLm.value as BackendChoice;
  const backend_codec = els.backendCodec.value as BackendChoice;
  resolvedBackends = await resolveBackends({ lm: backend_lm, codec: backend_codec });
  try {
    engine = await loadPlapreEngine({
      backend_lm,
      backend_codec,
      modelsBaseUrl: els.modelsBase.value.trim() || "/models",
      cache: {
        onProgress: (loaded, total) => {
          const pct = total ? Math.round((loaded / total) * 100) : 0;
          els.progressBar.style.width = `${pct}%`;
          els.progressLabel.textContent = `${(loaded / 1e6).toFixed(0)} MB${
            total ? ` / ${(total / 1e6).toFixed(0)} MB` : ""
          }`;
        },
      },
    });
    for (const key of REQUIRED) artifactRows[key]?.classList.add("loaded");
    populateVoices(engine.listVoices());
    els.synth.setAttribute("aria-disabled", "false");
    els.speak.disabled = false;
    els.cloneBtn.disabled = !engine.canCloneVoice();
    log(
      `Engine loaded — LM on ${resolvedBackends.lm}, decoder+vocoder on ` +
        `${resolvedBackends.codec}. ${engine.listVoices().length} voice(s) ready.`,
    );
  } catch (err) {
    log(`Load failed: ${err instanceof Error ? err.message : String(err)}`);
    els.load.disabled = false;
  } finally {
    showProgress(false);
  }
}

function populateVoices(voices: readonly Voice[]): void {
  const current = els.voice.value;
  els.voice.innerHTML = "";
  for (const v of voices) {
    const opt = document.createElement("option");
    opt.value = v.id;
    opt.textContent = v.displayName;
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
els.check.addEventListener("click", () => void refreshArtifacts());
els.load.addEventListener("click", () => void loadEngine());
els.speak.addEventListener("click", () => void synthesize());
els.stop.addEventListener("click", () => controller?.abort());
els.dlWav.addEventListener("click", () => download("wav"));
els.dlMp3.addEventListener("click", () => download("mp3"));
els.cloneBtn.addEventListener("click", () => void cloneVoice());

els.modelsBase.value = defaultModelsBase();
buildExamples();
void checkEnvironment();
