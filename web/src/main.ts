import { normalizeText } from "./pipeline/normalize.js";
import { reportArtifacts, type ArtifactKey, ARTIFACTS } from "./pipeline/assets.js";
import { isWebGpuAvailable } from "./pipeline/ort.js";
import { loadPlapreEngine } from "./pipeline/plapre.js";
import { loadSpeakers } from "./pipeline/speakers.js";
import { pcmToWavBlob } from "./pipeline/wav.js";

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const logEl = $("log");
const textEl = $<HTMLTextAreaElement>("text");
const speakerEl = $<HTMLSelectElement>("speaker");
const player = $<HTMLAudioElement>("player");

function log(msg: string): void {
  logEl.textContent += `${msg}\n`;
}
function clearLog(): void {
  logEl.textContent = "";
}

async function checkEnvironment(): Promise<void> {
  clearLog();
  const webgpu = await isWebGpuAvailable();
  log(`WebGPU available: ${webgpu ? "yes" : "no (will fall back to WASM)"}`);
  log(`crossOriginIsolated: ${crossOriginIsolated} (needed for threaded WASM)`);

  const present = await reportArtifacts();
  log("\nModel artifacts in /models/:");
  for (const key of Object.keys(present) as ArtifactKey[]) {
    const ok = present[key];
    log(`  [${ok ? "x" : " "}] ${ARTIFACTS[key].file}`);
    if (!ok) log(`        produce with: ${ARTIFACTS[key].producedBy}`);
  }

  const ready = Object.values(present).every(Boolean);
  log(`\nPipeline ready to run end-to-end: ${ready ? "yes" : "no"}`);
  if (!ready) log("Pure-JS stages (normalization) still work — try 'Preview normalization'.");

  await populateSpeakers();
}

async function populateSpeakers(): Promise<void> {
  try {
    const speakers = await loadSpeakers();
    speakerEl.innerHTML = "";
    for (const name of Object.keys(speakers)) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      speakerEl.appendChild(opt);
    }
  } catch {
    speakerEl.innerHTML = "<option>(speakers.json not found)</option>";
  }
}

function previewNormalization(): void {
  clearLog();
  log("Input:\n" + textEl.value);
  log("\nNormalized:\n" + normalizeText(textEl.value));
}

async function synthesize(): Promise<void> {
  clearLog();
  log("Loading pipeline…");
  try {
    const engine = await loadPlapreEngine();
    log("Synthesizing…");
    const { samples, sampleRate } = await engine.synthesizeToPcm({
      text: textEl.value,
      voice: speakerEl.value,
    });
    const blob = pcmToWavBlob(samples, sampleRate);
    player.src = URL.createObjectURL(blob);
    log(`Done: ${(samples.length / sampleRate).toFixed(2)}s of audio.`);
  } catch (err) {
    log("\n" + (err instanceof Error ? err.message : String(err)));
    log("\n(Expected until conversion + Phase 1 LM loop are complete.)");
  }
}

$("env").addEventListener("click", () => void checkEnvironment());
$("normalize").addEventListener("click", previewNormalization);
$("synthesize").addEventListener("click", () => void synthesize());

void checkEnvironment();
