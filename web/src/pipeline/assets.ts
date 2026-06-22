// Locates the converted model artifacts and reports which are present. Lets the
// app run today and tell the user exactly what conversion step is still needed.
//
// The base URL the artifacts are served from is configurable so the same code
// works whether they sit next to the app (the default `/models`), on a CDN, or
// behind a GitHub Release. Library consumers set it once via
// `loadPlapreEngine({ modelsBaseUrl })` (or `setModelsBaseUrl`) before loading.
//
// Artifacts are either `shared` (the Kanade decoder/vocoder/clone-encoder, which
// are identical across model variants) or `variant` (the LM-side files, which
// differ per variant). Variant artifacts are served from the active model's
// sub-path (see models.ts); shared ones always sit at the base URL root.

import { DEFAULT_MODEL, PLAPRE_MODELS, type PlapreModelId } from "./models.js";

const DEFAULT_MODELS_BASE = "/models";

let modelsBase = DEFAULT_MODELS_BASE;
let currentModel: PlapreModelId = DEFAULT_MODEL;

/** Where the converted artifacts are fetched from. Trailing slash is trimmed. */
export function setModelsBaseUrl(url: string): void {
  modelsBase = url.replace(/\/+$/, "");
}

export function getModelsBaseUrl(): string {
  return modelsBase;
}

/** Select which model variant subsequent artifact lookups resolve against. */
export function setModel(model: PlapreModelId): void {
  currentModel = model;
}

export function getModel(): PlapreModelId {
  return currentModel;
}

export interface Artifact {
  file: string;
  url: string;
  producedBy: string;
}

/** Whether an artifact is shared across variants or specific to one. */
type ArtifactScope = "shared" | "variant";

interface ArtifactSpec {
  file: string;
  scope: ArtifactScope;
  producedBy: string;
}

export const ARTIFACTS = {
  lm: { file: "lm/model.onnx", scope: "variant", producedBy: "conversion/export_lm.py" },
  lmMeta: { file: "lm/meta.json", scope: "variant", producedBy: "conversion/export_lm.py" },
  kanadeDecoder: {
    file: "kanade_decoder.onnx",
    scope: "shared",
    producedBy: "conversion/export_kanade_decoder.py",
  },
  vocoder: {
    file: "hift_vocoder.onnx",
    scope: "shared",
    producedBy: "conversion/export_hift_vocoder.py",
  },
  tokenizer: {
    file: "tokenizer.json",
    scope: "variant",
    producedBy: "conversion/fetch_tokenizer.py (copies it from the gated checkpoint)",
  },
  speakers: {
    file: "speakers.json",
    scope: "variant",
    producedBy: "conversion/precompute_speakers.py",
  },
  cloneEncoder: {
    file: "clone_encoder.onnx",
    scope: "shared",
    producedBy: "conversion/export_clone_encoder.py",
  },
  speakerProj: {
    file: "speaker_proj.json",
    scope: "variant",
    producedBy: "conversion/precompute_speakers.py",
  },
} as const satisfies Record<string, ArtifactSpec>;

export type ArtifactKey = keyof typeof ARTIFACTS;

export function artifactUrl(key: ArtifactKey): string {
  const spec = ARTIFACTS[key];
  const prefix = spec.scope === "variant" ? PLAPRE_MODELS[currentModel].prefix : "";
  return `${modelsBase}/${prefix}${spec.file}`;
}

/** HEAD-check whether an artifact is present. */
export async function hasArtifact(key: ArtifactKey): Promise<boolean> {
  try {
    const res = await fetch(artifactUrl(key), { method: "HEAD" });
    return res.ok;
  } catch {
    return false;
  }
}

export async function reportArtifacts(): Promise<Record<ArtifactKey, boolean>> {
  const keys = Object.keys(ARTIFACTS) as ArtifactKey[];
  const present = await Promise.all(keys.map(hasArtifact));
  return Object.fromEntries(keys.map((k, i) => [k, present[i]])) as Record<
    ArtifactKey,
    boolean
  >;
}
