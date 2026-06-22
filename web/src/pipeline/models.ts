// The Plapre model variants this library can run. They share the Kanade
// decoder/vocoder/clone-encoder; only the LM-side artifacts differ (the LM graph
// + meta, tokenizer, and speaker tables), and those depend on the upstream
// gated checkpoint each variant is converted from.
//
// The runtime never hard-codes a hidden size — it reads `hidden` from
// `lm/meta.json` — so adding a variant is purely a matter of locating its
// artifacts. `prefix` is the sub-path the variant-specific files are served from
// under `modelsBaseUrl`; each variant lives in its own directory (`pico/`,
// `nano/`). The shared Kanade decoder/vocoder/clone-encoder sit at the root.

export type PlapreModelId = "pico" | "nano";

export interface PlapreModelInfo {
  readonly id: PlapreModelId;
  readonly displayName: string;
  /** LM hidden size. Informational only — the runtime reads it from lm/meta.json. */
  readonly hidden: number;
  /** Upstream (gated) checkpoint the artifacts are converted from. */
  readonly checkpoint: string;
  /** Sub-path (under modelsBaseUrl) for this variant's LM-side artifacts. */
  readonly prefix: string;
}

export const PLAPRE_MODELS: Record<PlapreModelId, PlapreModelInfo> = {
  pico: {
    id: "pico",
    displayName: "Pico",
    hidden: 576,
    checkpoint: "syvai/plapre-pico",
    prefix: "pico/",
  },
  nano: {
    id: "nano",
    displayName: "Nano",
    hidden: 960,
    checkpoint: "syvai/plapre-nano",
    prefix: "nano/",
  },
} as const;

export const DEFAULT_MODEL: PlapreModelId = "pico";

export function modelInfo(id: PlapreModelId): PlapreModelInfo {
  return PLAPRE_MODELS[id];
}
