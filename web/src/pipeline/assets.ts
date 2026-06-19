// Locates the converted model artifacts under /models/ and reports which are
// present. Lets the app run today and tell the user exactly what conversion
// step is still needed.

export const MODELS_BASE = "/models";

export interface Artifact {
  file: string;
  url: string;
  producedBy: string;
}

export const ARTIFACTS = {
  lm: { file: "lm/model.onnx", producedBy: "conversion/export_lm.py" },
  kanadeDecoder: {
    file: "kanade_decoder.onnx",
    producedBy: "conversion/export_kanade_decoder.py",
  },
  vocoder: {
    file: "hift_vocoder.onnx",
    producedBy: "conversion/export_hift_vocoder.py",
  },
  tokenizer: {
    file: "tokenizer.json",
    producedBy: "copy tokenizer.json from the syvai/plapre-pico repo",
  },
  speakers: { file: "speakers.json", producedBy: "conversion/precompute_speakers.py" },
} as const;

export type ArtifactKey = keyof typeof ARTIFACTS;

export function artifactUrl(key: ArtifactKey): string {
  return `${MODELS_BASE}/${ARTIFACTS[key].file}`;
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
