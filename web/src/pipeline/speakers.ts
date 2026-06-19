import { artifactUrl, ARTIFACTS } from "./assets.js";
import { MissingModelError, type SpeakerTable } from "./types.js";

export async function loadSpeakers(): Promise<SpeakerTable> {
  const res = await fetch(artifactUrl("speakers"));
  if (!res.ok) {
    throw new MissingModelError("speakers.json", ARTIFACTS.speakers.producedBy);
  }
  return (await res.json()) as SpeakerTable;
}
