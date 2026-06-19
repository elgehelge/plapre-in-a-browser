"""Phase 1: precompute speaker artifacts for the browser.

For each built-in speaker we emit two things:

  raw_embedding    the 128-dim Kanade global embedding (used by the DECODER as
                   `global_embedding`).
  hidden           speaker_proj(raw_embedding) -> (hidden,) used as the FIRST
                   input embedding to the LM. Precomputing this removes the
                   nn.Linear(128 -> hidden) projection from the runtime entirely.

Output: web/public/models/speakers.json
    { "<name>": { "raw": [...128 floats...], "hidden": [...hidden floats...] } }

Traced from plapre/inference.py: _load_speakers + _load_speaker_proj +
_project_speaker.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent.parent / "web" / "public" / "models" / "speakers.json"
CHECKPOINT = "syvai/plapre-pico"
SPEAKER_DIM = 128


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download

    # config.json holds hidden_size (Pico 576, Nano 960)
    cfg = json.loads(Path(hf_hub_download(CHECKPOINT, "config.json")).read_text())
    hidden = cfg["hidden_size"]

    speakers_path = hf_hub_download(CHECKPOINT, "speakers.json")
    raw = json.loads(Path(speakers_path).read_text())

    proj = nn.Linear(SPEAKER_DIM, hidden)
    proj.load_state_dict(torch.load(hf_hub_download(CHECKPOINT, "speaker_proj.pt"), map_location="cpu"))
    proj = proj.float().eval()

    out: dict[str, dict[str, list[float]]] = {}
    with torch.no_grad():
        for name, emb in raw.items():
            t = torch.tensor(emb, dtype=torch.float32)
            hidden_vec = proj(t)
            out[name] = {"raw": t.tolist(), "hidden": hidden_vec.tolist()}

    OUT.write_text(json.dumps(out))
    print(f"Wrote {len(out)} speakers ({', '.join(out)}) -> {OUT}")
    print(f"hidden_size={hidden}")


if __name__ == "__main__":
    main()
