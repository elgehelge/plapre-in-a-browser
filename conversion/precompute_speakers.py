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

from _gated import (
    SPEAKER_DIM,
    checkpoint_for,
    ensure_access,
    parse_model,
    speakers_json,
    variant_dir,
)


def main() -> None:
    model_id = parse_model()
    checkpoint = checkpoint_for(model_id)
    ensure_access(checkpoint)
    out_dir = variant_dir(model_id)
    out = out_dir / "speakers.json"
    proj_out = out_dir / "speaker_proj.json"  # for runtime voice cloning
    out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download

    # config.json holds hidden_size (Pico 576, Nano 960)
    cfg = json.loads(Path(hf_hub_download(checkpoint, "config.json")).read_text())
    hidden = cfg["hidden_size"]

    speakers_path = speakers_json(checkpoint)
    raw = json.loads(Path(speakers_path).read_text())

    # speakers.json may map name -> [floats] or name -> {"embedding"/"emb": [...]}.
    def to_vec(v: object) -> list[float]:
        if isinstance(v, dict):
            for key in ("embedding", "emb", "raw", "vector"):
                if key in v:
                    return list(v[key])  # type: ignore[arg-type]
            raise ValueError(f"unrecognized speaker entry keys: {list(v)}")
        return list(v)  # type: ignore[arg-type]

    # speaker_proj.pt may be an nn.Linear state_dict ({weight,bias}) or a bare
    # weight tensor; handle both.
    proj = nn.Linear(SPEAKER_DIM, hidden)
    sd = torch.load(hf_hub_download(checkpoint, "speaker_proj.pt"), map_location="cpu")
    if isinstance(sd, dict) and any(k.endswith("weight") for k in sd):
        # tolerate prefixed keys (e.g. "speaker_proj.weight")
        clean = {k.split(".")[-1]: v for k, v in sd.items() if k.split(".")[-1] in ("weight", "bias")}
        proj.load_state_dict(clean)
    elif torch.is_tensor(sd):
        with torch.no_grad():
            proj.weight.copy_(sd)
            proj.bias.zero_()
    else:
        proj.load_state_dict(sd)
    proj = proj.float().eval()

    speakers: dict[str, dict[str, list[float]]] = {}
    with torch.no_grad():
        for name, emb in raw.items():
            t = torch.tensor(to_vec(emb), dtype=torch.float32)
            hidden_vec = proj(t)
            speakers[name] = {"raw": t.tolist(), "hidden": hidden_vec.tolist()}

    out.write_text(json.dumps(speakers))
    print(f"Wrote {len(speakers)} speakers ({', '.join(speakers)}) -> {out}")
    print(f"model={model_id} hidden_size={hidden}")

    # Ship speaker_proj (128 -> hidden) for RUNTIME voice cloning: the clone
    # encoder yields a raw 128-dim embedding, and the LM prompt needs the
    # projected hidden vector. It is a single Linear, so we ship the weights as
    # JSON and apply y = W·x + b in JS (web/src/pipeline/clone.ts) — no extra
    # ONNX session. Built-in speakers already have `hidden` precomputed; this is
    # only for voices cloned at runtime.
    proj_out.write_text(
        json.dumps(
            {
                "in": SPEAKER_DIM,
                "out": hidden,
                "weight": proj.weight.detach().tolist(),  # [hidden][128]
                "bias": proj.bias.detach().tolist(),  # [hidden]
            }
        )
    )
    print(f"Wrote speaker_proj ({SPEAKER_DIM}->{hidden}) -> {proj_out}")


if __name__ == "__main__":
    main()
