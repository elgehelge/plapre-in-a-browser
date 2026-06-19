"""Copy tokenizer.json from the gated repo into web/public/models/.

The JS tokenizer (web/src/pipeline/tokenizer.ts) loads this verbatim. config.json
is also fetched for reference (hidden_size etc.), though meta.json from
export_lm.py is the runtime source of truth for model dims.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from _gated import CHECKPOINT, ensure_access

OUT = Path(__file__).parent.parent / "web" / "public" / "models"


def main() -> None:
    ensure_access()
    from huggingface_hub import hf_hub_download

    OUT.mkdir(parents=True, exist_ok=True)
    for fname in ("tokenizer.json", "config.json"):
        src = hf_hub_download(CHECKPOINT, fname)
        dst = OUT / fname
        shutil.copyfile(src, dst)
        print(f"copied {fname} -> {dst}")


if __name__ == "__main__":
    main()
