"""Copy tokenizer.json from the gated repo into web/public/models/.

The JS tokenizer (web/src/pipeline/tokenizer.ts) loads this verbatim. config.json
is also fetched for reference (hidden_size etc.), though meta.json from
export_lm.py is the runtime source of truth for model dims.
"""

from __future__ import annotations

import shutil

from _gated import checkpoint_for, ensure_access, parse_model, variant_dir


def main() -> None:
    model_id = parse_model()
    checkpoint = checkpoint_for(model_id)
    ensure_access(checkpoint)
    from huggingface_hub import hf_hub_download

    out = variant_dir(model_id)
    out.mkdir(parents=True, exist_ok=True)
    for fname in ("tokenizer.json", "config.json"):
        src = hf_hub_download(checkpoint, fname)
        dst = out / fname
        shutil.copyfile(src, dst)
        print(f"copied {model_id} {fname} -> {dst}")


if __name__ == "__main__":
    main()
