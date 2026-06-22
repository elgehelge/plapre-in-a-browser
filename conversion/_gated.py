"""Shared helpers for the scripts that need the gated `syvai/plapre-*` repos.

Downloading the LM weights requires (a) a Hugging Face login and (b) accepting
the model's license on the model page. These are account actions a human must
take; the scripts below cannot bypass them. `ensure_access` turns the raw
GatedRepoError / 401 into a single, actionable message so the one human step is
obvious.

This module is also the single source of truth for the **model variants** we
convert (Pico, Nano). They share the Kanade decoder/vocoder/clone-encoder; only
the LM-side artifacts differ, and each comes from its own gated checkpoint. The
variant-specific outputs go to a per-model sub-directory that mirrors the web
layout (`web/src/pipeline/models.ts` `prefix`): each variant under its own id
(`pico/`, `nano/`), with the shared Kanade artifacts at the models root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CHECKPOINTS: dict[str, str] = {
    "pico": "syvai/plapre-pico",
    "nano": "syvai/plapre-nano",
}
MODELS: tuple[str, ...] = tuple(CHECKPOINTS)
DEFAULT_MODEL = "pico"
SPEAKER_DIM = 128

# Back-compat for callers that still import the single default checkpoint.
CHECKPOINT = CHECKPOINTS[DEFAULT_MODEL]

MODELS_ROOT = Path(__file__).resolve().parent.parent / "web" / "public" / "models"
GOLDEN_ROOT = Path(__file__).resolve().parent / "golden"


def checkpoint_for(model: str) -> str:
    """Resolve a variant id to its gated upstream checkpoint."""
    try:
        return CHECKPOINTS[model]
    except KeyError:
        raise SystemExit(f"unknown model {model!r}; choose one of {', '.join(MODELS)}")


def model_subdir(model: str) -> str:
    """Sub-path for a variant's artifacts — its id (`pico`, `nano`)."""
    if model not in CHECKPOINTS:
        raise SystemExit(f"unknown model {model!r}; choose one of {', '.join(MODELS)}")
    return model


def variant_dir(model: str) -> Path:
    """Where this variant's LM-side artifacts (lm/, tokenizer, speakers) live."""
    return MODELS_ROOT / model_subdir(model)


def golden_dir(model: str) -> Path:
    """Where this variant's golden reference fixtures live."""
    return GOLDEN_ROOT / model_subdir(model)


def speakers_json(checkpoint: str) -> str:
    """Path to a speakers.json holding the built-in raw 128-dim embeddings.

    These are Kanade speaker identities, independent of the LM, so a variant repo
    that ships only a speaker_proj (e.g. Nano) reuses the default checkpoint's
    embeddings; only the projection to the LM hidden size differs per variant.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        return hf_hub_download(checkpoint, "speakers.json")
    except EntryNotFoundError:
        fallback = CHECKPOINTS[DEFAULT_MODEL]
        print(f"[speakers] {checkpoint} has no speakers.json; reusing raw embeddings from {fallback}")
        return hf_hub_download(fallback, "speakers.json")


def add_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=MODELS,
        default=DEFAULT_MODEL,
        help=f"which Plapre variant to convert (default: {DEFAULT_MODEL})",
    )


def parse_model() -> str:
    """Standalone scripts: read just --model from argv (ignoring other flags)."""
    parser = argparse.ArgumentParser(add_help=False)
    add_model_arg(parser)
    args, _ = parser.parse_known_args()
    return args.model


def ensure_access(checkpoint: str = CHECKPOINT) -> None:
    """Verify the gated repo can be downloaded; exit with guidance if not."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    try:
        hf_hub_download(checkpoint, "config.json")
    except (GatedRepoError, RepositoryNotFoundError, OSError) as err:
        msg = str(err)
        if "401" in msg or "gated" in msg.lower() or isinstance(err, GatedRepoError):
            sys.exit(
                f"\n[blocked] Cannot download gated repo '{checkpoint}'.\n"
                "This is the one step that needs you (a license/account action):\n"
                f"  1. Open https://huggingface.co/{checkpoint} and click "
                "'Agree and access repository'.\n"
                "  2. Authenticate this machine:  huggingface-cli login\n"
                "  3. Re-run this script.\n"
            )
        raise
