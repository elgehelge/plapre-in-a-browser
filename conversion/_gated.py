"""Shared helper for the scripts that need the gated `syvai/plapre-pico` repo.

Downloading the LM weights requires (a) a Hugging Face login and (b) accepting
the model's license on the model page. These are account actions a human must
take; the scripts below cannot bypass them. This helper turns the raw
GatedRepoError / 401 into a single, actionable message so the one human step is
obvious.
"""

from __future__ import annotations

import sys

CHECKPOINT = "syvai/plapre-pico"
SPEAKER_DIM = 128


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
