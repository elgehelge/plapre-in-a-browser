"""Generate the Phase 0 browser-parity golden fixture.

Runs the SAME wrappers we export (deterministic decoder + patched vocoder) on
fixed inputs and dumps mel + wav so the in-browser smoke test
(web/phase0.html) can check onnxruntime-web reproduces them.

Output: web/public/models/phase0_golden.json (gitignored, regenerable).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from export_hift_vocoder import VocoderWrapper
from export_kanade_decoder import REPO, DecoderWrapper, _disable_attention_dropout
import hift_onnx

OUT = Path(__file__).parent.parent / "web" / "public" / "models" / "phase0_golden.json"
SEQ_LEN = 50
SEED = 1234


def main() -> None:
    from kanade_tokenizer import KanadeModel, load_vocoder

    torch.manual_seed(SEED)
    kanade = KanadeModel.from_pretrained(REPO).eval()
    _disable_attention_dropout(kanade)
    decoder = DecoderWrapper(kanade).eval()
    vocoder = VocoderWrapper(hift_onnx.patch_vocoder_for_onnx(load_vocoder("hift").eval())).eval()

    # Fixed, reproducible inputs (no RNG): the harness rebuilds these exactly.
    tokens = (torch.arange(SEQ_LEN, dtype=torch.long) * 7 + 3) % 100
    emb = torch.sin(torch.arange(128, dtype=torch.float32) * 0.1)

    with torch.no_grad():
        mel = decoder(tokens, emb)  # (n_mels, T)
        wav = vocoder(mel.unsqueeze(0))  # (1, samples)

    payload = {
        "tokens": tokens.tolist(),
        "emb": emb.tolist(),
        "melDims": list(mel.shape),
        "mel": mel.flatten().tolist(),
        "wavLen": int(wav.numel()),
        "wav": wav.flatten().tolist(),
    }
    OUT.write_text(json.dumps(payload))
    print(f"wrote {OUT} (mel {list(mel.shape)}, wav {wav.numel()} samples)")


if __name__ == "__main__":
    main()
