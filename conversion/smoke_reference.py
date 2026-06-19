"""Phase 0 golden fixtures.

Run the *reference* Plapre pipeline once for a fixed sentence + built-in speaker
and dump intermediate tensors so the browser port can be checked for parity:

  golden/tokens.json   audio-token ids emitted by the LM (greedy / fixed seed)
  golden/kanade.json   kanade content-token indices (tokens - <audio_0>)
  golden/mel.npy       decoder mel-spectrogram
  golden/reference.wav final 24 kHz waveform

We pin sampling to be deterministic (temperature small / fixed seed) so the JS
loop can reproduce the same ids.

Traced from plapre/inference.py: Plapre.speak -> _generate_audio ->
_generate_tokens + _tokens_to_audio.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).parent / "golden"
SENTENCE = "Hej, hvordan har du det i dag?"
SPEAKER = "tor"
SEED = 0


def main() -> None:
    GOLDEN.mkdir(exist_ok=True)

    import torch

    torch.manual_seed(SEED)

    from plapre import Plapre

    # NOTE: upstream defaults to vLLM. For deterministic, dependency-light golden
    # generation we want a plain torch forward loop instead of vLLM sampling.
    # TODO: either (a) run Plapre with use_async=False and low temperature and
    # accept its vLLM path, or (b) reimplement a tiny greedy torch loop over the
    # HF model so the JS port can match ids exactly. (b) is preferred for parity.
    tts = Plapre(f"syvai/plapre-pico", device="cpu")

    # speak() returns float32 24 kHz audio and also writes the wav.
    audio = tts.speak(
        SENTENCE,
        output=str(GOLDEN / "reference.wav"),
        speaker=SPEAKER,
        temperature=0.0,  # TODO: confirm 0.0 == greedy in this stack
        max_tokens=500,
    )
    np.save(GOLDEN / "audio.npy", np.asarray(audio, dtype=np.float32))

    # TODO: capture intermediate token ids + mel. The public speak() hides them,
    # so call the internals directly:
    #   spk = tts._resolve_speaker(SPEAKER, None, None)
    #   tokens = tts._generate_tokens([tts._normalize_text(SENTENCE)], spk, 0.0, 1.0, 0, 500)[0]
    #   kanade = [t - tts.audio_token_start for t in tokens if tts.audio_token_start <= t <= tts.audio_token_end]
    #   mel = tts.kanade.decode(content_token_indices=torch.tensor(kanade), global_embedding=spk.float())
    #   ... dump tokens.json, kanade.json, mel.npy

    print(f"Wrote golden fixtures to {GOLDEN}")
    print(f"  reference.wav  ({len(audio) / 24000:.2f}s)")
    print("  TODO: tokens.json / kanade.json / mel.npy (see comments)")


if __name__ == "__main__":
    main()
