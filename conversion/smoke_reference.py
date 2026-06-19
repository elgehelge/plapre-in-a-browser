"""Phase 1/2 golden fixtures + Python parity reference.

Runs a self-contained, dependency-light reference of the FULL Plapre chain for a
fixed sentence + built-in speaker and dumps intermediates so the browser port can
be checked for parity:

  golden/tokens.json   audio-token ids from the LM (greedy)
  golden/kanade.json   kanade content-token indices (tokens - <audio_0>)
  golden/mel.npy       decoder mel-spectrogram
  golden/reference.wav final 24 kHz waveform

The token loop here intentionally mirrors web/src/pipeline/lm.ts exactly
(inputs_embeds = [speaker_hidden, embed(prompt)], KV cache, greedy argmax, stop
at EOS / max_tokens) instead of using vLLM, so the JS ids must match bit-for-bit
under greedy decoding. This doubles as the cross-language oracle for Phase 1.

STATUS: UNVALIDATED end-to-end — gated weights (see conversion/_gated.py). The
loop shape is the same one proven by the toy browser gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from _gated import CHECKPOINT, SPEAKER_DIM, ensure_access

GOLDEN = Path(__file__).parent / "golden"
SENTENCE = "Hej, hvordan har du det i dag?"
SPEAKER = "tor"
MAX_TOKENS = 500


def _normalize(text: str) -> str:
    """Mirror plapre _normalize_text (and web/src/pipeline/normalize.ts)."""
    import re

    from num2words import num2words

    text = re.sub(r"\s*---\s*caption.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()

    def repl(m: "re.Match[str]") -> str:
        token = m.group(0).replace(".", "").replace(",", " komma ")
        parts = token.split(" komma ")
        words = [num2words(int(p), lang="da") if p.isdigit() else p for p in parts]
        return " komma ".join(words)

    return re.sub(r"\d[\d.,]*", repl, text)


@torch.no_grad()
def _generate_tokens(model, embed, prompt_ids, speaker_hidden, audio_start, audio_end, eos):
    """Greedy decode mirroring PlapreLM.generate. Returns audio-token ids."""
    from transformers.cache_utils import DynamicCache

    tok = embed(torch.tensor(prompt_ids, dtype=torch.long))  # [seq, H]
    inputs_embeds = torch.cat([speaker_hidden.unsqueeze(0), tok], dim=0).unsqueeze(0)
    cache = DynamicCache()
    pos = 0
    out = model.model(
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        position_ids=torch.arange(pos, pos + inputs_embeds.shape[1]).unsqueeze(0),
        use_cache=True,
    )
    pos += inputs_embeds.shape[1]
    logits = model.lm_head(out.last_hidden_state)
    cache = out.past_key_values

    ids: list[int] = []
    for _ in range(MAX_TOKENS):
        nxt = int(logits[0, -1].argmax())
        if nxt == eos:
            break
        ids.append(nxt)
        step = embed(torch.tensor([nxt], dtype=torch.long)).unsqueeze(0)
        out = model.model(
            inputs_embeds=step,
            past_key_values=cache,
            position_ids=torch.tensor([[pos]]),
            use_cache=True,
        )
        pos += 1
        logits = model.lm_head(out.last_hidden_state)
        cache = out.past_key_values
    return ids


def main() -> None:
    ensure_access()
    GOLDEN.mkdir(exist_ok=True)

    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    embed = model.get_input_embeddings()

    def tid(t: str) -> int:
        return tok.convert_tokens_to_ids(t)

    text_tag, audio_tag = tid("<text>"), tid("<audio>")
    audio_start, audio_end = tid("<audio_0>"), tid("<audio_12799>")
    eos = tok.eos_token_id if tok.eos_token_id is not None else tid("<|endoftext|>")

    # Speaker: raw 128-dim emb + speaker_proj(emb) hidden vector.
    raw = json.loads(Path(hf_hub_download(CHECKPOINT, "speakers.json")).read_text())[SPEAKER]
    raw_vec = torch.tensor(raw if isinstance(raw, list) else raw["embedding"], dtype=torch.float32)
    proj = nn.Linear(SPEAKER_DIM, model.config.hidden_size)
    proj.load_state_dict(torch.load(hf_hub_download(CHECKPOINT, "speaker_proj.pt"), map_location="cpu"))
    speaker_hidden = proj.eval()(raw_vec)

    norm = _normalize(SENTENCE)
    text_ids = tok.encode(norm, add_special_tokens=False)
    prompt_ids = [text_tag, *text_ids, audio_tag]

    tokens = _generate_tokens(
        model, embed, prompt_ids, speaker_hidden, audio_start, audio_end, eos
    )
    kanade = [t - audio_start for t in tokens if audio_start <= t <= audio_end]
    (GOLDEN / "tokens.json").write_text(json.dumps({"sentence": SENTENCE, "speaker": SPEAKER, "ids": tokens}))
    (GOLDEN / "kanade.json").write_text(json.dumps(kanade))
    print(f"LM: {len(tokens)} tokens ({len(kanade)} audio) for {SENTENCE!r}")

    # Stage 3: decoder + vocoder (reference, via the kanade + plapre stacks).
    try:
        from plapre import Plapre

        tts = Plapre(CHECKPOINT, device="cpu")
        mel = tts.kanade.decode(
            content_token_indices=torch.tensor(kanade), global_embedding=raw_vec
        )
        np.save(GOLDEN / "mel.npy", mel.detach().cpu().numpy())
        wav = tts.vocode(mel)
        from scipy.io import wavfile

        wavfile.write(GOLDEN / "reference.wav", 24000, np.asarray(wav, dtype=np.float32))
        print(f"Stage 3: mel{tuple(mel.shape)} -> {len(wav) / 24000:.2f}s wav")
    except Exception as exc:  # noqa: BLE001
        print(f"Stage 3 skipped ({type(exc).__name__}: {exc}); tokens/kanade still written.")


if __name__ == "__main__":
    main()
