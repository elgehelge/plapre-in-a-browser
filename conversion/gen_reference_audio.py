"""Ground-truth reference audio from the ORIGINAL Plapre pipeline (PyTorch).

The official `plapre` package runs the LM through vLLM on a GGUF (q8_0) model;
vLLM has no macOS build, so we reproduce its exact pipeline in full-precision
torch instead (a cleaner reference than the quantized path). The prompt, speaker
projection, EOS, decode and vocode logic are copied 1:1 from
github.com/syv-ai/plapre plapre/inference.py (audited against it).

Two things come out of this:

1. golden/reference_torch.wav — the upstream pipeline's audio for the test
   sentence + speaker, to listen to / compare against our ONNX output.
2. A numerical check of OUR exported decoder+vocoder against the REAL
   kanade.decode()+vocode() on the SAME generated tokens (isolates ONNX-conversion
   fidelity on real audio tokens, not the synthetic ones gen_phase0_golden used).

Run in the conversion venv (needs torch/transformers/kanade-tokenizer/soundfile —
already in requirements.txt; NO vllm). Requires HF auth for the gated weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from _gated import CHECKPOINT, SPEAKER_DIM
from smoke_reference import SENTENCE, SPEAKER

MODELS = Path(__file__).parent.parent / "web" / "public" / "models"
GOLDEN = Path(__file__).parent / "golden"
KANADE_REPO = "frothywater/kanade-25hz-clean"
SR = 24000
# Overridable via --seed N / --temp F; defaults match plapre.inference.speak().
SEED = 0
TEMPERATURE = 0.8


def _normalize_text(text: str) -> str:
    """Copy of plapre.inference.Plapre._normalize_text (numbers via num2words)."""
    import re

    from num2words import num2words

    def _num(m: "re.Match[str]") -> str:
        raw = m.group()
        try:
            return num2words(float(raw.replace(",", ".")), lang="da")
        except (ValueError, OverflowError):
            return raw

    text = re.sub(r"\s*-{2,}.*$", "", text.strip(), flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+(?:[,\.]\d+)?", _num, text)
    return text.strip()


@torch.no_grad()
def _sample_tokens(model, embed, prompt_ids, speaker_hidden, eos, *, max_tokens=500) -> list[int]:
    """Greedy-free autoregressive sampling (temp 0.8 / top_k 50 / top_p 0.95),
    the same params plapre uses, with a fixed seed for reproducibility."""
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(SEED)
    temperature, top_k, top_p = TEMPERATURE, 50, 0.95

    tok = embed(torch.tensor(prompt_ids, dtype=torch.long))
    inputs_embeds = torch.cat([speaker_hidden.unsqueeze(0), tok], dim=0).unsqueeze(0)
    cache = DynamicCache()
    out = model.model(inputs_embeds=inputs_embeds, past_key_values=cache, use_cache=True)
    logits = model.lm_head(out.last_hidden_state)[0, -1]
    cache = out.past_key_values

    ids: list[int] = []
    for _ in range(max_tokens):
        probs = torch.softmax(logits / temperature, dim=-1)
        topv, topi = probs.topk(min(top_k, probs.numel()))
        cum = torch.cumsum(topv, dim=-1)
        keep = cum <= top_p
        keep[0] = True  # always keep the top token
        topv, topi = topv[keep], topi[keep]
        topv = topv / topv.sum()
        nxt = int(topi[torch.multinomial(topv, 1)])
        if nxt == eos:
            break
        ids.append(nxt)
        step = embed(torch.tensor([nxt], dtype=torch.long)).unsqueeze(0)
        out = model.model(inputs_embeds=step, past_key_values=cache, use_cache=True)
        logits = model.lm_head(out.last_hidden_state)[0, -1]
        cache = out.past_key_values
    return ids


def _write_wav(path: Path, wav: np.ndarray) -> None:
    import wave

    pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main() -> None:
    global SEED, TEMPERATURE
    if "--seed" in sys.argv:
        SEED = int(sys.argv[sys.argv.index("--seed") + 1])
    if "--temp" in sys.argv:
        TEMPERATURE = float(sys.argv[sys.argv.index("--temp") + 1])
    suffix = f"_seed{SEED}_t{TEMPERATURE}"
    print(f"reference: seed={SEED} temperature={TEMPERATURE}")
    GOLDEN.mkdir(exist_ok=True)
    from kanade_tokenizer import KanadeModel, load_vocoder, vocode
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from export_kanade_decoder import _disable_attention_dropout

    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    embed = model.get_input_embeddings()
    hidden_size = model.config.hidden_size

    def tid(t: str) -> int:
        return tok.convert_tokens_to_ids(t)

    audio0, audio_end = tid("<audio_0>"), tid("<audio_12799>")
    eos = tok.eos_token_id

    # Speaker: raw 128-dim emb + learned projection -> hidden (as in inference.py).
    raw = torch.tensor(json.loads((MODELS / "speakers.json").read_text())[SPEAKER]["raw"], dtype=torch.float32)
    proj = nn.Linear(SPEAKER_DIM, hidden_size)
    from huggingface_hub import hf_hub_download

    proj.load_state_dict(torch.load(hf_hub_download(CHECKPOINT, "speaker_proj.pt"), map_location="cpu"))
    speaker_hidden = proj.eval()(raw)

    prompt = [tid("<text>"), *tok.encode(_normalize_text(SENTENCE), add_special_tokens=False), tid("<audio>")]
    tokens = _sample_tokens(model, embed, prompt, speaker_hidden, eos)
    kanade_idx = [t - audio0 for t in tokens if audio0 <= t <= audio_end]
    print(f"LM (torch sampled, seed={SEED}): {len(tokens)} tokens -> {len(kanade_idx)} kanade indices")

    # --- Reference: the REAL kanade decode + vocode (unpatched, full precision) ---
    kanade = KanadeModel.from_pretrained(KANADE_REPO).eval()
    _disable_attention_dropout(kanade)
    vocoder = load_vocoder(kanade.config.vocoder_name).eval()
    with torch.no_grad():
        mel_ref = kanade.decode(content_token_indices=torch.tensor(kanade_idx), global_embedding=raw)
        wav_ref = vocode(vocoder, mel_ref.unsqueeze(0)).squeeze().cpu().numpy()
    _write_wav(GOLDEN / f"reference_torch{suffix}.wav", wav_ref)
    print(f"reference (real torch): mel{tuple(mel_ref.shape)} -> {len(wav_ref) / SR:.2f}s -> golden/reference_torch{suffix}.wav")

    # --- Ours: the SAME tokens through the exported ONNX decoder + vocoder ---
    dec = ort.InferenceSession(str(MODELS / "kanade_decoder.onnx"), providers=["CPUExecutionProvider"])
    voc = ort.InferenceSession(str(MODELS / "hift_vocoder.onnx"), providers=["CPUExecutionProvider"])
    mel_onnx = dec.run(["mel"], {
        "content_token_indices": np.asarray(kanade_idx, np.int64),
        "global_embedding": raw.numpy(),
    })[0]
    mel_b = mel_onnx[None] if mel_onnx.ndim == 2 else mel_onnx
    wav_onnx = np.asarray(voc.run(["wav"], {"mel": mel_b.astype(np.float32)})[0]).reshape(-1)
    _write_wav(GOLDEN / f"reference_onnx{suffix}.wav", wav_onnx)

    # --- Numerical fidelity: our ONNX decode+vocode vs the real torch one ---
    mel_ref_np = mel_ref.cpu().numpy()
    mel_diff = float(np.abs(mel_ref_np - mel_onnx).max())
    n = min(len(wav_ref), len(wav_onnx))
    wav_diff = float(np.abs(wav_ref[:n] - wav_onnx[:n]).max())
    corr = float(np.corrcoef(wav_ref[:n], wav_onnx[:n])[0, 1])
    print(f"decoder mel  max|diff| (ONNX vs real torch) = {mel_diff:.4g}")
    print(f"vocoder wav  max|diff| (ONNX vs real torch) = {wav_diff:.4g}; corr = {corr:.6f}")
    print(f"wrote golden/reference_torch{suffix}.wav (listen) and golden/reference_onnx{suffix}.wav (ours, same tokens)")


if __name__ == "__main__":
    main()
