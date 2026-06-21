"""Render a long, multi-sentence text through OUR full ONNX pipeline at several
temperatures, the way the browser engine would: normalize, split into sentences,
synthesize each sentence independently (LM -> decoder -> vocoder), and concatenate
with a short silence between them.

Loads the 521 MB LM once and reuses it across every sentence and temperature.
Output: golden/demo_t{TEMP}.wav (one per temperature).

  python gen_demo_audio.py [--text-file FILE] [--seed N] [--temps 0.5,0.6,0.7]
                           [--speaker tor]

Sentence splitting + normalization mirror plapre.inference (audited 1:1); the
sampler + per-sentence RNG reset mirror web/src/pipeline/{lm,sampling}.ts.
"""

from __future__ import annotations

import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

from gen_reference_audio import _normalize_text
from validate_e2e import MODELS, SR, TOP_K, TOP_P, _sess, make_rng, sample_token

from tokenizers import Tokenizer

GOLDEN = Path(__file__).parent / "golden"
SILENCE_SEC = 0.1
MAX_TOKENS = 500


def split_sentences(text: str) -> list[str]:
    """Copy of plapre.inference.Plapre._split_sentences (normalize first)."""
    text = _normalize_text(text)
    parts = re.split(r"(?<=\w{2}[.!?])\s+", text)
    out: list[str] = []
    for p in parts:
        p = re.sub(r"^[-\u2013\u2014]\s+", "", p.strip())
        if p:
            out.append(p)
    return out


def _arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> None:
    text_file = Path(_arg("--text-file", str(GOLDEN / "demo_text.txt")))
    seed = int(_arg("--seed", "0"))
    temps = [float(t) for t in _arg("--temps", "0.5,0.6,0.7").split(",")]
    speaker = _arg("--speaker", "tor")

    sentences = split_sentences(text_file.read_text())
    print(f"{len(sentences)} sentence(s); speaker={speaker}; seed={seed}; temps={temps}")
    for i, s in enumerate(sentences):
        print(f"  [{i}] {s[:70]}{'…' if len(s) > 70 else ''}")

    tok = Tokenizer.from_file(str(MODELS / "tokenizer.json"))
    text_tag, audio_tag = tok.token_to_id("<text>"), tok.token_to_id("<audio>")
    audio0, audio_end, eos = (
        tok.token_to_id("<audio_0>"),
        tok.token_to_id("<audio_12799>"),
        tok.token_to_id("<eos>"),
    )

    meta = json.loads((MODELS / "lm" / "meta.json").read_text())
    L, KV, HD, H = meta["numLayers"], meta["kvHeads"], meta["headDim"], meta["hidden"]
    spk = json.loads((MODELS / "speakers.json").read_text())[speaker]
    hidden = np.asarray(spk["hidden"], np.float32).reshape(1, H)
    raw = np.asarray(spk["raw"], np.float32)

    lm = _sess(MODELS / "lm" / "model.onnx")
    dec = _sess(MODELS / "kanade_decoder.onnx")
    voc = _sess(MODELS / "hift_vocoder.onnx")
    pn = [f"past_key_values.{i}.{k}" for i in range(L) for k in ("key", "value")]
    prn = [f"present.{i}.{k}" for i in range(L) for k in ("key", "value")]

    def lm_step(ids, pfx, past):
        feeds = {"input_ids": np.asarray(ids, np.int64), "prefix_embeds": pfx}
        feeds.update({n: t for n, t in zip(pn, past)})
        outs = lm.run(["logits", *prn], feeds)
        return outs[0], outs[1:]

    def gen_sentence(sentence: str, temperature: float) -> np.ndarray:
        prompt = [text_tag, *tok.encode(sentence, add_special_tokens=False).ids, audio_tag]
        rng = make_rng(seed)  # reset per sentence, like web/src/engine/engine.ts
        past = [np.zeros((1, KV, 0, HD), np.float32) for _ in range(2 * L)]
        logits, past = lm_step(prompt, hidden, past)
        ids: list[int] = []
        for _ in range(MAX_TOKENS):
            nxt = sample_token(logits[0, -1], rng, temperature, TOP_K, TOP_P)
            if nxt == eos:
                break
            ids.append(nxt)
            logits, past = lm_step([nxt], np.zeros((0, H), np.float32), past)
        kanade = [t - audio0 for t in ids if audio0 <= t <= audio_end]
        if not kanade:
            return np.zeros(0, np.float32)
        mel = dec.run(["mel"], {
            "content_token_indices": np.asarray(kanade, np.int64),
            "global_embedding": raw,
        })[0]
        mel_b = mel[None] if mel.ndim == 2 else mel
        return np.asarray(voc.run(["wav"], {"mel": mel_b.astype(np.float32)})[0]).reshape(-1)

    silence = np.zeros(int(SILENCE_SEC * SR), np.float32)
    for temp in temps:
        chunks: list[np.ndarray] = []
        for i, s in enumerate(sentences):
            wav = gen_sentence(s, temp)
            if wav.size:
                chunks.append(wav)
                if i < len(sentences) - 1:
                    chunks.append(silence)
        full = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        out = GOLDEN / f"demo_t{temp}.wav"
        pcm = (np.clip(full, -1, 1) * 32767).astype("<i2")
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        print(f"temp {temp}: {len(full) / SR:.2f}s -> {out}")


if __name__ == "__main__":
    main()
