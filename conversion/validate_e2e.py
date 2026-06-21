"""End-to-end smoke test of the FULL exported pipeline, ONNX only.

Chains the three exported graphs exactly as the browser does — no torch, no gated
`plapre` package, no browser — to prove text actually turns into audio:

  text --(lm.onnx)----------> audio tokens   (greedy OR sampled)
       --(- <audio_0>)-------> kanade content indices
       --(kanade_decoder.onnx, + raw 128-dim speaker emb)--> mel
       --(hift_vocoder.onnx)--> 24 kHz waveform  -> golden/e2e[_sampled].wav

Mirrors web/src/pipeline/{lm,decoder,vocoder,sampling}.ts — including a faithful
port of the mulberry32 RNG + temperature/top-k/top-p sampler, so for a given seed
the browser would produce the identical token stream.

Usage:
  python validate_e2e.py            # sampled (temp 0.8/topK 50/topP 0.95, seed 0)
  python validate_e2e.py --greedy   # argmax (matches the token golden)
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from smoke_reference import SENTENCE, SPEAKER, _normalize

MODELS = Path(__file__).parent.parent / "web" / "public" / "models"
SR = 24000

# Browser engine defaults (web/src/engine/engine.ts DEFAULT_GENERATION).
# SEED is overridable with --seed N; TEMPERATURE with --temp F.
TEMPERATURE, TOP_K, TOP_P, SEED = 0.8, 50, 0.95, 0


def _imul(a: int, b: int) -> int:
    return (a * b) & 0xFFFFFFFF


def make_rng(seed: int):
    """Port of mulberry32 from web/src/pipeline/sampling.ts (bit-exact)."""
    a = seed & 0xFFFFFFFF

    def rng() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = _imul(t ^ (t >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rng


def sample_token(logits: np.ndarray, rng, temperature: float = None, top_k: int = TOP_K, top_p: float = TOP_P) -> int:
    """Port of sample() from web/src/pipeline/sampling.ts."""
    if temperature is None:
        temperature = TEMPERATURE
    if temperature <= 0:
        return int(logits.argmax())
    cands = sorted(
        ((i, logits[i] / temperature) for i in range(len(logits))),
        key=lambda c: c[1],
        reverse=True,
    )
    if 0 < top_k < len(cands):
        cands = cands[:top_k]
    probs = np.array([c[1] for c in cands], dtype=np.float64)
    probs = np.exp(probs - probs.max())
    probs /= probs.sum()
    if top_p < 1:
        cum = np.cumsum(probs)
        cutoff = int(np.searchsorted(cum, top_p) + 1)
        cands = cands[:cutoff]
        probs = probs[:cutoff]
        probs = probs / probs.sum()
    r = rng()
    acc = 0.0
    for (idx, _), p in zip(cands, probs):
        acc += p
        if r <= acc:
            return int(idx)
    return int(cands[-1][0])


def _sess(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run_lm(tok: Tokenizer, greedy: bool) -> list[int]:
    meta = json.loads((MODELS / "lm" / "meta.json").read_text())
    L, KV, HD, H = meta["numLayers"], meta["kvHeads"], meta["headDim"], meta["hidden"]
    spk = json.loads((MODELS / "speakers.json").read_text())[SPEAKER]
    hidden = np.asarray(spk["hidden"], np.float32).reshape(1, H)

    def tid(t: str) -> int:
        return tok.token_to_id(t)

    prompt = [tid("<text>"), *tok.encode(_normalize(SENTENCE), add_special_tokens=False).ids, tid("<audio>")]
    eos = tid("<eos>")

    sess = _sess(MODELS / "lm" / "model.onnx")
    pn = [f"past_key_values.{i}.{k}" for i in range(L) for k in ("key", "value")]
    prn = [f"present.{i}.{k}" for i in range(L) for k in ("key", "value")]

    def step(ids, pfx, past):
        feeds = {"input_ids": np.asarray(ids, np.int64), "prefix_embeds": pfx}
        feeds.update({n: t for n, t in zip(pn, past)})
        outs = sess.run(["logits", *prn], feeds)
        return outs[0], outs[1:]

    rng = make_rng(SEED)
    past = [np.zeros((1, KV, 0, HD), np.float32) for _ in range(2 * L)]
    logits, past = step(prompt, hidden, past)
    ids: list[int] = []
    for _ in range(500):
        row = logits[0, -1]
        nxt = int(row.argmax()) if greedy else sample_token(row, rng)
        if nxt == eos:
            break
        ids.append(nxt)
        logits, past = step([nxt], np.zeros((0, H), np.float32), past)
    return ids


def main() -> None:
    global SEED, TEMPERATURE
    greedy = "--greedy" in sys.argv
    if "--seed" in sys.argv:
        SEED = int(sys.argv[sys.argv.index("--seed") + 1])
    if "--temp" in sys.argv:
        TEMPERATURE = float(sys.argv[sys.argv.index("--temp") + 1])
    if greedy:
        out_wav = Path(__file__).parent / "golden" / "e2e.wav"
    else:
        out_wav = Path(__file__).parent / "golden" / f"e2e_seed{SEED}_t{TEMPERATURE}.wav"
    print(f"mode: {'greedy' if greedy else f'sampled (T={TEMPERATURE} topK={TOP_K} topP={TOP_P} seed={SEED})'}")
    tok = Tokenizer.from_file(str(MODELS / "tokenizer.json"))
    audio0 = tok.token_to_id("<audio_0>")
    audio_end = tok.token_to_id("<audio_12799>")

    audio_tokens = _run_lm(tok, greedy)
    kanade = [t - audio0 for t in audio_tokens if audio0 <= t <= audio_end]
    print(f"LM: {len(audio_tokens)} tokens -> {len(kanade)} kanade indices")
    if not kanade:
        raise SystemExit("no audio tokens generated")

    raw = np.asarray(json.loads((MODELS / "speakers.json").read_text())[SPEAKER]["raw"], np.float32)

    dec = _sess(MODELS / "kanade_decoder.onnx")
    mel = dec.run(["mel"], {
        "content_token_indices": np.asarray(kanade, np.int64),
        "global_embedding": raw,
    })[0]
    print(f"decoder: mel shape {mel.shape}")

    mel_b = mel[None] if mel.ndim == 2 else mel  # ensure (B, n_mels, T)
    voc = _sess(MODELS / "hift_vocoder.onnx")
    wav = np.asarray(voc.run(["wav"], {"mel": mel_b.astype(np.float32)})[0]).reshape(-1)
    dur = len(wav) / SR
    print(f"vocoder: {len(wav)} samples = {dur:.2f}s @ {SR} Hz")

    finite = bool(np.isfinite(wav).all())
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    rms = float(np.sqrt(np.mean(wav**2))) if len(wav) else 0.0
    print(f"audio: finite={finite} peak={peak:.3f} rms={rms:.4f}")

    out_wav.parent.mkdir(exist_ok=True)
    pcm16 = np.clip(wav, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm16.tobytes())
    print(f"wrote {out_wav}")

    problems = []
    if not finite:
        problems.append("non-finite samples")
    if not (0.05 < peak <= 1.01):
        problems.append(f"suspicious peak {peak:.3f}")
    if not (0.5 < dur < 30):
        problems.append(f"suspicious duration {dur:.2f}s")
    if problems:
        raise SystemExit("E2E sanity FAILED: " + "; ".join(problems))
    print("E2E PASSED: text -> audio through all three exported ONNX graphs.")


if __name__ == "__main__":
    main()
