"""End-to-end smoke test of the FULL exported pipeline, ONNX only.

Chains the three exported graphs exactly as the browser does — no torch, no gated
`plapre` package, no browser — to prove text actually turns into audio:

  text --(lm.onnx, greedy)--> audio tokens
       --(- <audio_0>)-------> kanade content indices
       --(kanade_decoder.onnx, + raw 128-dim speaker emb)--> mel
       --(hift_vocoder.onnx)--> 24 kHz waveform  -> golden/e2e.wav

Mirrors web/src/pipeline/{lm,decoder,vocoder}.ts. A sane WAV (right duration, no
NaNs, reasonable amplitude) means the whole chain is wired and runs.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from smoke_reference import SENTENCE, SPEAKER, _normalize

MODELS = Path(__file__).parent.parent / "web" / "public" / "models"
OUT_WAV = Path(__file__).parent / "golden" / "e2e.wav"
SR = 24000


def _sess(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run_lm(tok: Tokenizer) -> list[int]:
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

    past = [np.zeros((1, KV, 0, HD), np.float32) for _ in range(2 * L)]
    logits, past = step(prompt, hidden, past)
    ids: list[int] = []
    for _ in range(500):
        nxt = int(logits[0, -1].argmax())
        if nxt == eos:
            break
        ids.append(nxt)
        logits, past = step([nxt], np.zeros((0, H), np.float32), past)
    return ids


def main() -> None:
    tok = Tokenizer.from_file(str(MODELS / "tokenizer.json"))
    audio0 = tok.token_to_id("<audio_0>")
    audio_end = tok.token_to_id("<audio_12799>")

    audio_tokens = _run_lm(tok)
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

    OUT_WAV.parent.mkdir(exist_ok=True)
    pcm16 = np.clip(wav, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    with wave.open(str(OUT_WAV), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm16.tobytes())
    print(f"wrote {OUT_WAV}")

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
