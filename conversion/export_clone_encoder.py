"""Phase 5: export the Plapre/Kanade voice-clone encoder to ONNX.

Voice cloning derives the same 128-dim global speaker embedding that built-in
speakers ship as. Only the GLOBAL branch is needed (not content/FSQ):

    waveform (config.sample_rate) -> SSL frontend (WavLM layers 1..2, averaged)
                                  -> GlobalEncoder (ConvNext + AttentiveStatsPool)
                                  -> 128-dim raw embedding

This is the PUBLIC half of cloning (frothywater/kanade-25hz-clean). The raw
embedding feeds the DECODER directly; turning it into an LM `hidden` vector needs
`speaker_proj` (gated) — exported separately by precompute_speakers.py.

The SSL resample (config.sample_rate -> 16 kHz) is kept INSIDE the graph (the
exact torchaudio sinc kernel) so the browser only has to resample to
config.sample_rate; this avoids a JS resampler-parity problem on the SSL path.

GATE: reproduce a reference embedding under onnxruntime-web (cosine vs torch).
Here we validate ORT-CPU vs torch on the same input (graph parity); the WavLM
frontend is the op-support risk this gate retires.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO = "frothywater/kanade-25hz-clean"
MODELS = Path(__file__).parent.parent / "web" / "public" / "models"
OUT = MODELS / "clone_encoder.onnx"
GOLDEN = MODELS / "clone_golden.json"
SEED = 1234
COSINE_MIN = 0.9995  # ORT-CPU vs torch on the same waveform


class CloneEncoderWrapper(torch.nn.Module):
    """waveform (samples,) at config.sample_rate -> 128-dim global embedding."""

    def __init__(self, kanade: torch.nn.Module) -> None:
        super().__init__()
        self.ssl = kanade.ssl_feature_extractor
        self.global_ssl_layers = list(kanade.global_ssl_layers)  # e.g. [1, 2]
        self.global_encoder = kanade.global_encoder

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        feats = self.ssl(waveform, num_layers=max(self.global_ssl_layers))  # list per layer
        selected = [feats[i - 1] for i in self.global_ssl_layers]
        global_feat = torch.stack(selected, dim=0).mean(dim=0)  # (1, T, C)
        emb = self.global_encoder(global_feat)  # (1, 128)
        return emb.squeeze(0)


def main() -> None:
    from kanade_tokenizer import KanadeModel

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    kanade = KanadeModel.from_pretrained(REPO).eval()
    sr = kanade.config.sample_rate
    wrapper = CloneEncoderWrapper(kanade).eval()
    print(f"config.sample_rate={sr}, global_ssl_layers={wrapper.global_ssl_layers}")

    # ~2 s of audio at the model sample rate. Graph parity does not need real
    # speech; a fixed random clip exercises the full op set.
    waveform = torch.randn(1, 2 * sr) * 0.05

    with torch.no_grad():
        ref = wrapper(waveform).cpu().numpy()
        # Sanity: the slim wrapper matches the full encode() global branch
        # (no waveform padding here — the attentive-stats pool makes the few
        # padded samples negligible; reported below).
        full = kanade.encode(waveform.squeeze(0), return_content=False).global_embedding.cpu().numpy()
    cos_full = _cosine(ref, full)
    print(f"slim-vs-encode() global cosine (padding effect): {cos_full:.5f}")

    torch.onnx.export(
        wrapper,
        (waveform,),
        str(OUT),
        input_names=["waveform"],
        output_names=["embedding"],
        dynamic_axes={"waveform": {1: "samples"}},
        dynamo=False,  # dynamo trips on WavLM's data-dependent control flow
        opset_version=17,
    )
    print(f"Exported clone encoder -> {OUT}")

    _validate_cpu(waveform.numpy(), ref)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _validate_cpu(waveform: np.ndarray, ref: np.ndarray) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    out = sess.run(["embedding"], {"waveform": waveform.astype(np.float32)})[0]
    cos = _cosine(out, ref)
    status = "OK" if cos >= COSINE_MIN else "FAIL"
    print(f"ORT-CPU clone-encoder parity: dim={out.shape} cosine={cos:.6f} [{status}]")
    if cos < COSINE_MIN:
        raise SystemExit(f"clone encoder parity {cos} below {COSINE_MIN}")

    # Golden for the in-browser gate (web/phase5.html): same waveform + embedding.
    GOLDEN.write_text(
        json.dumps(
            {
                "sampleRate": 24000,
                "waveform": waveform.reshape(-1).astype(float).tolist(),
                "embedding": out.reshape(-1).astype(float).tolist(),
            }
        )
    )
    print(f"wrote browser golden -> {GOLDEN}")


if __name__ == "__main__":
    main()
