---
license: cc-by-4.0
language:
- da
pipeline_tag: text-to-speech
base_model: syvai/plapre-pico
tags:
- onnx
- onnxruntime-web
- webgpu
- text-to-speech
- danish
- in-browser
---

# Plapre — ONNX weights for in-browser inference

Converted **ONNX** artifacts for running [**Plapre**](https://huggingface.co/syvai/plapre-pico),
an open-source Danish text-to-speech model, **fully in the browser** on
[`onnxruntime-web`](https://onnxruntime.ai/docs/tutorials/web/) (WebGPU, with a
WASM fallback).

These files are consumed by the [**plapre-in-a-browser**](https://github.com/elgehelge/plapre-in-a-browser)
library and its [live demo](https://elgehelge.github.io/plapre-in-a-browser/).
They are a format conversion (PyTorch → ONNX) of the upstream model — no
retraining — verified to reproduce the PyTorch reference to float precision.

## Files

| File | Stage | Notes |
| ---- | ----- | ----- |
| `lm/model.onnx` (+ `meta.json`) | Plapre language model | SmolLM2/LLaMA, autoregressive; ~118M |
| `kanade_decoder.onnx` (+ `.onnx.data`) | Kanade decoder | audio tokens + speaker emb → mel |
| `hift_vocoder.onnx` (+ `.onnx.data`) | HiFT vocoder | mel → 24 kHz waveform |
| `clone_encoder.onnx` | Kanade clone encoder | reference audio → 128-dim speaker embedding |
| `tokenizer.json`, `config.json` | tokenizer / config | BPE + special audio tokens |
| `speakers.json`, `speaker_proj.json` | built-in voices | precomputed embeddings (`tor, ida, liv, ask, kaj`) |

## Usage

```ts
import { loadPlapreEngine } from "plapre-in-a-browser";

const engine = await loadPlapreEngine({
  modelsBaseUrl: "https://huggingface.co/elgehelge/plapre-onnx-web/resolve/main",
});
const { samples, sampleRate } = await engine.synthesizeToPcm({
  text: "Hej, hvordan har du det i dag?",
  voice: "ida",
});
```

## License & attribution

Distributed under **CC BY 4.0** (the most restrictive of the upstream licenses).
Each artifact derives from a permissively licensed work:

| Artifact | Upstream | License |
| -------- | -------- | ------- |
| `lm/*`, `tokenizer.json`, `config.json`, `speakers.json`, `speaker_proj.json` | [Plapre](https://huggingface.co/syvai/plapre-pico) by [syv.ai](https://syv.ai/produkter/plapre) | CC BY 4.0 |
| `kanade_decoder.*`, `clone_encoder.onnx` | [Kanade](https://huggingface.co/frothywater/kanade-25hz-clean) (frothywater) | MIT |
| `hift_vocoder.*` | HiFT, from [CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice) | Apache-2.0 |
| clone-encoder SSL frontend | [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) (Microsoft) | MIT |

These are format conversions (PyTorch → ONNX), not retraining. If you
redistribute these artifacts, keep this attribution and the MIT/Apache notices.
Full notices: https://github.com/elgehelge/plapre-in-a-browser/blob/main/NOTICE
