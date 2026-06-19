# models

Converted model artifacts go here (git-ignored — see root `.gitignore`).
The web app loads them from `/models/…` and reports which are missing.

Expected files (produce via `conversion/` — see its README for the full recipe):

| File                              | Produced by                           | Phase |
| --------------------------------- | ------------------------------------- | ----- |
| `kanade_decoder.onnx` + `.onnx.data` | `conversion/export_kanade_decoder.py` | 0 |
| `hift_vocoder.onnx` + `.onnx.data`   | `conversion/export_hift_vocoder.py`   | 0 |
| `phase0_golden.json`              | `conversion/gen_phase0_golden.py` (browser-gate reference) | 0 |
| `lm/model.onnx` (+ data)          | `conversion/export_lm.py`             | 1 |
| `tokenizer.json`                  | copy from the `syvai/plapre-pico` repo | 1 |
| `speakers.json`                   | `conversion/precompute_speakers.py`   | 1 |

The TorchDynamo exporter writes large tensors to a sibling `<model>.onnx.data`
file; keep it next to the `.onnx`. onnxruntime-web loads it via the
`externalData` session option (see `web/src/phase0.ts` / `web/src/pipeline/ort.ts`).
