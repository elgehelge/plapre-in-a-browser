# models

Converted model artifacts go here (git-ignored — see root `.gitignore`).
The web app loads them from `/models/…` and reports which are missing.

Expected files (produce via `conversion/` — see its README for the full recipe):

| File                              | Produced by                           | Phase | Gated? |
| --------------------------------- | ------------------------------------- | ----- | ------ |
| `kanade_decoder.onnx` + `.onnx.data` | `conversion/export_kanade_decoder.py` | 0 | no |
| `hift_vocoder.onnx` + `.onnx.data`   | `conversion/export_hift_vocoder.py`   | 0 | no |
| `phase0_golden.json`              | `conversion/gen_phase0_golden.py` (browser-gate reference) | 0 | no |
| `clone_encoder.onnx`              | `conversion/export_clone_encoder.py`  | 5 | no |
| `clone_golden.json`              | `conversion/export_clone_encoder.py` (browser-gate reference) | 5 | no |
| `lm_toy/model.onnx` + `meta.json`, `phase1_toy_golden.json` | `conversion/gen_toy_lm.py` (Phase 1 browser gate) | 1 | no |
| `lm/model.onnx` (+ data) + `lm/meta.json` | `conversion/export_lm.py`     | 1 | **yes** |
| `tokenizer.json` (+ `config.json`) | `conversion/fetch_tokenizer.py`     | 1 | **yes** |
| `speakers.json`                   | `conversion/precompute_speakers.py`   | 1 | **yes** |
| `speaker_proj.json`               | `conversion/precompute_speakers.py` (runtime voice cloning) | 5 | **yes** |

"Gated" files come from `syvai/plapre-pico`, which requires accepting the model
license + `huggingface-cli login` (see `conversion/_gated.py` and the conversion
README). Everything else is public and reproducible today.

The TorchDynamo/legacy exporter writes large tensors to a sibling
`<model>.onnx.data` file; keep it next to the `.onnx`. onnxruntime-web loads it
via the `externalData` session option, wired automatically in
`web/src/pipeline/ort.ts` (`createSession`'s `dataFile`).
