# models

Converted model artifacts go here (git-ignored — see root `.gitignore`).
The web app loads them from `/models/…` and reports which are missing.

## Layout: shared vs. per-variant

The Kanade **decoder / vocoder / clone-encoder** are identical across model
variants and sit at the root. The **LM-side** artifacts (the LM graph + meta,
tokenizer, and speaker tables) differ per variant and live under the variant's
own sub-directory (`pico/`, `nano/`).

```
models/
  kanade_decoder.onnx (+ .onnx.data)   # shared
  hift_vocoder.onnx (+ .onnx.data)     # shared
  clone_encoder.onnx                   # shared
  pico/
    lm/model.onnx (+ data), lm/meta.json
    tokenizer.json, config.json
    speakers.json, speaker_proj.json
  nano/
    lm/model.onnx (+ data), lm/meta.json
    tokenizer.json, config.json
    speakers.json, speaker_proj.json
```

The sub-paths mirror `PLAPRE_MODELS[<variant>].prefix` in
`web/src/pipeline/models.ts`.

## Expected files

Produce via `conversion/` — see its README for the full recipe. The LM-side rows
repeat per variant under its sub-path (e.g. `nano/lm/model.onnx`).

| File                              | Produced by                           | Phase | Gated? | Per-variant? |
| --------------------------------- | ------------------------------------- | ----- | ------ | ------------ |
| `kanade_decoder.onnx` + `.onnx.data` | `conversion/export_kanade_decoder.py` | 0 | no | shared |
| `hift_vocoder.onnx` + `.onnx.data`   | `conversion/export_hift_vocoder.py`   | 0 | no | shared |
| `phase0_golden.json`              | `conversion/gen_phase0_golden.py` (browser-gate reference) | 0 | no | shared |
| `clone_encoder.onnx`              | `conversion/export_clone_encoder.py`  | 5 | no | shared |
| `clone_golden.json`              | `conversion/export_clone_encoder.py` (browser-gate reference) | 5 | no | shared |
| `lm_toy/model.onnx` + `meta.json`, `phase1_toy_golden.json` | `conversion/gen_toy_lm.py` (Phase 1 browser gate) | 1 | no | shared |
| `<variant>/lm/model.onnx` (+ data) + `lm/meta.json` | `conversion/export_lm.py`     | 1 | **yes** | per-variant |
| `<variant>/tokenizer.json` (+ `config.json`) | `conversion/fetch_tokenizer.py`     | 1 | **yes** | per-variant |
| `<variant>/speakers.json`                   | `conversion/precompute_speakers.py`   | 1 | **yes** | per-variant |
| `<variant>/speaker_proj.json`               | `conversion/precompute_speakers.py` (runtime voice cloning) | 5 | **yes** | per-variant |

"Gated" files come from the variant's checkpoint (`syvai/plapre-pico` /
`syvai/plapre-nano`), which requires accepting the model license +
`huggingface-cli login` (see `conversion/_gated.py` and the conversion README).
Everything else is public and reproducible today.

The TorchDynamo/legacy exporter writes large tensors to a sibling
`<model>.onnx.data` file; keep it next to the `.onnx`. onnxruntime-web loads it
via the `externalData` session option, wired automatically in
`web/src/pipeline/ort.ts` (`createSession`'s `dataFile`).
