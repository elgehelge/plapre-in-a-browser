# models

Converted model artifacts go here (git-ignored — see root `.gitignore`).
The web app loads them from `/models/…` and reports which are missing.

Expected files (produce via `conversion/`):

| File                   | Produced by                          |
| ---------------------- | ------------------------------------ |
| `lm/model.onnx` (+data)| `conversion/export_lm.py`            |
| `kanade_decoder.onnx`  | `conversion/export_kanade_decoder.py`|
| `hift_vocoder.onnx`    | `conversion/export_hift_vocoder.py`  |
| `tokenizer.json`       | copy from the `syvai/plapre-pico` repo |
| `speakers.json`        | `conversion/precompute_speakers.py`  |
