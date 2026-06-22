# Architecture

Reference implementation: [`syv-ai/plapre`](https://github.com/syv-ai/plapre)
(`plapre/inference.py`). This document records the exact pipeline so the
browser port can match it stage by stage.

## Constants (from the reference)

| Name           | Value                       |
| -------------- | --------------------------- |
| sample rate    | 24000 Hz                    |
| speaker dim    | 128                         |
| hidden size    | 960 (Nano) / 576 (Pico)     |
| Kanade model   | `frothywater/kanade-25hz-clean` |
| audio tokens   | `<audio_0>` … `<audio_12799>` (12800 codebook) |
| token rate     | 25 audio tokens / second    |
| special tokens | `<text>`, `<audio>`, EOS    |

## Stage 1 — Text → prompt

1. Normalize text (`_normalize_text`):
   - strip trailing `--- caption` separators,
   - collapse all whitespace to single spaces,
   - replace numbers with Danish words via `num2words(lang="da")`
     (e.g. `2,1` → `to komma en`).
2. Optional sentence split (`_split_sentences`): split on `\w{2}[.!?]` followed
   by whitespace; strip leading Danish dialogue dashes (`- `).
3. BPE tokenize (no special tokens) and build:
   `prompt_ids = [<text>] + text_ids + [<audio>]`.

## Stage 2 — LM generation (the tricky part)

The model is conditioned on a speaker by **prepending a speaker hidden vector as
the first input embedding** — generation runs on `inputs_embeds`, not token ids:

```
token_embeds = embed_tokens(prompt_ids)          # (n, hidden)
speaker_hidden = speaker_proj(speaker_emb_128)    # (hidden,)   Linear(128→hidden)
inputs_embeds  = concat([speaker_hidden[None], token_embeds])  # (n+1, hidden)
```

Then autoregressive sampling (`temperature`, `top_k`, `top_p`, `max_tokens`)
produces audio-token ids. The reference uses vLLM with `enable_prompt_embeds`;
**the browser port replaces this with a hand-rolled ONNX Runtime Web loop**
(KV cache + JS sampling). vLLM is server-only and not used in the browser.

Browser simplification: precompute `speaker_hidden` for the 5 built-in speakers
offline, so neither `speaker_proj` nor a 128→hidden op is needed at runtime.

## Stage 3 — Tokens → audio

```
kanade_indices = [tid - id(<audio_0>) for tid in tokens if in audio range]
mel = kanade.decode(content_token_indices=kanade_indices,
                    global_embedding=speaker_emb_128)   # Kanade decoder
waveform = vocode(hift_vocoder, mel)                    # HiFT → 24 kHz
```

Note `global_embedding` here is the **raw 128-dim** speaker embedding (the same
one used to build `speaker_hidden`), not the projected hidden vector.

## What we deliberately drop

- **Kanade encoder + WavLM SSL frontend** — only used for voice cloning
  (`extract_speaker`). Built-in speakers ship as precomputed 128-dim embeddings
  (`speakers.json` in the model repo), so the encoder is out of scope.
- **vLLM / GGUF / `plapre-serve`** — server inference paths, not used in-browser.

## Browser model artifacts (target)

Dropped into `web/public/models/`:

- `lm.onnx` (+ external data) — Plapre LM with `inputs_embeds` + KV cache.
- `kanade_decoder.onnx` — content tokens + 128-dim emb → mel.
- `hift_vocoder.onnx` — mel → waveform.
- `tokenizer.json` — BPE tokenizer (from the model repo).
- `speakers.json` — built-in raw 128-dim embeddings (for the decoder) AND their
  precomputed hidden vectors (for the LM prompt).

## Model variants (Pico / Nano)

Plapre ships in two sizes that share this exact pipeline; only the LM differs
(hidden 576 for Pico, 960 for Nano). Because the decoder consumes the raw 128-dim
speaker embedding and the runtime reads the LM hidden size from `lm/meta.json`,
the only variant-dependent pieces are the **LM graph + meta, tokenizer, and
speaker tables** — the Kanade decoder/vocoder/clone-encoder are shared.

Variant-specific artifacts live under the variant's sub-path
(`web/src/pipeline/models.ts` `prefix`): `pico/` and `nano/`. The shared Kanade
artifacts stay at the root. Select a variant with `loadPlapreEngine({ model })`.
