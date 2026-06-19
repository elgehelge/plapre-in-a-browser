# Plan

Goal: run Plapre (Danish TTS) fully in the browser on `onnxruntime-web`
(WebGPU, WASM fallback). No server, no cloud. (Path A only — a local sidecar
server is explicitly out of scope.)

## Phase 0 — De-risk the vocoder (GATE: do this first)

The HiFT vocoder and Kanade decoder are non-standard architectures (STFT/iSTFT,
complex math, weight-normed convs) that historically are the painful ops for
ONNX Runtime Web. If these will not run on the web runtime, the whole approach
must change — so prove it before investing in the LM port.

- [ ] Export **HiFT vocoder** to ONNX (`conversion/export_hift_vocoder.py`).
- [ ] Export **Kanade decoder** to ONNX (`conversion/export_kanade_decoder.py`).
- [ ] Produce golden fixtures (`conversion/smoke_reference.py`): for a fixed
      sentence + built-in speaker, dump the generated audio-token ids, the
      decoder mel, and the final 24 kHz wav.
- [ ] In the web app, load decoder + vocoder, feed the golden audio-token ids,
      and reproduce the golden wav within tolerance. Test **WebGPU and WASM**.

**Success criterion:** golden sentence decodes to correct-sounding audio in the
browser on both backends.
**If it fails:** try the `kanade-25hz` (Vocos 24 kHz vocoder) variant instead of
`kanade-25hz-clean` (HiFT), which may export more cleanly; otherwise stop.

## Phase 1 — Port the LM generation

- [ ] Export **Plapre Pico LM** to ONNX with `inputs_embeds` input + KV cache
      (`conversion/export_lm.py`, via `optimum`). Nano later if Pico holds.
- [ ] Precompute speaker hidden vectors offline: `speaker_proj @ speakers[name]`
      for the 5 built-in voices, ship as a small JSON/bin
      (`conversion/precompute_speakers.py`). This removes the projection layer
      from the runtime entirely.
- [ ] JS: BPE tokenize via the HF `tokenizer.json`; build prompt
      `[<text>] + ids + [<audio>]`; embed tokens; prepend the speaker hidden vec.
- [ ] JS: autoregressive loop with KV cache + temperature/top-k/top-p sampling,
      stop at EOS / `max_tokens`, collect audio-token ids.
- [ ] Parity-check generated token ids against golden (greedy/seeded) output.

## Phase 2 — Wire the full chain + Danish normalization

- [ ] Map audio-token ids → Kanade indices; feed Phase-0 decoder + vocoder → PCM.
- [ ] Danish text normalization in JS (`web/src/pipeline/normalize.ts`):
      number→words, whitespace, sentence splitting. Mirror `plapre`'s
      `_normalize_text` / `_split_sentences`. Validate the Danish
      number-to-words against `num2words(lang="da")` using golden cases.
- [ ] Per-sentence synthesis; optional streaming by decoding audio-token chunks
      (~1 s) so playback can start early.

## Phase 3 — Packaging for reuse (MV3-ready)

- [ ] Expose the pipeline as the provider-neutral `Engine` from
      [docs/INTERFACE.md](INTERFACE.md): streaming `synthesize() →
      AsyncIterable<PcmChunk>` + buffered `synthesizeToPcm()`, `listVoices()`,
      `AbortSignal` cancellation, backend toggle (WebGPU→WASM), and model
      download + Cache/OPFS caching + warm-up (silent decode on load) with
      `onProgress`.
- [ ] Document how to host it in a Chrome MV3 **offscreen document** (not the
      service worker, not a content script), incl. the
      `cross_origin_embedder_policy` / `cross_origin_opener_policy` manifest keys
      needed for threaded WASM.

## Phase 3.5 — Drop-in API adapters

Make the engine a plug-in replacement for the two common hosted TTS APIs (see
[docs/INTERFACE.md](INTERFACE.md) for the request/voice/format mapping). Each
adapter is a thin layer over the engine: request mapping, voice-id mapping, and
a format encoder over the canonical 24 kHz PCM stream.

- [ ] PCM-passthrough + format encoders (`wav`, `mp3`/`opus` via a wasm encoder);
      `pcm` / `pcm_24000` need no resampling (native rate matches).
- [ ] **OpenAI adapter** — `audio.speech.create({ input, voice, speed,
      response_format })` shape; streaming + buffered.
- [ ] **ElevenLabs adapter** — `textToSpeech.convert/stream(voiceId, { text,
      voice_settings, output_format })` shape; map the subset of `voice_settings`
      that has an engine analogue.
- [ ] Voice-id mapping tables (provider voice names ↔ built-in Danish speakers).

## Phase 4 — Validate

- [ ] Real-time factor on WebGPU vs WASM (Pico, then Nano).
- [ ] Quality A/B vs reference Python output.
- [ ] Memory footprint.

## Open risks (validate early, cheap first)

1. **Vocoder ONNX/ORT-Web op support** — the make-or-break item (Phase 0).
2. **`inputs_embeds` generation** on the chosen runtime — folding the speaker
   vector into a precomputed hidden avoids needing a separate projection op.
3. **Pico-on-WASM real-time factor** — decides whether WebGPU is mandatory.
4. **Re-hosting weights** — upstream HF repos are gated; CC-BY permits
   redistributing our ONNX exports with attribution.
5. **num2words(da) parity** — Danish numerals are irregular (halvtreds, tres,
   firs…); port carefully and cover with golden cases.
