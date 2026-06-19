# Plan

Goal: run Plapre (Danish TTS) fully in the browser on `onnxruntime-web`
(WebGPU, WASM fallback). No server, no cloud. (Path A only — a local sidecar
server is explicitly out of scope.)

## Phase 0 — De-risk the vocoder (GATE: do this first)

The HiFT vocoder and Kanade decoder are non-standard architectures (STFT/iSTFT,
complex math, weight-normed convs) that historically are the painful ops for
ONNX Runtime Web. If these will not run on the web runtime, the whole approach
must change — so prove it before investing in the LM port.

### Gate findings (2026-06, torch 2.12 / onnxruntime 1.27)

- [x] Export **Kanade decoder** to ONNX (`conversion/export_kanade_decoder.py`).
      The transformer uses **complex-tensor RoPE**; the legacy exporter fails
      (`ScalarType ComplexFloat is an unexpected tensor scalar type`). The
      **TorchDynamo exporter** (`dynamo=True`, needs `onnxscript`) decomposes it
      into real ops and exports cleanly. We export a **slim decode-only wrapper**
      (quantizer + mel prenet/upsample/decoder/postnet; mel_length = 2·seq_len)
      that omits the SSL encoder. Also **disable attention dropout**: upstream's
      non-flash attention applies dropout at inference (no `if self.training`
      guard), which made decode nondeterministic and slightly degraded; with it
      off, the slim path matches `KanadeModel.decode` exactly and **ORT-CPU mel
      parity vs PyTorch is ≈ 1e-5 (PASS).**
- [x] Export **HiFT vocoder** to ONNX (`conversion/export_hift_vocoder.py` +
      `conversion/hift_onnx.py`). Stock `HiFTGenerator` is unexportable: it runs
      its source + synthesis through `torch.stft`/`torch.istft` on complex
      tensors (no `aten::istft`/`aten::complex` ONNX op; Vocos has the same
      blocker, so it is **not** a fallback), and its sine source uses
      `torch.rand`/`torch.randn`. **Fix:** `hift_onnx.patch_vocoder_for_onnx`
      swaps in **real-valued (i)STFT** (the transforms are tiny: n_fft=16,
      hop=4 — DFT matmul + ConvTranspose1d overlap-add; matches `torch.istft` to
      ~1e-8) and a **deterministic source** (zeroed phase init + noise). Also
      bypasses the `@torch.inference_mode()` `inference()` (replicating its body)
      and feeds a normal-tensor mel. **ORT-CPU wav parity vs PyTorch
      max|diff| ≈ 8.6e-7 (PASS).**
- [ ] Produce golden fixtures (`conversion/smoke_reference.py`): for a fixed
      sentence + built-in speaker, dump the generated audio-token ids, the
      decoder mel, and the final 24 kHz wav.
- [x] In the browser, load decoder + vocoder under onnxruntime-web, run
      decoder → vocoder on fixed inputs, and reproduce the PyTorch golden. Tested
      on **WASM and WebGPU** via a Vite harness (`web/phase0.html` +
      `web/src/phase0.ts`) driven by Playwright; golden from
      `conversion/gen_phase0_golden.py`. **Both PASS:**
      - WASM (opt=all): mel max|diff| ≈ 1.6e-5, wav ≈ 1.1e-7; ~1.5 s/utterance.
      - WebGPU (opt=basic): mel ≈ 6.7e-6, wav ≈ 5e-8; ~0.2 s/utterance.
      **WebGPU caveat:** with full graph optimization the EP builds a fused
      `SkipLayerNormalization` kernel that rejects our LayerNorm bias ("Beta
      must be 1D"). Using `graphOptimizationLevel: "basic"` on WebGPU avoids the
      fusion (and was fastest); `ort.ts` now does this. The ORT wasm CDN version
      must also match the installed package (`ort.env.versions.web`).

**Open follow-up (export sizes):** the decoder ONNX is ~365 MB — this is the
*genuine* decode path (mel_prenet ≈170 MB + mel_decoder ≈185 MB), NOT WavLM
bloat (the exporter already excludes the SSL encoder). The vocoder is ~83 MB.
Shrinking the browser download therefore needs weight quantization (fp16 ≈ half;
int8 less) rather than tree-shaking. Defer until after the browser gate.

**Success criterion:** golden sentence decodes to correct-sounding audio in the
browser on both backends.
**Status: GATE CLEARED.** Both stages export to ONNX and run under
onnxruntime-web on WASM and WebGPU, reproducing the PyTorch reference to float
precision. The in-browser numbers above are decoder+vocoder only (no LM yet);
they give an early real-time-factor signal (WebGPU ~0.2 s for ~2 s of audio).

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

- [x] Expose the pipeline as the provider-neutral `Engine` from
      [docs/INTERFACE.md](INTERFACE.md): streaming `synthesize() →
      AsyncIterable<PcmChunk>` + buffered `synthesizeToPcm()`, `listVoices()`,
      `AbortSignal` cancellation (`web/src/engine/`). Done and unit-tested.
- [ ] Backend toggle (WebGPU→WASM) plumbing, and model download + Cache/OPFS
      caching + warm-up (silent decode on load) with `onProgress`. Needs the
      converted model artifacts.
- [ ] Document how to host it in a Chrome MV3 **offscreen document** (not the
      service worker, not a content script), incl. the
      `cross_origin_embedder_policy` / `cross_origin_opener_policy` manifest keys
      needed for threaded WASM.

## Phase 3.5 — Drop-in API adapters

Make the engine a plug-in replacement for the two common hosted TTS APIs (see
[docs/INTERFACE.md](INTERFACE.md) for the request/voice/format mapping). Each
adapter is a thin layer over the engine: request mapping, voice-id mapping, and
a format encoder over the canonical 24 kHz PCM stream.

- [x] PCM + WAV + **MP3** encoders (`web/src/audio/format.ts`); MP3 via the
      pure-JS lamejs encoder (no native deps). Plus a Catmull-Rom **resampler**
      (`web/src/audio/resample.ts`) for non-24 kHz output. `opus`/`aac`/`flac`
      and `ulaw` remain `UnsupportedFormatError`.
- [x] **OpenAI adapter** — `audio.speech.create({ input, voice, speed,
      response_format })` shape, returning a web `Response`. Defaults to `mp3`
      (matching OpenAI); `mp3`/`wav` buffer, `pcm` streams.
- [x] **ElevenLabs adapter** — `textToSpeech.convert/stream(voiceId, { text,
      voice_settings, output_format })` shape. Defaults to `mp3_44100_128`
      (matching ElevenLabs); parses `mp3_<rate>_<kbps>` / `pcm_<rate>` and
      resamples as needed. `voice_settings.speed != 1` is rejected (pending
      time-stretch), other settings accepted but not applied.
- [x] Voice-id mapping (provider voice names ↔ built-in Danish speakers),
      overridable, validated against the engine catalog.

## Phase 4 — Validate

- [ ] Real-time factor on WebGPU vs WASM (Pico, then Nano).
- [ ] Quality A/B vs reference Python output.
- [ ] Memory footprint.

## Phase 5 — Voice cloning (optional)

Plapre clones a voice by deriving the same 128-dim speaker embedding that the
built-in speakers ship as — the rest of the pipeline is unchanged. In the
reference this is one call: `kanade.encode(wav).global_embedding`.

Mini de-risk (gate this phase, like Phase 0 gates the vocoder):

- [ ] Export the **clone encoder** to ONNX and reproduce one reference embedding
      in the browser (cosine similarity vs Python `extract_speaker`).

Only the *global* branch is needed (not the content/local encoder or FSQ):

- [ ] WavLM-base+ truncated to its conv frontend + first 2 transformer layers
      (`global_ssl_layers=(1,2)`), then the GlobalEncoder (ConvNext backbone +
      AttentiveStatsPool) → 128-dim embedding. Truncation shrinks size and the
      ONNX op surface.
- [ ] Runtime flow: decode reference audio → resample to 16 kHz → clone encoder
      → raw 128-dim → apply `speaker_proj` (128→hidden) → hidden; register a
      `Voice` carrying both. This is the **only** runtime use of `speaker_proj`
      (ship it as a small weight file or 1-op ONNX).
- [ ] Engine API: `cloneVoice(audio, sampleRate, opts?) → Voice`, usable in
      `synthesize()` exactly like a built-in voice. Cloning runs fully local —
      the reference audio never leaves the browser.
- [ ] Adapter mapping: ElevenLabs Instant Voice Cloning (`POST /v1/voices/add`)
      maps onto `cloneVoice()`; OpenAI has no cloning analogue (unsupported).

Encoder-specific risks: WavLM's gated relative-position bias and conv GroupNorm
are the likely ORT-Web op-support pain points; minimum clip length (~3–30 s);
resampler parity (16 kHz) against torchaudio.

## Deferred / known gaps (carried forward)

Things intentionally left out of the current implementation, tracked here so
they are not lost in commit messages:

- [ ] **Playback-speed (`rate`)** — omitted from `SynthesisRequest` until a
      pitch-preserving time-stretch (WSOLA/phase vocoder) exists; a naive
      resample would change pitch. Needed to map OpenAI `speed` and ElevenLabs
      `voice_settings.speed`. `docs/INTERFACE.md` still lists `rate` as the
      target shape.
- [x] **Inter-sentence silence** — opt-in via
      `EngineOptions.interSentenceSilenceSec` (default 0 = contiguous); the gap is
      inserted only between audio chunks, never leading or trailing.
- [ ] **Cancellation granularity** — `AbortSignal` is forwarded to the model but
      not yet honored *inside* `PlapreLM.generate`; the Phase 1 decode loop
      should check it so long sentences can be interrupted mid-generation (today
      the engine/stream only abort between sentences). The stream→engine
      cancellation wiring itself is done (`pcmStream` cancel handler).
- [ ] **opus / aac / flac encoders** — OpenAI can request these; they need a
      heavier (wasm) codec, so they currently raise `UnsupportedFormatError`.
      mp3/wav/pcm cover both providers' defaults.
- [ ] **Resampler quality** — `audio/resample.ts` is Catmull-Rom (not
      band-limited). Fine for speech PoC; upgrade to windowed-sinc if 24→44.1 kHz
      upsampling artifacts are audible.

## Open risks (validate early, cheap first)

1. **Vocoder ONNX/ORT-Web op support** — the make-or-break item (Phase 0).
   RESOLVED at the ONNX/ORT-CPU level: decoder via the dynamo exporter (parity
   ≈ 0.008); HiFT vocoder via the real-valued (i)STFT rewrite in `hift_onnx.py`
   (parity ≈ 9e-7). Still to confirm: in-browser wasm/webgpu execution.
2. **`inputs_embeds` generation** on the chosen runtime — folding the speaker
   vector into a precomputed hidden avoids needing a separate projection op.
3. **Pico-on-WASM real-time factor** — decides whether WebGPU is mandatory.
4. **Re-hosting weights** — upstream HF repos are gated; CC-BY permits
   redistributing our ONNX exports with attribution.
5. ~~**num2words(da) parity**~~ — RESOLVED: the port reproduces num2words(da)
   exactly, locked by a 2037-case golden fixture
   (`conversion/gen_num2words_fixtures.py`).
