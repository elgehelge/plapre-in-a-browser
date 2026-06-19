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

**Status: runtime PROVEN; gated export written + UNVALIDATED.** The browser
decode loop is implemented and validated; the real export of the gated weights is
the one human-gated step (see "Gated blocker" below).

- [x] JS: BPE tokenize via the HF `tokenizer.json`; build prompt
      `[<text>] + ids + [<audio>]` (`web/src/pipeline/tokenizer.ts`).
- [x] JS: autoregressive loop with KV cache + temperature/top-k/top-p sampling,
      stop at EOS / `max_tokens`, collect audio-token ids
      (`web/src/pipeline/lm.ts`). The loop talks to the model through the narrow
      `LmGraph` seam (prefill prepends the speaker hidden as `prefix_embeds`,
      decode feeds one token; `present.*` → `past_key_values.*`).
- [x] **Validated without gated weights**: 6 unit tests for the loop logic, plus
      a genuine causal-attention **toy LM** (`conversion/gen_toy_lm.py`) that
      follows the exact export contract — ORT-CPU greedy matches the torch golden,
      and the real `OrtLmGraph`+`PlapreLM` reproduce 30/30 golden ids in-browser
      on **WASM and WebGPU** (`web/phase1.html`).
- [x] Export **Plapre Pico LM** to ONNX (`input_ids` + `prefix_embeds` + KV cache
      → `logits` + `present`) via a custom wrapper (`conversion/export_lm.py`)
      that embeds ids internally and prepends the speaker row; emits `meta.json`
      with `{numLayers,kvHeads,headDim,hidden}`. Written to the proven contract;
      runs once authenticated. (optimum-cli can't express `inputs_embeds` + the
      speaker prepend, hence the wrapper.)
- [x] Precompute speaker hidden vectors offline: `speaker_proj @ speakers[name]`
      → `speakers.json` (`conversion/precompute_speakers.py`); robust to the
      repo's speaker/proj shapes. Removes the projection from the runtime.
- [x] `conversion/fetch_tokenizer.py` copies `tokenizer.json`/`config.json`.
- [x] `conversion/smoke_reference.py` is a self-contained greedy torch reference
      that mirrors `lm.ts` exactly (no vLLM), producing `tokens.json` /
      `kanade.json` / `mel.npy` / `reference.wav` as the cross-language oracle.
- [ ] **Parity-check generated token ids against golden** — blocked: needs the
      gated weights to produce the golden ids (loop + reference are ready).

### Gated blocker (the one human step)

`syvai/plapre-pico` is `gated: auto`. Downloading the weights needs a license
acceptance + login that an agent must not do on the user's behalf. Everything
that does not need the weights is complete and tested; the gated scripts print a
single actionable message (`conversion/_gated.py`). To finish Phase 1/4 e2e:

```
huggingface-cli login            # after clicking “Agree” on the model page
cd conversion && source .venv/bin/activate
python fetch_tokenizer.py && python export_lm.py && python precompute_speakers.py
python smoke_reference.py        # writes golden token ids for the parity check
```

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
- [x] Backend toggle (WebGPU→WASM) plumbing (`pickBackend`), and model download +
      **Cache API caching** + `onProgress`, opt-in via
      `loadPlapreEngine({ cache: { onProgress } })` (`pipeline/model-cache.ts`,
      wired through `createSession` and all loaders; cache-first fetch unit-tested
      with hit/miss/fallback). `clearModelCache()` frees it. Warm-up guidance is
      documented (a tiny throwaway synthesis after load) rather than baked in, so
      callers control when the kernels compile.
- [x] Documented hosting in a Chrome MV3 **offscreen document**
      ([docs/EXTENSION.md](EXTENSION.md)): why offscreen (not the service
      worker/content script), the `cross_origin_embedder_policy` /
      `cross_origin_opener_policy` keys for threaded WASM, packaging the ORT wasm
      locally (MV3 CSP), service-worker↔offscreen wiring, and caching.

## Phase 3.5 — Drop-in API adapters

Make the engine a plug-in replacement for the two common hosted TTS APIs (see
[docs/INTERFACE.md](INTERFACE.md) for the request/voice/format mapping). Each
adapter is a thin layer over the engine: request mapping, voice-id mapping, and
a format encoder over the canonical 24 kHz PCM stream.

- [x] PCM + WAV + **MP3** encoders (`web/src/audio/format.ts`); MP3 via the
      pure-JS lamejs encoder (no native deps). Plus a band-limited windowed-sinc
      **resampler** (`web/src/audio/resample.ts`) for non-24 kHz output.
      `opus`/`aac`/`flac`
      and `ulaw` remain `UnsupportedFormatError`.
- [x] **OpenAI adapter** — `audio.speech.create({ input, voice, speed,
      response_format })` shape, returning a web `Response`. Defaults to `mp3`
      (matching OpenAI); `mp3`/`wav` buffer, `pcm` streams.
- [x] **ElevenLabs adapter** — `textToSpeech.convert/stream(voiceId, { text,
      voice_settings, output_format })` shape. Defaults to `mp3_44100_128`
      (matching ElevenLabs); parses `mp3_<rate>_<kbps>` / `pcm_<rate>` and
      resamples as needed. `voice_settings.speed` maps to the engine `rate`
      (pitch-preserving, clamped to ElevenLabs' 0.7–1.2); other settings
      accepted but not applied.
- [x] Voice-id mapping (provider voice names ↔ built-in Danish speakers),
      overridable, validated against the engine catalog.

## Phase 4 — Validate

A reusable benchmark harness (`web/bench.html` + `src/bench.ts`) measures load
time, per-iteration latency, and real-time factor (RTF = audio seconds / wall
seconds) for whatever stages are present, on a selectable backend. It skips the
gated LM until its artifacts land.

- [x] **Real-time factor (Stage 3 + clone), measured in-browser** for 50 audio
      tokens → ~2 s of audio, 8 iters, on this machine (Apple GPU):

      | stage                    | WASM mean | WebGPU mean |
      | ------------------------ | --------- | ----------- |
      | decoder                  | 86 ms     | 30 ms       |
      | vocoder                  | 514 ms    | 77 ms       |
      | decoder+vocoder (chain)  | 600 ms    | 107 ms      |
      | **chain RTF**            | **3.3×**  | **18.6×**   |
      | clone encoder (2 s ref)  | 122 ms    | 35 ms       |

      Stage 3 is comfortably faster than real time on both backends; WebGPU is
      ~5–6× faster. The vocoder dominates on WASM.
- [ ] **LM + full-pipeline RTF** — blocked on the gated LM weights. The decode
      loop is O(tokens) sequential ORT calls (~25 tokens/s of audio); whether
      Pico hits real time on WASM is the remaining open RTF question (WebGPU is
      expected to). Re-run `bench.html` once `lm/model.onnx` is exported; the
      harness already structures the LM slot.
- [x] **Memory footprint** (artifact sizes, the dominant cost): decoder
      ≈ 367 MB (1.6 MB graph + 365 MB external data), vocoder ≈ 85 MB, clone
      encoder 174 MB (WavLM), toy LM 50 KB. Pico LM fp32 lands next to these once
      exported (q8 ≈ 121 MB per upstream). Measured JS heap during Stage-3 bench
      ≈ 190 MB. WebGPU also holds weights in GPU memory.
- [ ] **Quality A/B vs reference Python output** — blocked on the gated LM. The
      oracle is ready: `conversion/smoke_reference.py` writes `reference.wav` +
      golden token ids; compare the browser output once authenticated.

## Phase 5 — Voice cloning

**Status: GATE CLEARED + runtime wired.** The clone encoder exports and runs
under onnxruntime-web on WASM and WebGPU, reproducing the torch/ORT-CPU embedding
(cosine 1.000000 on both). This phase is fully public (no gated weights); only
`speaker_proj.json` (the 128→hidden projection) is gated, and it is emitted by
precompute_speakers.py.

Plapre clones a voice by deriving the same 128-dim speaker embedding that the
built-in speakers ship as — the rest of the pipeline is unchanged.

- [x] Export the **clone encoder** to ONNX (`conversion/export_clone_encoder.py`)
      and reproduce a reference embedding in the browser (cosine vs torch). The
      global SSL branch turned out to be `global_ssl_layers=(1,2,3,4)` averaged
      (not (1,2)); `config.sample_rate=24000`. ORT-CPU vs torch cosine 1.0; the
      slim-vs-`encode()` cosine is 0.997 (the small waveform-padding difference —
      see deferred). Legacy exporter (dynamo trips on WavLM control flow).
- [x] WavLM-base+ frontend (layers 1–4 averaged) → GlobalEncoder (ConvNext +
      AttentiveStatsPool) → 128-dim. The SSL 24→16 kHz resample stays INSIDE the
      graph (exact torchaudio sinc) so the browser only resamples to 24 kHz.
- [x] Runtime flow (`web/src/pipeline/clone.ts`): reference audio → resample →
      `clone_encoder.onnx` → raw 128-dim → `speaker_proj` (128→hidden, applied in
      JS from `speaker_proj.json`) → hidden; register as a `SpeakerData`. The
      projection is a single Linear shipped as JSON and applied in JS (no extra
      session), so its math is unit-tested.
- [x] Engine API: `Engine.cloneVoice(audio, sampleRate, opts?) → Voice` +
      `canCloneVoice()`; `PlapreSpeechModel` implements `VoiceCloner` and
      registers the cloned voice into its catalog so `synthesize()` uses it like
      a built-in. Fully local — reference audio never leaves the browser.
- [x] Adapter mapping: ElevenLabs Instant Voice Cloning → `createElevenLabsVoices`
      / `createElevenLabsClient` (`voices.add({ name, files }) → cloneVoice`),
      with a pluggable `AudioDecoder` (WebAudio default). OpenAI has no cloning
      analogue (unsupported).

Risks retired: WavLM under ORT-Web (the op-support worry) runs clean on both
backends. Remaining nuance tracked in deferred: waveform-padding parity (cosine
0.997 vs the padded reference) and minimum clip length guidance.

## Deferred / known gaps (carried forward)

Things intentionally left out of the current implementation, tracked here so
they are not lost in commit messages:

- [x] **Playback-speed (`rate`)** — `SynthesisRequest.rate` applies a
      pitch-preserving WSOLA time-stretch (`web/src/audio/time-stretch.ts`,
      per sentence so no cross-sentence continuity is needed). OpenAI `speed`
      (0.25–4) and ElevenLabs `voice_settings.speed` (0.7–1.2) map onto it;
      out-of-range values raise `UnsupportedSpeedError`. Tested for duration
      scaling + pitch preservation.
- [x] **Inter-sentence silence** — opt-in via
      `EngineOptions.interSentenceSilenceSec` (default 0 = contiguous); the gap is
      inserted only between audio chunks, never leading or trailing.
- [x] **Cancellation granularity** — `PlapreLM.generate` now takes the
      `AbortSignal` and calls `throwIfAborted()` before prefill and on every
      decode step, so long sentences interrupt mid-generation (not only between
      sentences). The stream→engine wiring (`pcmStream` cancel handler) and the
      engine→model forwarding were already in place.
- [ ] **opus / aac / flac encoders** — OpenAI can request these; they need a
      heavier (wasm) codec, so they currently raise `UnsupportedFormatError`.
      mp3/wav/pcm cover both providers' defaults.
- [x] **Resampler quality** — `audio/resample.ts` is now a band-limited
      windowed-sinc (Lanczos, 8 lobes) with the cutoff lowered to the output
      Nyquist on downsample (anti-aliasing) and normalized weights (exact DC).
      Tested: constant/linear reproduction + above-Nyquist tones suppressed on
      downsample.
- [~] **Clone waveform-padding parity** — the clone encoder skips the reference
      audio's symmetric padding that `KanadeModel.encode()` applies, costing
      ~0.003 cosine (slim-vs-encode 0.997). The attentive-stats pool makes this
      negligible; replicate `_calculate_waveform_padding` in the wrapper if clone
      fidelity ever needs it. (Clip-length guidance is now enforced:
      `embedSpeaker` rejects clips < `MIN_CLONE_SECONDS` (1 s) and documents the
      recommended 3–30 s range via `RECOMMENDED_CLONE_SECONDS`.)

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
