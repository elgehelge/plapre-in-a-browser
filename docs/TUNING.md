# Tuning generation (temperature & friends)

The TTS model is **autoregressive**: the LM emits a probability distribution
(logits) over the next audio token at every step, and we *sample* from it. The
sampling parameters — chiefly **temperature** — trade off between safe/flat and
varied/expressive prosody. This page explains where they live, how to set them,
and how to pick good values.

## TL;DR — is temperature baked into the model files?

**No.** The exported ONNX graphs (`lm.onnx`, `kanade_decoder.onnx`,
`hift_vocoder.onnx`) are temperature-agnostic. `lm.onnx` only ever outputs **raw
logits**; all sampling happens afterward in JavaScript in
[`web/src/pipeline/sampling.ts`](../web/src/pipeline/sampling.ts), outside the
graph. So the *same* artifacts serve every temperature — it's purely a **runtime
knob**, chosen by the developer at call time. You never re-export to change it.

## The knobs

From `GenerateOptions` (see [`docs/INTERFACE.md`](./INTERFACE.md)):

| Option        | Default | Meaning                                                        |
| ------------- | ------- | -------------------------------------------------------------- |
| `temperature` | `0.8`   | Softmax sharpness. `0` = greedy/deterministic; higher = more varied. |
| `topK`        | `50`    | Sample only from the K most-likely tokens (`0` = disabled).    |
| `topP`        | `0.95`  | Nucleus: smallest set whose cumulative prob ≥ P (`1` = disabled). |
| `seed`        | `0`     | Seeds the RNG; same seed + params ⇒ identical audio (reproducible). |
| `maxTokens`   | `500`   | Hard cap on generated audio tokens per sentence.               |

The defaults (`DEFAULT_GENERATION` in
[`web/src/engine/engine.ts`](../web/src/engine/engine.ts)) match upstream
Plapre's `inference.py`.

## Where developers set it

Three levels, each overriding the previous:

1. **Library default** — `temperature: 0.8`. Used when nothing is specified.
2. **Per engine**, at construction (applies to every request):

```ts
import { loadPlapreEngine } from "plapre-in-a-browser";

const engine = await loadPlapreEngine({
  generation: { temperature: 0.6 }, // partial override; the rest keep defaults
});
```

3. **Per request** — highest precedence, merged over the engine default:

```ts
await engine.synthesizeToPcm({
  text: "Hej, hvordan har du det?",
  voice: "ida",
  generation: { temperature: 0.6, seed: 7 },
});
```

### Adapters caveat (OpenAI / ElevenLabs)

The drop-in [`openai`](../web/src/adapters/openai.ts) and
[`elevenlabs`](../web/src/adapters/elevenlabs.ts) adapters mirror those wire
protocols, and **neither protocol has a temperature field**. So when you're
calling through an adapter, set the temperature once at **engine construction**
(the adapter forwards all synthesis to that engine). Per-request temperature is
only available through the native `Engine` API.

## Picking a value

### In the browser (live A/B)

The demo page ([`web/index.html`](../web/index.html)) has a **Temperature**
slider next to Speed. Run `npm run dev` in `web/`, load the engine, and sweep it
while re-synthesizing the same sentence. This is the fastest feedback loop and
exercises the exact runtime path users get.

### Offline sweep (reproducible, batch)

`conversion/tune_temperature.py` renders a long, multi-sentence text through the
full ONNX pipeline at several temperatures and writes one WAV per temperature to
`conversion/golden/` (git-ignored). It mirrors the browser engine exactly
(normalization, sentence splitting, per-sentence RNG reset), so a temperature
that sounds good there is the same value to pass as `generation.temperature`.

```bash
cd conversion
python tune_temperature.py --temps 0.5,0.6,0.7,0.8 --speaker tor
afplay golden/demo_t0.6.wav   # listen and compare
```

Put your own text in a file and point at it:

```bash
python tune_temperature.py --text-file my_text.txt --temps 0.55,0.65 --seed 7
```

### Rules of thumb

- **`0.6`–`0.8`** is the useful range for natural Danish prosody. `0.8` is the
  upstream default; `0.6` is calmer/steadier and often preferred for long-form.
- **`temperature: 0`** is greedy — fully deterministic, but tends to sound flat
  and can loop; mainly useful for parity tests, not production audio.
- Fix the **`seed`** when you want bit-identical output across runs (e.g. tests,
  caching, A/B comparisons). Vary it to get different takes at the same temp.

## See also

- [`conversion/verify.py`](../conversion/verify.py) (see `conversion/README.md`)
  — the separate "did the generated artifacts come out right?" integrity check.
  Tuning is about *how it sounds*; verification is about *whether the files are
  correct*.
- [`docs/INTERFACE.md`](./INTERFACE.md) — the full `GenerateOptions` contract.
