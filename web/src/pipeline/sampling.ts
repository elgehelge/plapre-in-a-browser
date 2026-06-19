// Autoregressive sampling utilities (temperature / top-k / top-p) over a logits
// row. Replaces vLLM's SamplingParams for the in-browser LM loop.
//
// A seedable RNG is provided so generation can be made deterministic for parity
// testing against golden token ids.

export interface SamplingParams {
  temperature: number; // 0 => greedy (argmax)
  topK: number; // 0 => disabled
  topP: number; // 1 => disabled
}

/** Deterministic PRNG (mulberry32) in [0, 1). */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function argmax(logits: Float32Array | number[]): number {
  let best = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > bestVal) {
      bestVal = logits[i];
      best = i;
    }
  }
  return best;
}

function softmaxInPlace(values: number[]): void {
  let max = -Infinity;
  for (const v of values) if (v > max) max = v;
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    values[i] = Math.exp(values[i] - max);
    sum += values[i];
  }
  for (let i = 0; i < values.length; i++) values[i] /= sum;
}

/** Sample one token id from a logits row. */
export function sample(
  logits: Float32Array | number[],
  params: SamplingParams,
  rng: () => number,
): number {
  if (params.temperature <= 0) return argmax(logits);

  // index -> logit, scaled by temperature
  let candidates: { id: number; logit: number }[] = [];
  for (let i = 0; i < logits.length; i++) {
    candidates.push({ id: i, logit: logits[i] / params.temperature });
  }
  candidates.sort((a, b) => b.logit - a.logit);

  if (params.topK > 0 && params.topK < candidates.length) {
    candidates = candidates.slice(0, params.topK);
  }

  const probs = candidates.map((c) => c.logit);
  softmaxInPlace(probs);

  // top-p (nucleus): keep the smallest prefix whose cumulative prob >= topP.
  if (params.topP < 1) {
    let cum = 0;
    let cutoff = probs.length;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (cum >= params.topP) {
        cutoff = i + 1;
        break;
      }
    }
    candidates = candidates.slice(0, cutoff);
    const kept = probs.slice(0, cutoff);
    const total = kept.reduce((s, p) => s + p, 0);
    for (let i = 0; i < kept.length; i++) kept[i] /= total;
    return pick(candidates, kept, rng());
  }

  return pick(candidates, probs, rng());
}

function pick(
  candidates: { id: number; logit: number }[],
  probs: number[],
  r: number,
): number {
  let cum = 0;
  for (let i = 0; i < candidates.length; i++) {
    cum += probs[i];
    if (r <= cum) return candidates[i].id;
  }
  return candidates[candidates.length - 1].id;
}
