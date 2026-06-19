// Pitch-preserving time-stretch (WSOLA) for the `rate`/`speed` knob.
//
// A naive resample changes duration AND pitch (chipmunk effect). WSOLA
// (Waveform Similarity Overlap-Add) changes duration only: it overlap-adds
// windowed analysis frames at a synthesis hop, sliding each analysis frame
// within a small search window to the offset that best correlates with the
// expected continuation, so the waveform stays phase-coherent and pitch is
// preserved.
//
// speed > 1 → faster/shorter; speed < 1 → slower/longer. speed === 1 is a no-op.

const FRAME = 1024; // ~43 ms at 24 kHz — long enough to span a pitch period
const SYNTH_HOP = FRAME >> 1; // 50% overlap
const SEARCH = 256; // ± samples WSOLA may slide an analysis frame

function hann(n: number): Float32Array {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

/** Best offset in [-SEARCH, SEARCH] maximizing correlation with `ref`. */
function bestOffset(x: Float32Array, center: number, ref: Float32Array): number {
  let best = 0;
  let bestScore = -Infinity;
  for (let d = -SEARCH; d <= SEARCH; d++) {
    const start = center + d;
    if (start < 0 || start + ref.length > x.length) continue;
    let score = 0;
    for (let i = 0; i < ref.length; i += 4) score += x[start + i] * ref[i]; // strided dot
    if (score > bestScore) {
      bestScore = score;
      best = d;
    }
  }
  return best;
}

/**
 * Time-stretch mono PCM by `speed` (duration scales by 1/speed), preserving
 * pitch. Falls back to a copy for speed ≈ 1 or very short input.
 */
export function timeStretch(pcm: Float32Array, speed: number): Float32Array {
  if (!(speed > 0)) throw new Error(`speed must be > 0, got ${speed}`);
  if (Math.abs(speed - 1) < 1e-3 || pcm.length < FRAME * 2) return pcm.slice();

  const window = hann(FRAME);
  const analysisHop = SYNTH_HOP * speed;
  const outLen = Math.ceil(pcm.length / speed) + FRAME;
  const out = new Float32Array(outLen);
  const norm = new Float32Array(outLen); // accumulated window energy for normalization

  // The "expected continuation" we correlate against: the tail of the previous
  // analysis frame advanced by the synthesis hop.
  let expected = pcm.subarray(0, FRAME);
  let analysisPos = 0;
  let outPos = 0;

  while (analysisPos + FRAME + SEARCH < pcm.length && outPos + FRAME < outLen) {
    const center = Math.round(analysisPos);
    const delta = outPos === 0 ? 0 : bestOffset(pcm, center, expected.subarray(0, FRAME));
    const start = Math.max(0, Math.min(center + delta, pcm.length - FRAME));

    for (let i = 0; i < FRAME; i++) {
      const w = window[i];
      out[outPos + i] += pcm[start + i] * w;
      norm[outPos + i] += w;
    }

    // Next expected continuation = this frame shifted by the synthesis hop.
    const contStart = start + SYNTH_HOP;
    expected =
      contStart + FRAME <= pcm.length
        ? pcm.subarray(contStart, contStart + FRAME)
        : pcm.subarray(pcm.length - FRAME);

    outPos += SYNTH_HOP;
    analysisPos += analysisHop;
  }

  const end = outPos + FRAME;
  const result = new Float32Array(end);
  for (let i = 0; i < end; i++) result[i] = norm[i] > 1e-6 ? out[i] / norm[i] : 0;
  return result;
}
