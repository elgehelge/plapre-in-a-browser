// Sample-rate conversion for the adapter boundary. The engine is fixed at
// NATIVE_SAMPLE_RATE (24 kHz); provider formats may ask for other rates
// (ElevenLabs mp3_44100_*, pcm_22050, …).
//
// Band-limited windowed-sinc (Lanczos) resampling. Unlike a polynomial
// (Catmull-Rom/linear) interpolator, a windowed sinc approximates the ideal
// brick-wall reconstruction filter, so it does not fold high-frequency energy
// back into the audible band. When downsampling we lower the kernel's cutoff to
// the output Nyquist (anti-aliasing); when upsampling the cutoff stays at the
// input Nyquist. Weights are normalized so a DC/constant signal is preserved
// exactly. A no-op when the rates already match.

const LOBES = 8; // Lanczos half-width in (cutoff-scaled) input samples

function sinc(x: number): number {
  if (x === 0) return 1;
  const p = Math.PI * x;
  return Math.sin(p) / p;
}

/** Lanczos kernel L(t) = sinc(t)·sinc(t/a) for |t| < a, else 0. */
function lanczos(t: number): number {
  if (t <= -LOBES || t >= LOBES) return 0;
  return sinc(t) * sinc(t / LOBES);
}

export function resample(pcm: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate <= 0 || toRate <= 0) throw new RangeError("sample rates must be positive");
  if (fromRate === toRate || pcm.length === 0) return pcm;

  const ratio = toRate / fromRate;
  const outLen = Math.max(1, Math.round(pcm.length * ratio));
  const out = new Float32Array(outLen);
  const n = pcm.length;
  const at = (i: number): number => pcm[i < 0 ? 0 : i >= n ? n - 1 : i]; // edge-clamp

  // Cutoff (in input-sample cycles): full Nyquist when upsampling, lowered to
  // the output Nyquist when downsampling to suppress aliasing.
  const cutoff = Math.min(1, ratio);
  const half = LOBES / cutoff; // kernel half-width in input samples

  for (let j = 0; j < outLen; j++) {
    const x = j / ratio; // position in input-sample space
    const left = Math.ceil(x - half);
    const right = Math.floor(x + half);

    let acc = 0;
    let norm = 0;
    for (let i = left; i <= right; i++) {
      const w = lanczos((x - i) * cutoff);
      if (w === 0) continue;
      acc += at(i) * w;
      norm += w;
    }
    out[j] = norm !== 0 ? acc / norm : at(Math.round(x));
  }
  return out;
}
