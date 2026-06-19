// Sample-rate conversion for the adapter boundary. The engine is fixed at
// NATIVE_SAMPLE_RATE (24 kHz); provider formats may ask for other rates
// (ElevenLabs mp3_44100_*, pcm_22050, …).
//
// Uses Catmull-Rom (cubic Hermite) interpolation: it reproduces constant and
// linear signals exactly and is smoother than linear interpolation for the
// upsampling the adapters need. It is NOT band-limited, so it is PoC-grade —
// adequate for speech, to be swapped for a windowed-sinc resampler if quality
// demands it. A no-op when the rates already match.

export function resample(pcm: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate <= 0 || toRate <= 0) throw new RangeError("sample rates must be positive");
  if (fromRate === toRate || pcm.length === 0) return pcm;

  const ratio = toRate / fromRate;
  const outLen = Math.max(1, Math.round(pcm.length * ratio));
  const out = new Float32Array(outLen);
  const at = (i: number): number => pcm[i < 0 ? 0 : i >= pcm.length ? pcm.length - 1 : i];

  for (let j = 0; j < outLen; j++) {
    const x = j / ratio;
    const i = Math.floor(x);
    const t = x - i;
    const p0 = at(i - 1);
    const p1 = at(i);
    const p2 = at(i + 1);
    const p3 = at(i + 2);
    const a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    const b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
    const c = -0.5 * p0 + 0.5 * p2;
    out[j] = ((a * t + b) * t + c) * t + p1;
  }
  return out;
}
