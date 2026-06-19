// Errors adapters raise for requests the local engine cannot honor yet, kept
// separate so both adapters share one vocabulary (and callers can branch on it).

/** Raised when a requested playback speed is outside the provider's range. */
export class UnsupportedSpeedError extends Error {
  constructor(
    readonly speed: number,
    readonly min: number,
    readonly max: number,
  ) {
    super(`Playback speed ${speed} is out of range (must be ${min}–${max}).`);
    this.name = "UnsupportedSpeedError";
  }
}
