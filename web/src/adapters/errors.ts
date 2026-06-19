// Errors adapters raise for requests the local engine cannot honor yet, kept
// separate so both adapters share one vocabulary (and callers can branch on it).

export class UnsupportedSpeedError extends Error {
  constructor(readonly speed: number) {
    super(
      `Playback speed ${speed} is not supported yet (only 1). Speed needs a ` +
        `pitch-preserving time-stretch; see docs/PLAN.md "Deferred / known gaps".`,
    );
    this.name = "UnsupportedSpeedError";
  }
}
