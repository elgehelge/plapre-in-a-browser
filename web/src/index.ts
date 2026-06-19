// Public API surface of the in-browser Danish TTS module.
//
// Consumers should import from here. Internals (the ONNX stages, tokenizer,
// sampling, normalization) are intentionally not all re-exported — they sit
// behind the Engine and SpeechModel seams.

// Provider-neutral engine + domain types.
export * from "./engine/engine.js";
export * from "./engine/voice.js";
export * from "./engine/speech-model.js";

// Plapre as the concrete SpeechModel + the loader that returns an Engine.
export * from "./pipeline/plapre.js";

// Model caching (Cache API) for offline / instant reloads of the large weights.
export { fetchCached, clearModelCache, type ProgressFn } from "./pipeline/model-cache.js";

// Audio serialization + sample-rate conversion + decoding (for cloning input).
export * from "./audio/format.js";
export * from "./audio/resample.js";
export * from "./audio/decode.js";

// Drop-in API adapters.
export * from "./adapters/openai.js";
export * from "./adapters/elevenlabs.js";
export * from "./adapters/voice-map.js";
export * from "./adapters/errors.js";
