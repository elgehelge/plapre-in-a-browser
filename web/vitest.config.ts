import { defineConfig } from "vitest/config";

// Tests cover pure, environment-agnostic logic (text normalization, sampling,
// audio encoding, the provider-neutral engine orchestration). They run in the
// node environment; stages that require a browser runtime (onnxruntime-web,
// WebGPU) are exercised behind injected seams, not in unit tests.
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
