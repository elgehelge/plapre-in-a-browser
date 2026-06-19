import { defineConfig } from "vite";

// Cross-origin isolation (COOP/COEP) is required for threaded WASM
// (SharedArrayBuffer) used by onnxruntime-web. The same headers are mirrored in
// the MV3 manifest via cross_origin_*_policy when this ships in an extension.
export default defineConfig({
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  // onnxruntime-web ships prebuilt wasm; let Vite leave it as-is.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
