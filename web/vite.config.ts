import { defineConfig } from "vite";

// Demo-app config (dev server + the GitHub Pages build). The library build is
// separate (vite.lib.config.ts).
//
// Cross-origin isolation (COOP/COEP) is required for threaded WASM
// (SharedArrayBuffer) used by onnxruntime-web. The same headers are mirrored in
// the MV3 manifest via cross_origin_*_policy when this ships in an extension.
export default defineConfig(({ command }) => ({
  // Relative base so the built demo works under a project page
  // (https://<user>.github.io/plapre-in-a-browser/). Dev stays at root.
  base: command === "build" ? "./" : "/",
  build: {
    // The ~1 GB model artifacts in public/models are NOT bundled into the demo:
    // GitHub Pages can't host them. The deployed demo points `modelsBaseUrl` at a
    // hosted bundle; `npm run dev` serves public/models locally as usual.
    copyPublicDir: false,
  },
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
}));
