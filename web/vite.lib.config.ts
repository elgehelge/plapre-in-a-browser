import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

// Library build: emit a single ESM bundle of the public API (src/index.ts).
// Heavy runtime deps are left external so the consumer's bundler dedupes them
// and controls their versions. Type declarations are emitted separately by
// `tsc -p tsconfig.build.json` (see the build:lib script).
const external = ["onnxruntime-web", "@huggingface/transformers", "@breezystack/lamejs"];

export default defineConfig({
  build: {
    target: "es2022",
    sourcemap: true,
    emptyOutDir: true,
    // The library must not ship model artifacts; consumers fetch those at runtime
    // from `modelsBaseUrl`. Without this, Vite copies public/ (~1 GB) into dist.
    copyPublicDir: false,
    lib: {
      entry: fileURLToPath(new URL("src/index.ts", import.meta.url)),
      formats: ["es"],
      fileName: () => "index.js",
    },
    rollupOptions: {
      external: (id) => external.some((e) => id === e || id.startsWith(`${e}/`)),
    },
  },
});
