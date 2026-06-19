# Hosting in a Chrome MV3 extension (offscreen document)

This module is a plain ES module with no DOM dependency in its core, so it drops
into a Manifest V3 extension. The one non-obvious requirement is **where** to run
it: ONNX Runtime Web needs a normal document context (and, for the fast threaded
WASM backend, cross-origin isolation). The right home is an **offscreen
document**.

## Why an offscreen document (not the service worker or a content script)

| Context             | Problem                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| Service worker      | No DOM; can't use WebGPU canvas/WebAudio; killed after ~30 s idle, which loses warmed sessions and 100s of MB of weights. |
| Content script      | Runs in the page's origin/CSP; can't be cross-origin isolated; pollutes arbitrary sites; weights re-download per tab. |
| **Offscreen doc**   | A real, hidden document the extension owns: DOM + WebGPU + WebAudio, its own COOP/COEP, one long-lived instance shared by the whole extension. |

The service worker stays the coordinator: it creates the offscreen document on
demand and relays synthesis requests to it.

## Threaded WASM needs cross-origin isolation

The multi-threaded WASM backend uses `SharedArrayBuffer`, which requires the
document to be **cross-origin isolated**. For an offscreen document this is set
via the manifest:

```json
{
  "cross_origin_embedder_policy": { "value": "require-corp" },
  "cross_origin_opener_policy": { "value": "same-origin" }
}
```

Then `crossOriginIsolated === true` inside the offscreen document and ORT can
spin up WASM threads. If you skip this, force single-threaded WASM
(`ort.env.wasm.numThreads = 1`) — it still works, just slower. WebGPU does not
require isolation, so a WebGPU-only build can omit COOP/COEP.

### Package the ORT WASM binaries locally

The PoC loads the ORT `.wasm`/`.mjs` from a CDN (see `pipeline/ort.ts`). MV3's
CSP forbids remote script/wasm, so an extension build must ship them in the
bundle and point `ort.env.wasm.wasmPaths` at a packaged path
(`chrome.runtime.getURL("ort/")`). Add the ORT dist files (and the model
artifacts) to `web_accessible_resources`.

## Wiring

```jsonc
// manifest.json (excerpt)
{
  "manifest_version": 3,
  "permissions": ["offscreen"],
  "background": { "service_worker": "sw.js" },
  "cross_origin_embedder_policy": { "value": "require-corp" },
  "cross_origin_opener_policy":   { "value": "same-origin" },
  "web_accessible_resources": [
    { "resources": ["ort/*", "models/*"], "matches": ["<all_urls>"] }
  ]
}
```

```js
// sw.js — create the offscreen document once, then relay messages.
async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument?.()) return;
  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["AUDIO_PLAYBACK"], // closest MV3 reason for local TTS
    justification: "Run the local Danish TTS model (ONNX Runtime Web).",
  });
}
```

```js
// offscreen.js — load the engine once; synthesize on request.
import { loadPlapreEngine, encodeAudio } from "plapre-in-a-browser-web";

const enginePromise = loadPlapreEngine({
  backend: "webgpu", // falls back to wasm automatically
  cache: { onProgress: (loaded, total) => postProgress(loaded, total) },
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type !== "synthesize") return;
  (async () => {
    const engine = await enginePromise;
    const { samples } = await engine.synthesizeToPcm({ text: msg.text, voice: msg.voice });
    sendResponse({ wav: encodeAudio(samples, 24000, "wav") });
  })();
  return true; // async response
});
```

## Caching the weights

`loadPlapreEngine({ cache: { onProgress } })` fetches every artifact cache-first
through the Cache API (`pipeline/model-cache.ts`), so the multi-hundred-MB
weights download once and reload instantly/offline. Use `clearModelCache()` to
free space or force a refresh. The Cache API is available in offscreen documents.

## Warm-up

The first `synthesize()` compiles ORT kernels (especially on WebGPU). To avoid a
cold first request, kick off a tiny throwaway synthesis right after
`loadPlapreEngine()` resolves (e.g. one short word) and ignore the output; later
real requests then hit warm sessions.
