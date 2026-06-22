// Cache-first model fetching with progress, for the large ONNX artifacts.
//
// The browser HTTP cache already helps, but an offline-capable extension wants
// explicit control. This uses the Cache API (available in windows, workers, and
// MV3 offscreen documents) keyed by URL, falling back to a plain fetch where
// `caches` is unavailable (e.g. Node tests). It streams the body so callers can
// render a download progress bar for the multi-hundred-MB weights.

// `url` identifies which artifact the bytes belong to, so a UI can aggregate
// progress across the several files that download concurrently. Older callers
// that only take (loaded, total) keep working.
export type ProgressFn = (loaded: number, total: number, url?: string) => void;

export interface FetchCachedOptions {
  cacheName?: string;
  onProgress?: ProgressFn;
  signal?: AbortSignal;
}

const DEFAULT_CACHE = "plapre-models-v1";

function cacheStorage(): CacheStorage | undefined {
  return (globalThis as { caches?: CacheStorage }).caches;
}

async function readWithProgress(res: Response, onProgress?: ProgressFn): Promise<ArrayBuffer> {
  const total = Number(res.headers.get("content-length") ?? 0);
  if (!onProgress || !res.body) {
    const buf = await res.arrayBuffer();
    onProgress?.(buf.byteLength, total || buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total || loaded);
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  return out.buffer;
}

/**
 * Fetch `url` as bytes, cache-first. On a miss the response is streamed (with
 * progress) and returned; the downloaded bytes are then stored in the Cache for
 * instant repeat loads. Without a Cache API it falls back to a streamed fetch.
 *
 * The body is read exactly once and cached from the assembled bytes — never via
 * `response.clone()` + `await cache.put(clone)` before reading. Teeing a
 * multi-hundred-MB response and only consuming one branch makes the browser
 * buffer the entire unread branch in memory (and reports no progress until the
 * whole file lands); with the large LM + decoder + vocoder downloading at once
 * that stalls badly. Streaming once keeps memory bounded and the progress bar
 * live throughout the download.
 */
export async function fetchCached(url: string, opts: FetchCachedOptions = {}): Promise<ArrayBuffer> {
  const report: ProgressFn | undefined = opts.onProgress
    ? (loaded, total) => opts.onProgress!(loaded, total, url)
    : undefined;

  const storage = cacheStorage();
  const cache = storage ? await storage.open(opts.cacheName ?? DEFAULT_CACHE) : undefined;

  if (cache) {
    const hit = await cache.match(url);
    if (hit) return readWithProgress(hit, report);
  }

  const res = await fetch(url, { signal: opts.signal });
  if (!res.ok) throw new Error(`fetch ${url} failed: ${res.status}`);
  const contentType = res.headers.get("content-type") ?? "application/octet-stream";
  const buf = await readWithProgress(res, report);

  if (cache) {
    // Store from the bytes we already have (no second download, no tee). A
    // failed write is non-fatal — the caller still gets the bytes.
    try {
      await cache.put(
        url,
        new Response(buf, {
          headers: { "content-type": contentType, "content-length": String(buf.byteLength) },
        }),
      );
    } catch {
      /* cache write is best-effort (e.g. storage quota); ignore. */
    }
  }
  return buf;
}

/** Remove the model cache (e.g. to free space or force re-download). */
export async function clearModelCache(cacheName: string = DEFAULT_CACHE): Promise<boolean> {
  const storage = cacheStorage();
  return storage ? storage.delete(cacheName) : false;
}
