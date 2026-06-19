import { describe, it, expect, vi, afterEach } from "vitest";
import { fetchCached, clearModelCache } from "./model-cache.js";

function bodyResponse(bytes: Uint8Array): Response {
  // A Response whose body streams in two chunks, with a content-length header.
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes.slice(0, Math.ceil(bytes.length / 2)));
      controller.enqueue(bytes.slice(Math.ceil(bytes.length / 2)));
      controller.close();
    },
  });
  return new Response(stream, { headers: { "content-length": String(bytes.length) } });
}

afterEach(() => {
  vi.restoreAllMocks();
  delete (globalThis as { caches?: unknown }).caches;
});

describe("fetchCached", () => {
  it("falls back to a plain streamed fetch when no Cache API is present", async () => {
    const bytes = new Uint8Array([1, 2, 3, 4, 5, 6]);
    vi.stubGlobal("fetch", vi.fn(async () => bodyResponse(bytes)));

    const progress: number[] = [];
    const buf = await fetchCached("/models/x.onnx", { onProgress: (l) => progress.push(l) });

    expect(new Uint8Array(buf)).toEqual(bytes);
    expect(progress.at(-1)).toBe(bytes.length); // progress reaches total
    expect(progress.length).toBeGreaterThan(1); // streamed in chunks
  });

  it("stores on a cache miss and serves from cache on the next call", async () => {
    const bytes = new Uint8Array([9, 8, 7]);
    const store = new Map<string, Response>();
    const fakeCache = {
      match: async (k: string) => store.get(k),
      put: async (k: string, v: Response) => void store.set(k, v),
    };
    vi.stubGlobal("caches", { open: async () => fakeCache, delete: async () => true });
    const fetchMock = vi.fn(async () => bodyResponse(bytes));
    vi.stubGlobal("fetch", fetchMock);

    const first = await fetchCached("/models/y.onnx");
    expect(new Uint8Array(first)).toEqual(bytes);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(store.has("/models/y.onnx")).toBe(true);

    const second = await fetchCached("/models/y.onnx");
    expect(new Uint8Array(second)).toEqual(bytes);
    expect(fetchMock).toHaveBeenCalledTimes(1); // served from cache, no new fetch
  });

  it("throws on a non-ok response", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("nope", { status: 404 })));
    await expect(fetchCached("/models/missing.onnx")).rejects.toThrow(/404/);
  });

  it("clearModelCache returns false without a Cache API", async () => {
    expect(await clearModelCache()).toBe(false);
  });
});
