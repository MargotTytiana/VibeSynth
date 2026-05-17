// =============================================================================
// File: frontend/src/WasmLoader.ts
// Purpose: Fetches, validates, and caches compiled WASM modules from the
//          backend registry endpoint. Provides a module registry keyed by
//          module_id so that repeated requests for the same algorithm do not
//          re-fetch the binary. Also exposes utility functions for checking
//          WASM magic bytes, reading export names, and measuring instantiation
//          time — used by AudioEngine and the experiments notebooks via the
//          browser DevTools console.
// =============================================================================

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface WasmModuleEntry {
  moduleId:        string;
  bytes:           ArrayBuffer;
  byteLength:      number;
  fetchTimeMs:     number;
  instantiateTimeMs: number;
  exports:         string[];
  cachedAt:        number;   // Date.now() timestamp
}

export interface WasmLoadResult {
  success:         boolean;
  entry:           WasmModuleEntry | null;
  error:           string | null;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API_BASE      = "/api/v1";
const WASM_MAGIC    = new Uint8Array([0x00, 0x61, 0x73, 0x6d]); // \0asm
const WASM_VERSION  = new Uint8Array([0x01, 0x00, 0x00, 0x00]); // version 1
const MAX_CACHE_ENTRIES = 20;  // maximum modules to keep in the client-side cache

// ---------------------------------------------------------------------------
// Client-side module registry
// ---------------------------------------------------------------------------

const _registry: Map<string, WasmModuleEntry> = new Map();

/**
 * Return all currently cached module IDs.
 */
export function listCachedModules(): string[] {
  return Array.from(_registry.keys());
}

/**
 * Return a cached module entry by ID, or null if not in cache.
 */
export function getCachedModule(moduleId: string): WasmModuleEntry | null {
  return _registry.get(moduleId) ?? null;
}

/**
 * Evict the oldest cached module if the registry is at capacity.
 */
function _evictIfFull(): void {
  if (_registry.size >= MAX_CACHE_ENTRIES) {
    // Evict the entry with the smallest cachedAt timestamp (FIFO)
    let oldestId   = "";
    let oldestTime = Infinity;
    for (const [id, entry] of _registry) {
      if (entry.cachedAt < oldestTime) {
        oldestTime = entry.cachedAt;
        oldestId   = id;
      }
    }
    if (oldestId) {
      _registry.delete(oldestId);
      console.debug(`[WasmLoader] evicted module '${oldestId}' from client cache.`);
    }
  }
}

// ---------------------------------------------------------------------------
// Magic-byte validation
// ---------------------------------------------------------------------------

/**
 * Verify that an ArrayBuffer starts with the WASM magic number and version.
 *
 * @param buffer  Raw bytes to check.
 * @returns       True if the buffer is a valid (non-empty) WASM binary header.
 */
export function validateWasmHeader(buffer: ArrayBuffer): boolean {
  if (buffer.byteLength < 8) return false;
  const view = new Uint8Array(buffer, 0, 8);
  for (let i = 0; i < 4; i++) {
    if (view[i] !== WASM_MAGIC[i]) return false;
  }
  for (let i = 0; i < 4; i++) {
    if (view[i + 4] !== WASM_VERSION[i]) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Export name reader
// ---------------------------------------------------------------------------

/**
 * Compile a WASM binary and return the names of all its exports.
 * Used for diagnostics and to verify the AudioWorklet processor name is present.
 *
 * @param buffer  Raw WASM bytes.
 * @returns       Array of export name strings, or [] if compilation fails.
 */
async function _readExports(buffer: ArrayBuffer): Promise<string[]> {
  try {
    const compiled = await WebAssembly.compile(buffer);
    return WebAssembly.Module.exports(compiled).map((e) => e.name);
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Core loader
// ---------------------------------------------------------------------------

/**
 * Fetch a compiled WASM module from the backend registry.
 *
 * Flow:
 *   1. Check the client-side in-memory cache.
 *   2. On cache miss, fetch from GET /api/v1/wasm/<module_id>.
 *   3. Validate the WASM magic-byte header.
 *   4. Instantiate once to measure instantiation time and read exports.
 *   5. Store in the client-side registry and return.
 *
 * @param moduleId  The module ID returned by the backend generation endpoint.
 * @returns         WasmLoadResult — check result.success before using result.entry.
 */
export async function loadModule(moduleId: string): Promise<WasmLoadResult> {
  // 1. Client-side cache hit
  const cached = _registry.get(moduleId);
  if (cached) {
    console.debug(`[WasmLoader] cache hit for '${moduleId}'.`);
    return { success: true, entry: cached, error: null };
  }

  // 2. Fetch from backend
  const fetchStart = performance.now();
  let buffer: ArrayBuffer;

  try {
    const response = await fetch(`${API_BASE}/wasm/${moduleId}`, {
      method:  "GET",
      headers: { Accept: "application/wasm" },
    });

    if (!response.ok) {
      return {
        success: false,
        entry:   null,
        error:   `Backend returned HTTP ${response.status} for module '${moduleId}'.`,
      };
    }

    buffer = await response.arrayBuffer();
  } catch (err) {
    return {
      success: false,
      entry:   null,
      error:   `Network error fetching module '${moduleId}': ${err}`,
    };
  }

  const fetchTimeMs = Math.round(performance.now() - fetchStart);

  // 3. Magic-byte validation
  if (!validateWasmHeader(buffer)) {
    return {
      success: false,
      entry:   null,
      error:   `Module '${moduleId}' failed WASM header validation. ` +
               `Expected \\x00asm version 1. Byte length: ${buffer.byteLength}.`,
    };
  }

  // 4. Instantiate to measure time and read exports
  const instantiateStart = performance.now();
  let exports: string[] = [];

  try {
    exports = await _readExports(buffer);
  } catch (err) {
    // Non-fatal — log and continue without export names
    console.warn(`[WasmLoader] export read failed for '${moduleId}':`, err);
  }

  const instantiateTimeMs = Math.round(performance.now() - instantiateStart);

  // 5. Store in registry
  _evictIfFull();

  const entry: WasmModuleEntry = {
    moduleId,
    bytes:           buffer,
    byteLength:      buffer.byteLength,
    fetchTimeMs,
    instantiateTimeMs,
    exports,
    cachedAt:        Date.now(),
  };

  _registry.set(moduleId, entry);

  console.info(
    `[WasmLoader] loaded '${moduleId}' | ` +
    `size=${buffer.byteLength}B fetch=${fetchTimeMs}ms ` +
    `instantiate=${instantiateTimeMs}ms exports=[${exports.join(", ")}]`
  );

  return { success: true, entry, error: null };
}

// ---------------------------------------------------------------------------
// Convenience helpers
// ---------------------------------------------------------------------------

/**
 * Load a module and return only the raw ArrayBuffer.
 * Throws if the load fails — use for call sites that already handle errors.
 */
export async function loadModuleBytes(moduleId: string): Promise<ArrayBuffer> {
  const result = await loadModule(moduleId);
  if (!result.success || !result.entry) {
    throw new Error(result.error ?? `Failed to load WASM module '${moduleId}'.`);
  }
  return result.entry.bytes;
}

/**
 * Pre-warm the client cache by loading a list of module IDs concurrently.
 * Useful for pre-fetching common algorithms on page load.
 *
 * @param moduleIds  List of module IDs to pre-load.
 * @returns          Map of module_id → success boolean.
 */
export async function prefetch(moduleIds: string[]): Promise<Map<string, boolean>> {
  const results = await Promise.allSettled(moduleIds.map((id) => loadModule(id)));
  const summary = new Map<string, boolean>();
  results.forEach((result, i) => {
    summary.set(
      moduleIds[i],
      result.status === "fulfilled" && result.value.success,
    );
  });
  return summary;
}

/**
 * Remove a specific module from the client-side cache.
 * Call after the backend has invalidated a module.
 */
export function evictModule(moduleId: string): boolean {
  return _registry.delete(moduleId);
}

/**
 * Clear the entire client-side WASM cache.
 */
export function clearCache(): void {
  _registry.clear();
  console.info("[WasmLoader] client cache cleared.");
}

// ---------------------------------------------------------------------------
// WasmLoader class (stateful wrapper — optional alternative to the functions)
// ---------------------------------------------------------------------------

/**
 * Class-based wrapper around the module-level functions.
 * Useful when you need to scope the cache to a specific component lifecycle
 * rather than using the global registry.
 */
export class WasmLoader {
  private _localRegistry: Map<string, WasmModuleEntry> = new Map();

  async load(moduleId: string): Promise<WasmLoadResult> {
    // Check local registry first
    const local = this._localRegistry.get(moduleId);
    if (local) return { success: true, entry: local, error: null };

    // Fall through to global loader (also populates global registry)
    const result = await loadModule(moduleId);
    if (result.success && result.entry) {
      this._localRegistry.set(moduleId, result.entry);
    }
    return result;
  }

  async loadBytes(moduleId: string): Promise<ArrayBuffer> {
    const result = await this.load(moduleId);
    if (!result.success || !result.entry) {
      throw new Error(result.error ?? `Failed to load '${moduleId}'.`);
    }
    return result.entry.bytes;
  }

  get cachedIds(): string[] {
    return Array.from(this._localRegistry.keys());
  }

  clear(): void {
    this._localRegistry.clear();
  }

  stats(): { count: number; totalBytes: number } {
    let totalBytes = 0;
    for (const entry of this._localRegistry.values()) {
      totalBytes += entry.byteLength;
    }
    return { count: this._localRegistry.size, totalBytes };
  }
}