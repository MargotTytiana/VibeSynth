// =============================================================================
// File: frontend/vite.config.ts
// Purpose: Vite build configuration for the Vibe-Synth frontend. Configures
//          the dev-server proxy (forwarding /api/v1/* and /wasm/* to the
//          FastAPI backend), path aliases, build output settings, and the
//          AudioWorklet worker entry point bundled as a separate IIFE chunk.
// =============================================================================

import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  // ── Path aliases ─────────────────────────────────────────────────────────
  // Keep import paths clean — @api/* and @utils/* resolve from src/
  resolve: {
    alias: {
      "@api":   resolve(__dirname, "src/api"),
      "@utils": resolve(__dirname, "src/utils"),
    },
  },

  // ── Development server ───────────────────────────────────────────────────
  server: {
    host:        "0.0.0.0",   // bind to all interfaces (required for Docker)
    port:        5173,
    strictPort:  true,         // fail fast if 5173 is already in use

    proxy: {
      // Forward all API calls to the FastAPI backend.
      // This avoids CORS issues during development — the browser sees
      // everything as coming from localhost:5173.
      "/api": {
        target:      "http://localhost:8000",
        changeOrigin: true,
        rewrite:     (path) => path,   // keep /api/v1/... prefix unchanged
      },

      // Forward WASM module fetch requests to the backend registry.
      "/wasm": {
        target:      "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },

  // ── Build output ─────────────────────────────────────────────────────────
  build: {
    outDir:          "dist",
    emptyOutDir:     true,
    sourcemap:       true,       // include source maps for production debugging
    target:          "es2022",   // match tsconfig target

    rollupOptions: {
      input: {
        // Main application entry point
        main: resolve(__dirname, "index.html"),

        // AudioWorklet processor must be bundled as a separate IIFE chunk
        // because AudioWorklet scripts run in a dedicated audio rendering
        // thread that does not share the module graph with the main thread.
        // The filename is referenced by AudioEngine.ts as WORKLET_SCRIPT.
        "audio-worklet-processor": resolve(
          __dirname,
          "src/audio-worklet-processor.ts"
        ),
      },

      output: {
        // Keep the worklet processor as a standalone IIFE file.
        // The main bundle uses standard ES module chunks.
        entryFileNames: (chunk) =>
          chunk.name === "audio-worklet-processor"
            ? "audio-worklet-processor.js"
            : "assets/[name]-[hash].js",

        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },

    // Warn if any individual chunk exceeds 600 kB (uncompressed).
    // The WASM bytes fetched at runtime are not part of the JS bundle.
    chunkSizeWarningLimit: 600,
  },

  // ── Worker / WASM support ─────────────────────────────────────────────────
  // WebAssembly modules are fetched at runtime from the backend registry
  // rather than bundled into the frontend — no special Vite WASM plugin needed.
  // The AudioWorklet is handled via rollupOptions.input above.

  // ── Optimisation ─────────────────────────────────────────────────────────
  optimizeDeps: {
    // Vite pre-bundles CJS dependencies into ESM. Since this project has no
    // third-party runtime dependencies (pure TypeScript), the list is empty.
    include: [],
  },
});
