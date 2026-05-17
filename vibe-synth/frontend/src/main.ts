// =============================================================================
// File: frontend/src/main.ts
// Purpose: Application entry point. Imports global styles, instantiates all
//          TypeScript components (VibeInput, AudioEngine, ShaderPreview,
//          ParamPanel), wires their callbacks together, and sets up the
//          audio start/stop buttons. This file is the single glue layer —
//          all business logic lives in the individual component modules.
// =============================================================================

import "./styles.css";

import { VibeInput }     from "./VibeInput.js";
import { AudioEngine }   from "./AudioEngine.js";
import { ShaderPreview } from "./ShaderPreview.js";
import { ParamPanel }    from "./ParamPanel.js";

// ---------------------------------------------------------------------------
// Guard — ensure the DOM is fully parsed before mounting components
// ---------------------------------------------------------------------------

function main(): void {

  // ── Instantiate components ────────────────────────────────────────────────

  const vibeInput = new VibeInput("#vibe-input-root");

  const audioEngine   = new AudioEngine();
  const shaderPreview = new ShaderPreview("#shader-canvas");

  const dspParamPanel    = new ParamPanel("#param-panel-dsp");
  const shaderParamPanel = new ParamPanel("#param-panel-shader");

  // ── Global status line ────────────────────────────────────────────────────

  const statusEl = document.querySelector<HTMLParagraphElement>("#global-status");

  function setGlobalStatus(msg: string, isError = false): void {
    if (!statusEl) return;
    statusEl.textContent = msg;
    statusEl.style.color = isError
      ? "var(--vs-color-error, #d32f2f)"
      : "var(--vs-color-muted, #666)";
  }

  // ── Wire AudioEngine callbacks ────────────────────────────────────────────

  audioEngine.onStatusChanged = (msg, isError) => {
    setGlobalStatus(msg, isError);
  };

  audioEngine.onParametersChanged = (params) => {
    dspParamPanel.setParameters(params);
  };

  // ── Wire ShaderPreview callbacks ──────────────────────────────────────────

  shaderPreview.onStatusChanged = (msg, isError) => {
    setGlobalStatus(msg, isError);
  };

  shaderPreview.onParametersChanged = (params) => {
    shaderParamPanel.setParameters(params);
  };

  // ── Wire ParamPanel → AudioEngine (DSP parameters) ───────────────────────

  dspParamPanel.onParamChange = ({ name, value }) => {
    audioEngine.setParameter(name, value);
  };

  // ── Wire ParamPanel → ShaderPreview (shader uniforms) ────────────────────

  shaderParamPanel.onParamChange = ({ name, value }) => {
    shaderPreview.setParameter(name, value);
  };

  // ── Audio start / stop buttons ────────────────────────────────────────────

  const btnStart = document.querySelector<HTMLButtonElement>("#btn-start");
  const btnStop  = document.querySelector<HTMLButtonElement>("#btn-stop");

  btnStart?.addEventListener("click", async () => {
    try {
      await audioEngine.start();
      btnStart.disabled = true;
      if (btnStop) btnStop.disabled = false;
      setGlobalStatus("Audio running.");
    } catch (err) {
      setGlobalStatus(`Could not start audio: ${err}`, true);
    }
  });

  btnStop?.addEventListener("click", async () => {
    await audioEngine.suspend();
    if (btnStart) btnStart.disabled = false;
    if (btnStop)  btnStop.disabled  = true;
    setGlobalStatus("Audio suspended.");
  });

  // ── Canvas resize observer ────────────────────────────────────────────────
  // Keep the shader canvas resolution in sync with its CSS display size
  // so the fragment shader always receives correct UV coordinates.

  const canvas = document.querySelector<HTMLCanvasElement>("#shader-canvas");
  if (canvas) {
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        canvas.width  = Math.round(width);
        canvas.height = Math.round(height);
      }
    });
    ro.observe(canvas);
  }

  // ── URL parameter pre-fill ────────────────────────────────────────────────
  // Allow sharing a Vibe via URL: ?vibe=freezing+ice+cave&type=dsp
  // The VibeInput component exposes setVibe() and setGenerationType()
  // as public methods for exactly this purpose.

  const params = new URLSearchParams(window.location.search);
  const preVibe = params.get("vibe");
  const preType = params.get("type");

  if (preVibe) {
    vibeInput.setVibe(decodeURIComponent(preVibe));
  }
  if (preType === "dsp" || preType === "shader") {
    vibeInput.setGenerationType(preType);
  }

  // ── Service worker (future PWA support) ───────────────────────────────────
  // Placeholder: register a service worker for offline caching of the
  // compiled WASM modules and shader source when PWA support is added.
  // if ("serviceWorker" in navigator) {
  //   navigator.serviceWorker.register("/sw.js");
  // }

  // ── Development helpers ───────────────────────────────────────────────────
  if (import.meta.env.DEV) {
    // Expose instances on window for browser DevTools debugging
    (window as Record<string, unknown>).vibeInput     = vibeInput;
    (window as Record<string, unknown>).audioEngine   = audioEngine;
    (window as Record<string, unknown>).shaderPreview = shaderPreview;
    (window as Record<string, unknown>).dspParams     = dspParamPanel;
    (window as Record<string, unknown>).shaderParams  = shaderParamPanel;
    console.info(
      "[Vibe-Synth] Dev mode — instances available on window:\n" +
      "  window.vibeInput, window.audioEngine, window.shaderPreview,\n" +
      "  window.dspParams, window.shaderParams"
    );
  }

  setGlobalStatus("Ready — enter a Vibe and press Generate.");
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main);
} else {
  main();
}
