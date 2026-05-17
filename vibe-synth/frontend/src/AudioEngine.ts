// =============================================================================
// File: frontend/src/AudioEngine.ts
// Purpose: Manages the Web Audio API context and AudioWorklet pipeline.
//          Listens for 'vibe:submit' events from VibeInput, fetches the
//          compiled WASM module ID from the backend, loads the WASM into an
//          AudioWorklet processor, and hot-swaps the active DSP node with a
//          crossfade to ensure pop-free transitions between algorithms.
//          Also exposes parameter update methods consumed by ParamPanel.
// =============================================================================

import type { VibeSubmitEvent } from "./VibeInput.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DSPParameter {
  name:       string;
  type:       string;
  range:      [number, number];
  default:    number;
  ui_mapping: string;
  unit:       string | null;
}

export interface GenerateDSPResponse {
  status:                    string;
  compile_time_ms:           number;
  cache_hit:                 boolean;
  faust_code:                string;
  wasm_module_id:            string;
  parameters:                DSPParameter[];
  self_correction_attempts:  number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API_BASE          = "/api/v1";
const CROSSFADE_MS      = 40;     // duration of gain crossfade on hot-swap
const WORKLET_SCRIPT    = "/audio-worklet-processor.js"; // served by Vite

// ---------------------------------------------------------------------------
// AudioEngine
// ---------------------------------------------------------------------------

export class AudioEngine {
  private context:        AudioContext | null        = null;
  private activeNode:     AudioWorkletNode | null    = null;
  private outputGain:     GainNode | null            = null;
  private currentModuleId: string | null             = null;
  private parameters:     DSPParameter[]             = [];

  // Callbacks
  public onParametersChanged: ((params: DSPParameter[]) => void) | null = null;
  public onStatusChanged:     ((msg: string, isError?: boolean) => void) | null = null;

  constructor() {
    // Listen for Vibe submission events dispatched by VibeInput
    document.addEventListener("vibe:submit", (e: Event) => {
      const event = e as CustomEvent<VibeSubmitEvent>;
      if (event.detail.generationType === "dsp") {
        this._handleVibeSubmit(event.detail);
      }
    });
  }

  // ------------------------------------------------------------------
  // AudioContext lifecycle
  // ------------------------------------------------------------------

  private async _ensureContext(): Promise<AudioContext> {
    if (this.context && this.context.state !== "closed") {
      // Resume if suspended (browser autoplay policy)
      if (this.context.state === "suspended") {
        await this.context.resume();
      }
      return this.context;
    }

    this.context = new AudioContext({ sampleRate: 44100 });

    // Load the AudioWorklet processor module
    try {
      await this.context.audioWorklet.addModule(WORKLET_SCRIPT);
    } catch (err) {
      this._emitStatus(
        `AudioWorklet module failed to load: ${err}. ` +
        "Ensure the dev server serves /audio-worklet-processor.js.",
        true,
      );
    }

    // Master output gain node — used for crossfade
    this.outputGain = this.context.createGain();
    this.outputGain.gain.value = 1.0;
    this.outputGain.connect(this.context.destination);

    return this.context;
  }

  // ------------------------------------------------------------------
  // Vibe submit handler
  // ------------------------------------------------------------------

  private async _handleVibeSubmit(event: VibeSubmitEvent): Promise<void> {
    this._emitStatus("Generating DSP algorithm…");

    let response: GenerateDSPResponse;
    try {
      response = await this._requestDSP(event);
    } catch (err) {
      this._emitStatus(`Generation failed: ${err}`, true);
      return;
    }

    this._emitStatus(
      `Algorithm ready — compiling WASM (${response.compile_time_ms} ms)` +
      (response.cache_hit ? " [cache hit]" : "") + "…"
    );

    try {
      await this._loadModule(response.wasm_module_id, response.parameters);
    } catch (err) {
      this._emitStatus(`WASM load failed: ${err}`, true);
      return;
    }

    this._emitStatus(
      `Running — ${response.parameters.length} parameter(s) available.` +
      (response.self_correction_attempts > 0
        ? ` (self-corrected ${response.self_correction_attempts}×)`
        : "")
    );
  }

  // ------------------------------------------------------------------
  // API request
  // ------------------------------------------------------------------

  private async _requestDSP(event: VibeSubmitEvent): Promise<GenerateDSPResponse> {
    const response = await fetch(`${API_BASE}/generate/dsp`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt:         event.vibe,
        input_type:     "audio_stream",
        sample_rate:    this.context?.sampleRate ?? 44100,
        max_parameters: 8,
        force_refresh:  event.forceRefresh,
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail ?? `HTTP ${response.status}`);
    }

    return response.json() as Promise<GenerateDSPResponse>;
  }

  // ------------------------------------------------------------------
  // WASM module loading and hot-swap
  // ------------------------------------------------------------------

  private async _loadModule(
    moduleId:   string,
    parameters: DSPParameter[],
  ): Promise<void> {
    // Skip reload if the same module is already active
    if (moduleId === this.currentModuleId) {
      this._emitStatus("Same module already loaded — parameters reset.");
      this._resetParameters(parameters);
      return;
    }

    const ctx = await this._ensureContext();

    // Fetch WASM bytes from the registry endpoint
    const wasmResponse = await fetch(`${API_BASE}/wasm/${moduleId}`);
    if (!wasmResponse.ok) {
      throw new Error(`Failed to fetch WASM module '${moduleId}': HTTP ${wasmResponse.status}`);
    }
    const wasmBuffer = await wasmResponse.arrayBuffer();

    // Instantiate the new AudioWorklet node
    const newNode = new AudioWorkletNode(ctx, "vibe-synth-processor", {
      processorOptions: { wasmBuffer, moduleId },
    });

    // Connect new node to the output (muted initially for crossfade)
    const newGain = ctx.createGain();
    newGain.gain.value = 0.0;
    newNode.connect(newGain);
    newGain.connect(this.outputGain!);

    // Crossfade: fade in new node, fade out old node
    const now = ctx.currentTime;
    const fadeTime = CROSSFADE_MS / 1000;

    newGain.gain.linearRampToValueAtTime(1.0, now + fadeTime);

    if (this.activeNode) {
      const oldGain = ctx.createGain();
      oldGain.gain.value = 1.0;
      this.activeNode.connect(oldGain);
      oldGain.connect(this.outputGain!);
      oldGain.gain.linearRampToValueAtTime(0.0, now + fadeTime);

      // Disconnect old node after crossfade completes
      setTimeout(() => {
        try {
          this.activeNode?.disconnect();
          oldGain.disconnect();
        } catch {
          // Ignore — node may already be disconnected
        }
      }, CROSSFADE_MS + 10);
    }

    this.activeNode      = newNode;
    this.currentModuleId = moduleId;
    this.parameters      = parameters;

    // Notify ParamPanel of new parameters
    this.onParametersChanged?.(parameters);

    // Apply default parameter values
    for (const param of parameters) {
      this._sendParameter(param.name, param.default);
    }
  }

  // ------------------------------------------------------------------
  // Parameter control
  // ------------------------------------------------------------------

  /**
   * Send a parameter value update to the active AudioWorklet processor.
   * Called by ParamPanel whenever a slider or knob is adjusted.
   */
  public setParameter(name: string, value: number): void {
    if (!this.activeNode) {
      console.warn("AudioEngine: no active DSP node — parameter update ignored.");
      return;
    }
    this._sendParameter(name, value);
  }

  private _sendParameter(name: string, value: number): void {
    this.activeNode?.port.postMessage({ type: "setParam", name, value });
  }

  private _resetParameters(parameters: DSPParameter[]): void {
    this.parameters = parameters;
    for (const p of parameters) {
      this._sendParameter(p.name, p.default);
    }
    this.onParametersChanged?.(parameters);
  }

  // ------------------------------------------------------------------
  // Playback control
  // ------------------------------------------------------------------

  /** Start or resume audio output. Must be called from a user gesture. */
  public async start(): Promise<void> {
    const ctx = await this._ensureContext();
    if (ctx.state === "suspended") {
      await ctx.resume();
    }
    this._emitStatus("Audio running.");
  }

  /** Suspend audio output without discarding the loaded module. */
  public async suspend(): Promise<void> {
    await this.context?.suspend();
    this._emitStatus("Audio suspended.");
  }

  /** Disconnect all nodes and close the AudioContext. */
  public async destroy(): Promise<void> {
    this.activeNode?.disconnect();
    this.outputGain?.disconnect();
    await this.context?.close();
    this.context        = null;
    this.activeNode     = null;
    this.outputGain     = null;
    this.currentModuleId = null;
  }

  // ------------------------------------------------------------------
  // Status helper
  // ------------------------------------------------------------------

  private _emitStatus(msg: string, isError = false): void {
    if (this.onStatusChanged) {
      this.onStatusChanged(msg, isError);
    } else {
      isError ? console.error("[AudioEngine]", msg) : console.log("[AudioEngine]", msg);
    }
  }

  // ------------------------------------------------------------------
  // Getters
  // ------------------------------------------------------------------

  public get activeModuleId(): string | null {
    return this.currentModuleId;
  }

  public get activeParameters(): DSPParameter[] {
    return [...this.parameters];
  }

  public get isRunning(): boolean {
    return this.context?.state === "running" && this.activeNode !== null;
  }
}