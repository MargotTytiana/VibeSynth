// =============================================================================
// File: frontend/src/VibeInput.ts
// Purpose: UI component that captures the user's natural language Vibe
//          description and dispatches generation requests to the backend API.
//          Handles input validation, debouncing, cache-probe feedback,
//          platform selection (WebGPU / WebGL), and generation-type toggling
//          (DSP audio vs shader visual). Emits typed CustomEvents consumed by
//          AudioEngine and ShaderPreview.
// =============================================================================

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type GenerationType = "dsp" | "shader";
export type TargetPlatform = "webgpu" | "webgl";

export interface VibeSubmitEvent {
  vibe:           string;
  generationType: GenerationType;
  platform:       TargetPlatform;
  forceRefresh:   boolean;
}

export interface CacheProbeResult {
  hit:             boolean;
  similarityScore: number | null;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API_BASE          = "/api/v1";
const DEBOUNCE_MS       = 600;   // delay before cache probe fires on keystroke
const MIN_VIBE_LENGTH   = 5;     // minimum characters before enabling submit
const MAX_VIBE_LENGTH   = 512;

// ---------------------------------------------------------------------------
// VibeInput
// ---------------------------------------------------------------------------

export class VibeInput {
  // DOM references
  private readonly root:          HTMLElement;
  private readonly textarea:      HTMLTextAreaElement;
  private readonly submitBtn:     HTMLButtonElement;
  private readonly typeToggle:    HTMLSelectElement;
  private readonly platformSelect: HTMLSelectElement;
  private readonly refreshCheck:  HTMLInputElement;
  private readonly charCount:     HTMLSpanElement;
  private readonly cacheIndicator: HTMLSpanElement;
  private readonly statusLine:    HTMLParagraphElement;

  // State
  private generationType: GenerationType = "dsp";
  private platform:       TargetPlatform = "webgpu";
  private debounceTimer:  ReturnType<typeof setTimeout> | null = null;
  private isLoading:      boolean = false;

  constructor(rootSelector: string) {
    const el = document.querySelector<HTMLElement>(rootSelector);
    if (!el) {
      throw new Error(`VibeInput: root element '${rootSelector}' not found.`);
    }
    this.root = el;
    this._render();

    // Bind references after render
    this.textarea       = this.root.querySelector<HTMLTextAreaElement>("#vibe-textarea")!;
    this.submitBtn      = this.root.querySelector<HTMLButtonElement>("#vibe-submit")!;
    this.typeToggle     = this.root.querySelector<HTMLSelectElement>("#vibe-type")!;
    this.platformSelect = this.root.querySelector<HTMLSelectElement>("#vibe-platform")!;
    this.refreshCheck   = this.root.querySelector<HTMLInputElement>("#vibe-refresh")!;
    this.charCount      = this.root.querySelector<HTMLSpanElement>("#vibe-charcount")!;
    this.cacheIndicator = this.root.querySelector<HTMLSpanElement>("#vibe-cache-indicator")!;
    this.statusLine     = this.root.querySelector<HTMLParagraphElement>("#vibe-status")!;

    this._attachListeners();
    this._updateSubmitState();
  }

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------

  private _render(): void {
    this.root.innerHTML = `
      <div class="vibe-input-container">

        <label for="vibe-type" class="vibe-label">Generation type</label>
        <select id="vibe-type" class="vibe-select">
          <option value="dsp">DSP — Audio Effect</option>
          <option value="shader">Shader — Visual Effect</option>
        </select>

        <label for="vibe-platform" class="vibe-label" id="platform-label">
          Target platform
        </label>
        <select id="vibe-platform" class="vibe-select" id="vibe-platform">
          <option value="webgpu">WebGPU (WGSL) — recommended</option>
          <option value="webgl">WebGL (GLSL ES 3.0) — wider support</option>
        </select>

        <label for="vibe-textarea" class="vibe-label">Describe your Vibe</label>
        <textarea
          id="vibe-textarea"
          class="vibe-textarea"
          rows="4"
          maxlength="${MAX_VIBE_LENGTH}"
          placeholder="e.g. 'Make the sound feel like an empty, freezing ice cave' or 'Glitchy VHS with neon RGB split'"
        ></textarea>

        <div class="vibe-meta-row">
          <span id="vibe-charcount" class="vibe-charcount">0 / ${MAX_VIBE_LENGTH}</span>
          <span id="vibe-cache-indicator" class="vibe-cache-indicator"></span>
        </div>

        <label class="vibe-label vibe-refresh-label">
          <input type="checkbox" id="vibe-refresh" />
          Force refresh (bypass cache)
        </label>

        <button id="vibe-submit" class="vibe-submit-btn" disabled>
          Generate
        </button>

        <p id="vibe-status" class="vibe-status" aria-live="polite"></p>

      </div>
    `;
  }

  // ------------------------------------------------------------------
  // Event listeners
  // ------------------------------------------------------------------

  private _attachListeners(): void {
    // Textarea — character count + debounced cache probe
    this.textarea.addEventListener("input", () => {
      const len = this.textarea.value.length;
      this.charCount.textContent = `${len} / ${MAX_VIBE_LENGTH}`;
      this._updateSubmitState();
      this._scheduleCacheProbe();
    });

    // Generation type toggle — show/hide platform selector
    this.typeToggle.addEventListener("change", () => {
      this.generationType = this.typeToggle.value as GenerationType;
      const platformRow = this.root.querySelector<HTMLElement>("#platform-label");
      if (platformRow) {
        platformRow.style.display = this.generationType === "shader" ? "" : "none";
        this.platformSelect.style.display = this.generationType === "shader" ? "" : "none";
      }
      this._clearCacheIndicator();
      this._scheduleCacheProbe();
    });

    // Platform selector
    this.platformSelect.addEventListener("change", () => {
      this.platform = this.platformSelect.value as TargetPlatform;
      this._clearCacheIndicator();
    });

    // Submit button
    this.submitBtn.addEventListener("click", () => {
      this._handleSubmit();
    });

    // Keyboard shortcut: Ctrl/Cmd + Enter to submit
    this.textarea.addEventListener("keydown", (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        if (!this.submitBtn.disabled) {
          this._handleSubmit();
        }
      }
    });
  }

  // ------------------------------------------------------------------
  // Submit handler
  // ------------------------------------------------------------------

  private async _handleSubmit(): Promise<void> {
    if (this.isLoading) return;

    const vibe = this.textarea.value.trim();
    if (vibe.length < MIN_VIBE_LENGTH) return;

    this._setLoading(true);
    this._setStatus("Generating… this may take a few seconds.");

    const event: VibeSubmitEvent = {
      vibe,
      generationType: this.generationType,
      platform:       this.platform,
      forceRefresh:   this.refreshCheck.checked,
    };

    // Dispatch a typed CustomEvent for AudioEngine / ShaderPreview to consume
    this.root.dispatchEvent(
      new CustomEvent<VibeSubmitEvent>("vibe:submit", {
        detail:  event,
        bubbles: true,
      })
    );

    // Call the appropriate generation endpoint
    try {
      const result = await this._generate(event);
      this._setStatus(
        `Done! Compiled in ${result.compile_time_ms} ms` +
        (result.cache_hit ? " (served from cache)." : ".")
      );
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      this._setStatus(`Error: ${msg}`, true);
    } finally {
      this._setLoading(false);
    }
  }

  // ------------------------------------------------------------------
  // API calls
  // ------------------------------------------------------------------

  private async _generate(event: VibeSubmitEvent): Promise<{ compile_time_ms: number; cache_hit: boolean }> {
    const endpoint =
      event.generationType === "dsp"
        ? `${API_BASE}/generate/dsp`
        : `${API_BASE}/generate/shader`;

    const body =
      event.generationType === "dsp"
        ? {
            prompt:         event.vibe,
            input_type:     "audio_stream",
            sample_rate:    44100,
            max_parameters: 8,
            force_refresh:  event.forceRefresh,
          }
        : {
            prompt:          event.vibe,
            input_type:      "video_stream",
            target_platform: event.platform,
            max_parameters:  8,
            force_refresh:   event.forceRefresh,
          };

    const response = await fetch(endpoint, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(err.detail ?? err.message ?? `HTTP ${response.status}`);
    }

    return response.json();
  }

  private async _probCache(vibe: string): Promise<CacheProbeResult> {
    const endpoint =
      this.generationType === "dsp"
        ? `${API_BASE}/cache/lookup/dsp`
        : `${API_BASE}/cache/lookup/shader`;

    const response = await fetch(endpoint, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ prompt: vibe, generation_type: this.generationType }),
    });

    if (!response.ok) return { hit: false, similarityScore: null };

    const data = await response.json();
    return {
      hit:             data.hit,
      similarityScore: data.similarity_score ?? null,
    };
  }

  // ------------------------------------------------------------------
  // Cache probe debounce
  // ------------------------------------------------------------------

  private _scheduleCacheProbe(): void {
    if (this.debounceTimer !== null) {
      clearTimeout(this.debounceTimer);
    }
    const vibe = this.textarea.value.trim();
    if (vibe.length < MIN_VIBE_LENGTH) {
      this._clearCacheIndicator();
      return;
    }
    this.debounceTimer = setTimeout(async () => {
      try {
        const result = await this._probCache(vibe);
        this._updateCacheIndicator(result);
      } catch {
        this._clearCacheIndicator();
      }
    }, DEBOUNCE_MS);
  }

  // ------------------------------------------------------------------
  // UI helpers
  // ------------------------------------------------------------------

  private _updateSubmitState(): void {
    const len = this.textarea.value.trim().length;
    this.submitBtn.disabled = this.isLoading || len < MIN_VIBE_LENGTH;
  }

  private _setLoading(loading: boolean): void {
    this.isLoading = loading;
    this.submitBtn.textContent = loading ? "Generating…" : "Generate";
    this.textarea.disabled     = loading;
    this._updateSubmitState();
  }

  private _setStatus(message: string, isError = false): void {
    this.statusLine.textContent = message;
    this.statusLine.style.color = isError ? "var(--color-error, #d32f2f)" : "";
  }

  private _updateCacheIndicator(result: CacheProbeResult): void {
    if (result.hit) {
      const score = result.similarityScore !== null
        ? ` (${(result.similarityScore * 100).toFixed(1)}% match)`
        : "";
      this.cacheIndicator.textContent = `⚡ Cached result available${score}`;
      this.cacheIndicator.style.color = "var(--color-success, #2e7d32)";
    } else {
      this.cacheIndicator.textContent = "○ No cached result — will generate fresh";
      this.cacheIndicator.style.color = "var(--color-muted, #757575)";
    }
  }

  private _clearCacheIndicator(): void {
    this.cacheIndicator.textContent = "";
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  /** Pre-fill the textarea with a vibe string (e.g. from a URL parameter). */
  public setVibe(vibe: string): void {
    this.textarea.value = vibe.slice(0, MAX_VIBE_LENGTH);
    this.textarea.dispatchEvent(new Event("input"));
  }

  /** Programmatically switch the generation type. */
  public setGenerationType(type: GenerationType): void {
    this.typeToggle.value = type;
    this.typeToggle.dispatchEvent(new Event("change"));
  }
}