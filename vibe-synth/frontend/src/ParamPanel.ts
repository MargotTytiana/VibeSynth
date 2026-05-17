// =============================================================================
// File: frontend/src/ParamPanel.ts
// Purpose: Dynamically renders a control panel from a ParameterMeta manifest
//          received from the backend. Each parameter becomes a labelled slider,
//          knob (rendered as a range input), toggle, or dropdown depending on
//          its ui_mapping field. Value changes are dispatched to AudioEngine
//          (for DSP parameters) and ShaderPreview (for shader uniforms) via
//          a shared callback. Supports smooth value interpolation and
//          double-click-to-reset on individual controls.
// =============================================================================

import type { DSPParameter } from "./AudioEngine.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ParamChangeEvent {
  name:  string;
  value: number;
}

// Re-export for convenience — shader parameters share the same shape
export type ShaderParameter = DSPParameter;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SLIDER_STEPS = 1000;   // number of discrete steps for continuous sliders
const KNOB_STEPS   = 1000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Map a normalised [0, 1] position to a value within [min, max].
 * Uses a linear scale for most parameters and a logarithmic scale
 * for frequency parameters (detected by unit === "Hz").
 */
function normToValue(norm: number, min: number, max: number, isLog: boolean): number {
  if (isLog) {
    const logMin = Math.log(Math.max(min, 0.001));
    const logMax = Math.log(Math.max(max, 0.001));
    return Math.exp(logMin + norm * (logMax - logMin));
  }
  return min + norm * (max - min);
}

function valueToNorm(value: number, min: number, max: number, isLog: boolean): number {
  if (isLog) {
    const logMin = Math.log(Math.max(min, 0.001));
    const logMax = Math.log(Math.max(max, 0.001));
    const logVal = Math.log(Math.max(value, 0.001));
    return (logVal - logMin) / (logMax - logMin);
  }
  return (value - min) / (max - min);
}

function formatValue(value: number, unit: string | null): string {
  const rounded = Math.round(value * 100) / 100;
  return unit ? `${rounded} ${unit}` : String(rounded);
}

// ---------------------------------------------------------------------------
// ParamPanel
// ---------------------------------------------------------------------------

export class ParamPanel {
  private readonly root: HTMLElement;
  private parameters:    DSPParameter[] = [];
  private values:        Map<string, number> = new Map();

  /** Called whenever any parameter value changes. Wire to AudioEngine or ShaderPreview. */
  public onParamChange: ((event: ParamChangeEvent) => void) | null = null;

  constructor(rootSelector: string) {
    const el = document.querySelector<HTMLElement>(rootSelector);
    if (!el) {
      throw new Error(`ParamPanel: root element '${rootSelector}' not found.`);
    }
    this.root = el;
    this._renderEmpty();
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  /**
   * Replace the current controls with a new parameter set.
   * Called by AudioEngine.onParametersChanged and ShaderPreview.onParametersChanged.
   */
  public setParameters(params: DSPParameter[]): void {
    this.parameters = params;
    this.values.clear();
    for (const p of params) {
      this.values.set(p.name, p.default);
    }
    this._render();
  }

  /**
   * Programmatically update a single parameter value and refresh its control.
   */
  public setValue(name: string, value: number): void {
    this.values.set(name, value);
    this._updateControl(name, value);
  }

  /**
   * Reset all parameters to their default values.
   */
  public resetAll(): void {
    for (const p of this.parameters) {
      this.setValue(p.name, p.default);
      this.onParamChange?.({ name: p.name, value: p.default });
    }
  }

  /** Return a snapshot of current parameter values. */
  public getValues(): Record<string, number> {
    return Object.fromEntries(this.values);
  }

  // ------------------------------------------------------------------
  // Rendering
  // ------------------------------------------------------------------

  private _renderEmpty(): void {
    this.root.innerHTML = `
      <div class="param-panel-empty">
        <p>Generate an algorithm to see its parameters here.</p>
      </div>
    `;
  }

  private _render(): void {
    if (this.parameters.length === 0) {
      this._renderEmpty();
      return;
    }

    const header = `
      <div class="param-panel-header">
        <h3 class="param-panel-title">Parameters</h3>
        <button class="param-reset-all-btn" id="param-reset-all">Reset all</button>
      </div>
    `;

    const controls = this.parameters.map((p) => this._renderControl(p)).join("");

    this.root.innerHTML = `
      <div class="param-panel">
        ${header}
        <div class="param-controls">
          ${controls}
        </div>
      </div>
    `;

    // Attach reset-all button
    this.root
      .querySelector<HTMLButtonElement>("#param-reset-all")
      ?.addEventListener("click", () => this.resetAll());

    // Attach per-control listeners
    for (const p of this.parameters) {
      this._attachListeners(p);
    }
  }

  private _renderControl(p: DSPParameter): string {
    const value   = this.values.get(p.name) ?? p.default;
    const [min, max] = p.range;
    const isLog   = p.unit === "Hz";
    const norm    = valueToNorm(value, min, max, isLog);
    const display = formatValue(value, p.unit);

    const id      = `param-${p.name}`;
    const labelId = `${id}-label`;
    const valueId = `${id}-value`;

    if (p.ui_mapping === "toggle") {
      const checked = value >= 0.5 ? "checked" : "";
      return `
        <div class="param-row param-toggle-row" data-param="${p.name}">
          <label class="param-label" id="${labelId}" for="${id}">${p.name}</label>
          <input
            type="checkbox"
            id="${id}"
            class="param-toggle"
            data-param="${p.name}"
            ${checked}
          />
        </div>
      `;
    }

    if (p.ui_mapping === "dropdown") {
      // Discrete integer dropdown: render each integer step as an option
      const steps = Math.round(max - min);
      const options = Array.from({ length: steps + 1 }, (_, i) => {
        const v    = min + i;
        const sel  = Math.round(value) === v ? "selected" : "";
        return `<option value="${v}" ${sel}>${v}${p.unit ? ` ${p.unit}` : ""}</option>`;
      }).join("");
      return `
        <div class="param-row" data-param="${p.name}">
          <label class="param-label" id="${labelId}" for="${id}">${p.name}</label>
          <select id="${id}" class="param-dropdown" data-param="${p.name}">
            ${options}
          </select>
        </div>
      `;
    }

    // slider and knob both render as <input type="range">
    const steps = p.ui_mapping === "knob" ? KNOB_STEPS : SLIDER_STEPS;
    const normRounded = Math.round(norm * steps);

    return `
      <div class="param-row" data-param="${p.name}">
        <div class="param-label-row">
          <label class="param-label" id="${labelId}" for="${id}">${p.name}</label>
          <span class="param-value" id="${valueId}">${display}</span>
        </div>
        <input
          type="range"
          id="${id}"
          class="param-${p.ui_mapping}"
          data-param="${p.name}"
          data-min="${min}"
          data-max="${max}"
          data-log="${isLog}"
          data-unit="${p.unit ?? ""}"
          data-default="${p.default}"
          min="0"
          max="${steps}"
          value="${normRounded}"
          aria-labelledby="${labelId}"
          aria-valuenow="${display}"
        />
      </div>
    `;
  }

  // ------------------------------------------------------------------
  // Event listeners
  // ------------------------------------------------------------------

  private _attachListeners(p: DSPParameter): void {
    const id  = `param-${p.name}`;
    const el  = this.root.querySelector<HTMLElement>(`#${id}`);
    if (!el) return;

    if (p.ui_mapping === "toggle") {
      el.addEventListener("change", () => {
        const checked = (el as HTMLInputElement).checked;
        const value   = checked ? 1.0 : 0.0;
        this._emitChange(p.name, value);
      });
      return;
    }

    if (p.ui_mapping === "dropdown") {
      el.addEventListener("change", () => {
        const value = parseFloat((el as HTMLSelectElement).value);
        this._emitChange(p.name, value);
      });
      return;
    }

    // Range input (slider / knob)
    el.addEventListener("input", () => {
      const input   = el as HTMLInputElement;
      const steps   = parseInt(input.max, 10);
      const norm    = parseInt(input.value, 10) / steps;
      const min     = parseFloat(input.dataset.min!);
      const max     = parseFloat(input.dataset.max!);
      const isLog   = input.dataset.log === "true";
      const unit    = input.dataset.unit || null;
      const value   = normToValue(norm, min, max, isLog);

      this._emitChange(p.name, value);

      // Update displayed value label
      const valueEl = this.root.querySelector<HTMLSpanElement>(`#param-${p.name}-value`);
      if (valueEl) valueEl.textContent = formatValue(value, unit);

      // ARIA live update
      input.setAttribute("aria-valuenow", formatValue(value, unit));
    });

    // Double-click to reset to default
    el.addEventListener("dblclick", () => {
      const input       = el as HTMLInputElement;
      const defaultVal  = parseFloat(input.dataset.default!);
      const min         = parseFloat(input.dataset.min!);
      const max         = parseFloat(input.dataset.max!);
      const isLog       = input.dataset.log === "true";
      const unit        = input.dataset.unit || null;
      const steps       = parseInt(input.max, 10);
      const norm        = valueToNorm(defaultVal, min, max, isLog);
      input.value       = String(Math.round(norm * steps));

      this._emitChange(p.name, defaultVal);

      const valueEl = this.root.querySelector<HTMLSpanElement>(`#param-${p.name}-value`);
      if (valueEl) valueEl.textContent = formatValue(defaultVal, unit);
    });
  }

  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  private _emitChange(name: string, value: number): void {
    this.values.set(name, value);
    this.onParamChange?.({ name, value });
  }

  private _updateControl(name: string, value: number): void {
    const p = this.parameters.find((x) => x.name === name);
    if (!p) return;

    const id    = `param-${name}`;
    const el    = this.root.querySelector<HTMLElement>(`#${id}`);
    if (!el) return;

    if (p.ui_mapping === "toggle") {
      (el as HTMLInputElement).checked = value >= 0.5;
      return;
    }

    if (p.ui_mapping === "dropdown") {
      (el as HTMLSelectElement).value = String(Math.round(value));
      return;
    }

    // Range input
    const input  = el as HTMLInputElement;
    const [min, max] = p.range;
    const isLog  = p.unit === "Hz";
    const steps  = parseInt(input.max, 10);
    const norm   = valueToNorm(value, min, max, isLog);
    input.value  = String(Math.round(norm * steps));

    const valueEl = this.root.querySelector<HTMLSpanElement>(`#param-${name}-value`);
    if (valueEl) valueEl.textContent = formatValue(value, p.unit);
  }
}