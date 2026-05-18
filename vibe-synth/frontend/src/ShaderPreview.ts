// =============================================================================
// File: frontend/src/ShaderPreview.ts
// Purpose: Manages the WebGPU / WebGL rendering pipeline for real-time shader
//          preview. Listens for 'vibe:submit' events from VibeInput (shader
//          generation type), compiles the returned WGSL or GLSL source in the
//          browser, binds a uniform buffer for time and user parameters, and
//          renders to a <canvas> element at 60 fps via requestAnimationFrame.
//          Hot-swaps the active shader pipeline without dropping frames.
// =============================================================================

import type { VibeSubmitEvent } from "./VibeInput.js";
import type { ShaderParameter } from "./ParamPanel.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GenerateShaderResponse {
  status:                   string;
  compile_time_ms:          number;
  cache_hit:                boolean;
  shader_code:              string;
  target_platform:          string;
  parameters:               ShaderParameter[];
  self_correction_attempts: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API_BASE         = "/api/v1";
const TARGET_FPS       = 60;
const FRAME_BUDGET_MS  = 1000 / TARGET_FPS;

// Uniform buffer layout (std140 / WGSL):
//   offset 0:  f32 time
//   offset 4:  f32 param0
//   offset 8:  f32 param1
//   ...
//   Total: (1 + MAX_PARAMS) * 4 bytes, padded to 16-byte alignment
const MAX_PARAMS         = 8;
const UNIFORM_FLOATS     = 1 + MAX_PARAMS;                   // time + params
const UNIFORM_BYTE_SIZE  = Math.ceil(UNIFORM_FLOATS / 4) * 16; // 16-byte aligned

// ---------------------------------------------------------------------------
// ShaderPreview
// ---------------------------------------------------------------------------

export class ShaderPreview {
  private readonly canvas:   HTMLCanvasElement;

  // WebGPU state
  private gpuDevice:         GPUDevice | null         = null;
  private gpuContext:        GPUCanvasContext | null  = null;
  private gpuPipeline:       GPURenderPipeline | null = null;
  private gpuUniformBuffer:  GPUBuffer | null         = null;
  private gpuBindGroup:      GPUBindGroup | null      = null;

  // WebGL fallback state
  private glContext:         WebGL2RenderingContext | null = null;
  private glProgram:         WebGLProgram | null           = null;

  // Shared state
  private platform:          string           = "webgpu";
  private uniformData:       Float32Array     = new Float32Array(UNIFORM_FLOATS);
  private paramValues:       Map<string, number> = new Map();
  private parameters:        ShaderParameter[] = [];
  private rafId:             number           = 0;
  private startTime:         number           = performance.now();
  private isRunning:         boolean          = false;

  // Callbacks
  public onParametersChanged: ((params: ShaderParameter[]) => void) | null = null;
  public onStatusChanged:     ((msg: string, isError?: boolean) => void) | null = null;

  constructor(canvasSelector: string) {
    const el = document.querySelector<HTMLCanvasElement>(canvasSelector);
    if (!el) {
      throw new Error(`ShaderPreview: canvas '${canvasSelector}' not found.`);
    }
    this.canvas = el;

    // Listen for Vibe submission events — shader type only
    document.addEventListener("vibe:submit", (e: Event) => {
      const event = e as CustomEvent<VibeSubmitEvent>;
      if (event.detail.generationType === "shader") {
        this._handleVibeSubmit(event.detail);
      }
    });
  }

  // ------------------------------------------------------------------
  // Vibe submit handler
  // ------------------------------------------------------------------

  private async _handleVibeSubmit(event: VibeSubmitEvent): Promise<void> {
    this._emitStatus("Generating shader…");

    let response: GenerateShaderResponse;
    try {
      response = await this._requestShader(event);
    } catch (err) {
      this._emitStatus(`Shader generation failed: ${err}`, true);
      return;
    }

    this.platform = response.target_platform;
    this._emitStatus(
      `Shader received (${response.compile_time_ms} ms` +
      (response.cache_hit ? ", cache hit" : "") +
      "). Compiling in browser…"
    );

    try {
      if (response.target_platform === "webgpu") {
        await this._initWebGPU(response.shader_code, response.parameters);
      } else {
        this._initWebGL(response.shader_code, response.parameters);
      }
    } catch (err) {
      this._emitStatus(`Browser shader compilation failed: ${err}`, true);
      return;
    }

    this._startRenderLoop();
    this.onParametersChanged?.(response.parameters);

    this._emitStatus(
      `Rendering at ${TARGET_FPS} fps — ${response.parameters.length} parameter(s).` +
      (response.self_correction_attempts > 0
        ? ` (self-corrected ${response.self_correction_attempts}×)`
        : "")
    );
  }

  // ------------------------------------------------------------------
  // API request
  // ------------------------------------------------------------------

  private async _requestShader(event: VibeSubmitEvent): Promise<GenerateShaderResponse> {
    const response = await fetch(`${API_BASE}/generate/shader`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt:          event.vibe,
        input_type:      "video_stream",
        target_platform: event.platform,
        max_parameters:  MAX_PARAMS,
        force_refresh:   event.forceRefresh,
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail ?? `HTTP ${response.status}`);
    }

    return response.json() as Promise<GenerateShaderResponse>;
  }

  // ------------------------------------------------------------------
  // WebGPU pipeline
  // ------------------------------------------------------------------

  private async _initWebGPU(wgslCode: string, params: ShaderParameter[]): Promise<void> {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not supported in this browser. Try Chrome 113+ or Edge 113+.");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter found.");

    this.gpuDevice  = await adapter.requestDevice();
    this.gpuContext = this.canvas.getContext("webgpu") as GPUCanvasContext;

    const format = navigator.gpu.getPreferredCanvasFormat();
    this.gpuContext.configure({ device: this.gpuDevice, format });

    // Compile the WGSL shader module
    const shaderModule = this.gpuDevice.createShaderModule({ code: wgslCode });

    // Uniform buffer: time + up to MAX_PARAMS floats
    this.gpuUniformBuffer = this.gpuDevice.createBuffer({
      size:  UNIFORM_BYTE_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Bind group layout
    const bindGroupLayout = this.gpuDevice.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });

    this.gpuBindGroup = this.gpuDevice.createBindGroup({
      layout:  bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.gpuUniformBuffer } }],
    });

    // Render pipeline — fullscreen triangle (no vertex buffer needed)
    this.gpuPipeline = this.gpuDevice.createRenderPipeline({
      layout:   this.gpuDevice.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex:   { module: shaderModule, entryPoint: "vs_main" },
      fragment: {
        module:      shaderModule,
        entryPoint:  "fs_main",
        targets:     [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });

    this._initParams(params);
  }

  // ------------------------------------------------------------------
  // WebGL fallback pipeline
  // ------------------------------------------------------------------

  private _initWebGL(glslCode: string, params: ShaderParameter[]): void {
    const gl = this.canvas.getContext("webgl2");
    if (!gl) throw new Error("WebGL 2 is not supported in this browser.");
    this.glContext = gl;

    // Minimal fullscreen quad vertex shader
    const vsSource = `#version 300 es
      in vec2 a_position;
      out vec2 v_uv;
      void main() {
        v_uv        = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `;

    const vs = this._compileGLShader(gl, gl.VERTEX_SHADER, vsSource);
    const fs = this._compileGLShader(gl, gl.FRAGMENT_SHADER, glslCode);

    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program) ?? "unknown link error";
      gl.deleteProgram(program);
      throw new Error(`GLSL program link failed: ${log}`);
    }

    this.glProgram = program;

    // Upload a fullscreen quad (two triangles)
    const positions = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const loc = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    this._initParams(params);
  }

  private _compileGLShader(
    gl: WebGL2RenderingContext,
    type: number,
    source: string,
  ): WebGLShader {
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader) ?? "unknown compile error";
      gl.deleteShader(shader);
      throw new Error(`GLSL ${type === gl.VERTEX_SHADER ? "vertex" : "fragment"} shader error: ${log}`);
    }
    return shader;
  }

  // ------------------------------------------------------------------
  // Parameter management
  // ------------------------------------------------------------------

  private _initParams(params: ShaderParameter[]): void {
    this.parameters = params;
    this.paramValues.clear();
    for (const p of params) {
      this.paramValues.set(p.name, p.default);
    }
    this._buildUniformData();
  }

  /**
   * Update a single shader uniform parameter value.
   * Called by ParamPanel.onParamChange.
   */
  public setParameter(name: string, value: number): void {
    this.paramValues.set(name, value);
    this._buildUniformData();
  }

  private _buildUniformData(): void {
    // Layout: [time, param0, param1, ..., paramN]
    // time is set per-frame; pre-fill parameters now
    for (let i = 0; i < this.parameters.length && i < MAX_PARAMS; i++) {
      const value = this.paramValues.get(this.parameters[i].name) ?? this.parameters[i].default;
      this.uniformData[i + 1] = value;
    }
  }

  // ------------------------------------------------------------------
  // Render loop
  // ------------------------------------------------------------------

  private _startRenderLoop(): void {
    this._stopRenderLoop();
    this.isRunning = true;
    this.startTime = performance.now();

    const frame = (): void => {
      if (!this.isRunning) return;
      this._drawFrame();
      this.rafId = requestAnimationFrame(frame);
    };

    this.rafId = requestAnimationFrame(frame);
  }

  private _stopRenderLoop(): void {
    this.isRunning = false;
    if (this.rafId !== 0) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }
  }

  private _drawFrame(): void {
    const elapsed = (performance.now() - this.startTime) / 1000;
    this.uniformData[0] = elapsed;  // time uniform

    if (this.platform === "webgpu") {
      this._drawWebGPU();
    } else {
      this._drawWebGL(elapsed);
    }
  }

  private _drawWebGPU(): void {
    if (!this.gpuDevice || !this.gpuContext || !this.gpuPipeline ||
        !this.gpuUniformBuffer || !this.gpuBindGroup) return;

    // Upload uniform data
    this.gpuDevice.queue.writeBuffer(
      this.gpuUniformBuffer, 0,
      this.uniformData.buffer, 0,
      UNIFORM_FLOATS * 4,
    );

    const encoder    = this.gpuDevice.createCommandEncoder();
    const colorView  = this.gpuContext.getCurrentTexture().createView();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view:       colorView,
        loadOp:     "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp:    "store",
      }],
    });

    renderPass.setPipeline(this.gpuPipeline);
    renderPass.setBindGroup(0, this.gpuBindGroup);
    renderPass.draw(3);  // fullscreen triangle
    renderPass.end();

    this.gpuDevice.queue.submit([encoder.finish()]);
  }

  private _drawWebGL(elapsed: number): void {
    const gl = this.glContext;
    if (!gl || !this.glProgram) return;

    gl.useProgram(this.glProgram);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Upload time uniform
    const timeLoc = gl.getUniformLocation(this.glProgram, "u_time");
    if (timeLoc !== null) gl.uniform1f(timeLoc, elapsed);

    // Upload parameter uniforms
    for (let i = 0; i < this.parameters.length && i < MAX_PARAMS; i++) {
      const p     = this.parameters[i];
      const loc   = gl.getUniformLocation(this.glProgram, `u_${p.name}`);
      const value = this.paramValues.get(p.name) ?? p.default;
      if (loc !== null) gl.uniform1f(loc, value);
    }

    gl.drawArrays(gl.TRIANGLES, 0, 6);  // two triangles = fullscreen quad
  }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  public stop(): void {
    this._stopRenderLoop();
  }

  public destroy(): void {
    this._stopRenderLoop();
    this.gpuDevice?.destroy();
    this.gpuDevice  = null;
    this.gpuContext = null;
    this.glContext  = null;
    this.glProgram  = null;
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  private _emitStatus(msg: string, isError = false): void {
    if (this.onStatusChanged) {
      this.onStatusChanged(msg, isError);
    } else {
      isError ? console.error("[ShaderPreview]", msg) : console.log("[ShaderPreview]", msg);
    }
  }

  public get activeParameters(): ShaderParameter[] {
    return [...this.parameters];
  }
}