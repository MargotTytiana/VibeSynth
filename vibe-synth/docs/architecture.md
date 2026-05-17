# Vibe-Synth — System Architecture

**Document version:** 1.0  
**Last updated:** 2026-04-20  
**Status:** Living document — update when structural changes are merged.

---

## 1. Executive Summary

Vibe-Synth is a real-time algorithm synthesis engine powered by Large Language Models. Users describe a desired audio or visual character in natural language (a "Vibe"), and the system generates:

- **DSP path** — a compiled Faust algorithm loaded into a browser AudioWorklet
- **Shader path** — a validated WGSL or GLSL fragment shader rendered via WebGPU/WebGL

Unlike end-to-end AIGC models that output static `.wav` or `.png` files, Vibe-Synth outputs *rules* (algorithms), not data. The resulting algorithms are white-box, parameter-exposed, and infinitely re-renderable at zero marginal cost.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser Client                        │
│                                                             │
│  VibeInput.ts   →   AudioEngine.ts   →   AudioWorklet      │
│                 →   ShaderPreview.ts →   WebGPU/WebGL       │
│                 →   ParamPanel.ts    →   (live controls)    │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTPS (REST JSON)
┌──────────────────────────▼──────────────────────────────────┐
│                       FastAPI Backend                        │
│                                                             │
│  routes_dsp.py / routes_shader.py                           │
│       │                                                     │
│       ├── CacheService  ←──── VectorStore + SimilaritySearch│
│       ├── PromptBuilder ←──── system_dsp.txt / system_shader│
│       ├── LLMClient     ──→   OpenAI / Anthropic / Local    │
│       ├── Validator      ←──── banned patterns + structure  │
│       └── WasmCompiler  ──→   Faust binary / stub           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Module Map

### 3.1 Backend (`backend/`)

| Module | Path | Role |
|---|---|---|
| Config | `app/config.py` | Pydantic settings, single source of truth for all env vars |
| Dependencies | `app/dependencies.py` | FastAPI DI providers — singleton LLMClient, CacheService, Validator |
| Request models | `app/models/request_models.py` | Validated API input schemas |
| Response models | `app/models/response_models.py` | Typed API output schemas |
| Routes — DSP | `app/api/routes_dsp.py` | 7-step DSP generation pipeline |
| Routes — Shader | `app/api/routes_shader.py` | 6-step shader generation pipeline |
| Routes — Health | `app/api/routes_health.py` | Liveness, readiness, cache diagnostics |
| Entry point | `app/main.py` | FastAPI app factory, CORS, middleware, lifespan hooks |
| LLM client | `app/services/llm_client.py` | Multi-provider async LLM wrapper (OpenAI / Anthropic / local) |
| Prompt builder | `app/services/prompt_builder.py` | Assembles (system, user) prompt pairs from templates |
| Cache service | `app/services/cache_service.py` | Orchestrates VectorStore + SimilaritySearch |
| Validator | `app/services/validator.py` | Static banned-pattern checks + parameter manifest extraction |

### 3.2 Generation Layer (`generation/`)

| Module | Path | Role |
|---|---|---|
| DSP templates | `generation/dsp/dsp_templates.json` | 10 Faust algorithm archetypes with tagged skeletons |
| Shader templates | `generation/shader/shader_templates.json` | 10 WGSL/GLSL effect archetypes |
| Prompt templates | `generation/prompts/` | `system_dsp.txt`, `system_shader.txt`, `few_shot_examples.json` |
| AST builder | `generation/dsp/ast_builder.py` | Tag-scored template selection → SignalAST |
| Faust generator | `generation/dsp/faust_generator.py` | End-to-end DSP generation orchestrator |
| BIBO checker | `generation/dsp/bibo_checker.py` | Stability heuristic analysis |
| WGSL generator | `generation/shader/wgsl_generator.py` | WebGPU shader generation orchestrator |
| GLSL generator | `generation/shader/glsl_generator.py` | WebGL shader generation orchestrator |

### 3.3 Compiler (`compiler/`)

| Module | Path | Role |
|---|---|---|
| WASM compiler | `compiler/wasm_compiler.py` | Faust → WASM via local binary or stub |
| Sandbox runner | `compiler/sandbox_runner.py` | wasmtime fuel-limited execution check |
| Error handler | `compiler/error_handler.py` | Pattern-based error classification → LLM correction hints |
| Compile log | `compiler/compile_log.csv` | Per-attempt compilation telemetry |

### 3.4 Cache (`cache/`)

| Module | Path | Role |
|---|---|---|
| Vector store | `cache/vector_store.py` | CRUD + JSON persistence for VectorEntry objects |
| Similarity search | `cache/similarity_search.py` | Brute-force cosine similarity + threshold sweep |
| Cache index | `cache/cache_index.json` | On-disk index of prompt embeddings + cached results |

### 3.5 Frontend (`frontend/src/`)

| Module | Path | Role |
|---|---|---|
| VibeInput | `frontend/src/VibeInput.ts` | Input UI — debounced cache probe, API dispatch |
| AudioEngine | `frontend/src/AudioEngine.ts` | Web Audio context, WASM hot-swap, crossfade |
| WasmLoader | `frontend/src/WasmLoader.ts` | WASM fetch, magic-byte validation, client cache |
| ParamPanel | `frontend/src/ParamPanel.ts` | Dynamic control panel from parameter manifest |
| ShaderPreview | `frontend/src/ShaderPreview.ts` | WebGPU/WebGL render loop, uniform buffer |

---

## 4. Request Lifecycle — DSP Path

```
POST /api/v1/generate/dsp
         │
         ▼
1. CacheService.lookup()
   ├── hit  → return cached WASM module ID (< 25 ms)
   └── miss ↓
         │
2. ASTBuilder.build(vibe)
   └── tag-score templates → SignalAST → chain hint
         │
3. PromptBuilder.build_dsp_prompts()
   └── base template + few-shots + constraints + AST hint
         │
4. LLMClient.generate()
   └── OpenAI / Anthropic / local → raw Faust + json-params block
         │
5. Validator.validate_dsp()
   ├── pass  → (clean_code, parameters)
   └── fail  → LLMClient.self_correct() → re-validate (max 2 rounds)
         │
6. BIBOChecker.check()        [optional stability gate]
         │
7. WasmCompiler.compile_faust()
   └── local faust binary → .wasm bytes → module registry
         │
8. SandboxRunner.run()         [wasmtime fuel-limited check]
         │
9. CacheService.store()
         │
         ▼
    GenerateDSPResponse { wasm_module_id, parameters, ... }
```

---

## 5. Request Lifecycle — Shader Path

Mirrors the DSP path but skips steps 6 (BIBO) and 7–8 (WASM compilation/sandbox). The validated WGSL/GLSL source is returned as a string; compilation happens in the browser via the WebGPU device or WebGL context.

---

## 6. Caching Strategy

### Embedding
Prompt strings are embedded into a 128-dimensional float vector using a deterministic SHA-256 hash fallback in development. In production, replace `_embed()` in `cache_service.py` with a call to `text-embedding-3-small` or `sentence-transformers`.

### Similarity threshold
Default: **0.92 cosine similarity**. Tuned via `experiments/prompt_ablation.ipynb` (Ablation 3). Below 0.92, semantically unrelated results start appearing as hits; above 0.98, the cache is too conservative for near-duplicate Vibes.

### Eviction
FIFO at `cache_max_entries` (default: 1000). `VectorStore` tracks `hit_count` and `last_hit_at` for future LRU migration.

---

## 7. LLM Self-Correction Loop

```
LLMClient.generate(system, user)
    │ raw output
    ▼
Validator.validate_dsp/shader()
    │ failed
    ▼
ErrorHandler.parse_auto(stderr)
    │ structured ErrorReport + correction_hint
    ▼
LLMClient.self_correct(system, user, failed_code, error_message)
    │ corrected output
    ▼
Validator.validate_dsp/shader()     ← second pass
    │ still failed?
    ▼
HTTP 422 — Unprocessable Entity
```

Maximum self-correction rounds: **2**. Timeout and memory errors bypass the loop (marked `is_recoverable=False` by `ErrorHandler`).

---

## 8. Security Considerations

| Surface | Mitigation |
|---|---|
| Malicious Faust code | `Validator` banned-pattern scan + `BIBOChecker` before compilation |
| WASM infinite loops | `SandboxRunner` fuel limit (1M units) via wasmtime |
| Shader GPU hang | Loop-unrolling enforcement + `Validator` dynamic-bound check |
| LLM injection via Vibe | Prompt is user-controlled text; system prompt is server-side only |
| API abuse | Rate limiting should be added at the reverse proxy layer (nginx/Cloudflare) |

---

## 9. Scalability Notes

| Concern | Current approach | Production path |
|---|---|---|
| LLM latency | Single blocking async call | Request queue + streaming response |
| Cache | In-process dict + JSON file | Redis or PostgreSQL + pgvector |
| WASM module store | In-process dict | S3 or GCS object storage |
| Embedding | SHA-256 hash (offline) | OpenAI or Cohere embedding endpoint |
| Multiple workers | Single Uvicorn worker | Cache must move out-of-process first |

---

## 10. Key Design Decisions

**Why Faust for DSP?**  
Faust is a functional language purpose-built for audio DSP. It compiles to WebAssembly cleanly, expresses mathematical audio algorithms concisely, and its type system naturally rejects the most common LLM mistakes (arity mismatches, non-productive recursion).

**Why return shader source instead of compiling server-side?**  
WGSL and GLSL are compiled by the browser GPU driver, which has access to the actual GPU hardware. Server-side compilation would require matching the exact driver the client uses, which is impractical. Returning source and compiling client-side is the correct model.

**Why FIFO eviction instead of LRU?**  
The cache is small (≤ 1000 entries) and the access pattern for Vibe descriptions is not strongly recency-biased. FIFO is simpler to implement correctly and avoids the lock contention of LRU in a single-process async server.
