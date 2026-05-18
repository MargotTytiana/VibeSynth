# Vibe-Synth — API Reference

**Base URL:** `http://localhost:8000` (development) · `https://api.vibesynth.io` (production)  
**API version:** v1  
**Authentication:** None required for development. Add Bearer token middleware before production deployment.  
**Content-Type:** `application/json` for all request and response bodies.

---

## Table of Contents

1. [DSP Generation](#1-dsp-generation)
2. [Shader Generation](#2-shader-generation)
3. [Cache Endpoints](#3-cache-endpoints)
4. [Health & Diagnostics](#4-health--diagnostics)
5. [Common Types](#5-common-types)
6. [Error Responses](#6-error-responses)
7. [Rate Limits & Timeouts](#7-rate-limits--timeouts)

---

## 1. DSP Generation

### POST `/api/v1/generate/dsp`

Generate a Faust DSP algorithm from a natural language Vibe description.

**Pipeline:** cache lookup → AST template selection → LLM generation → validation → WASM compilation → cache store

#### Request Body

```json
{
  "prompt":         "Make the sound feel like an empty, freezing ice cave",
  "input_type":     "audio_stream",
  "sample_rate":    44100,
  "max_parameters": 8,
  "force_refresh":  false
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | ✅ | — | Natural language Vibe description. 3–512 characters. |
| `input_type` | enum | ❌ | `audio_stream` | `audio_stream` or `text_only` |
| `sample_rate` | integer | ❌ | `44100` | Target sample rate in Hz. Range: 8000–192000. |
| `max_parameters` | integer | ❌ | `8` | Maximum number of exposed UI parameters. Range: 1–32. |
| `force_refresh` | boolean | ❌ | `false` | If `true`, bypass the vector cache and generate fresh. |

#### Response Body — `200 OK`

```json
{
  "status":                   "success",
  "compile_time_ms":          312,
  "cache_hit":                false,
  "faust_code":               "import(\"stdfaust.lib\");\n\nroom_size = ...",
  "wasm_module_id":           "wasm-a3f1c2d4e5b6f7a8",
  "parameters": [
    {
      "name":       "room_size",
      "type":       "float",
      "range":      [0.1, 1.0],
      "default":    0.35,
      "ui_mapping": "slider",
      "unit":       null
    }
  ],
  "self_correction_attempts": 0
}
```

| Field | Type | Description |
|---|---|---|
| `status` | enum | `success` · `cache_hit` · `self_corrected` · `error` |
| `compile_time_ms` | integer | Wall-clock time for the full pipeline in milliseconds. |
| `cache_hit` | boolean | `true` if the result was served from the vector cache. |
| `faust_code` | string | Clean, validated Faust DSP source code. |
| `wasm_module_id` | string | ID referencing the compiled WASM module in the server registry. Pass to `GET /api/v1/wasm/{module_id}`. |
| `parameters` | array | Parameter manifest. See [ParameterMeta](#parametermeta). |
| `self_correction_attempts` | integer | Number of LLM self-correction rounds used (0 on first-pass success). |

#### Status Codes

| Code | Meaning |
|---|---|
| `200` | Success — result returned (may be `cache_hit` or `self_corrected`). |
| `422` | Validation failure — generated code could not be corrected after max attempts. |
| `502` | LLM provider error — upstream API returned an error or timed out. |
| `500` | Internal server error — WASM compilation or sandbox failure. |

---

### POST `/api/v1/cache/lookup/dsp`

Probe the vector cache for a DSP prompt without triggering generation.

#### Request Body

```json
{
  "prompt":          "Small wooden room reverb",
  "generation_type": "dsp"
}
```

#### Response Body — `200 OK`

```json
{
  "hit":             true,
  "similarity_score": 0.9612,
  "cached_result":   { ... }
}
```

| Field | Type | Description |
|---|---|---|
| `hit` | boolean | `true` if `similarity_score` ≥ `cache_similarity_threshold` (default 0.92). |
| `similarity_score` | float \| null | Cosine similarity of the best matching cached entry. `null` if no entries exist. |
| `cached_result` | object \| null | Full cached API response dict, or `null` on a miss. |

---

## 2. Shader Generation

### POST `/api/v1/generate/shader`

Generate a WGSL or GLSL ES 3.0 fragment shader from a Vibe description.

**Pipeline:** cache lookup → template selection → LLM generation → validation → cache store  
*(No server-side compilation — shader source is returned as a string; compilation happens in the browser.)*

#### Request Body

```json
{
  "prompt":          "Frosted glass distortion with cyberpunk chromatic aberration",
  "input_type":      "video_stream",
  "target_platform": "webgpu",
  "max_parameters":  8,
  "force_refresh":   false
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | ✅ | — | Natural language Vibe description. 3–512 characters. |
| `input_type` | enum | ❌ | `video_stream` | `video_stream`, `audio_stream`, or `text_only` |
| `target_platform` | enum | ❌ | `webgpu` | `webgpu` (WGSL) or `webgl` (GLSL ES 3.0) |
| `max_parameters` | integer | ❌ | `8` | Maximum number of exposed UI parameters. Range: 1–32. |
| `force_refresh` | boolean | ❌ | `false` | Bypass cache and generate fresh. |

#### Response Body — `200 OK`

```json
{
  "status":                   "success",
  "compile_time_ms":          287,
  "cache_hit":                false,
  "shader_code":              "@group(0) @binding(0) var input_texture ...",
  "target_platform":          "webgpu",
  "parameters": [
    {
      "name":       "frost_intensity",
      "type":       "float",
      "range":      [0.0, 1.0],
      "default":    0.5,
      "ui_mapping": "slider",
      "unit":       null
    }
  ],
  "self_correction_attempts": 0
}
```

| Field | Type | Description |
|---|---|---|
| `status` | enum | `success` · `cache_hit` · `self_corrected` · `error` |
| `compile_time_ms` | integer | Wall-clock time for the server-side pipeline (excludes browser GPU compilation). |
| `cache_hit` | boolean | `true` if result was served from cache. |
| `shader_code` | string | Validated WGSL or GLSL ES 3.0 fragment shader source. |
| `target_platform` | string | `webgpu` or `webgl` — matches the request. |
| `parameters` | array | Parameter manifest. See [ParameterMeta](#parametermeta). |
| `self_correction_attempts` | integer | LLM self-correction rounds used. |

#### Available Platforms

```
GET /api/v1/generate/shader/platforms
```

```json
{
  "platforms": [
    { "id": "webgpu", "language": "WGSL",        "description": "WebGPU — modern, Chrome 113+" },
    { "id": "webgl",  "language": "GLSL ES 3.0", "description": "WebGL — wider compatibility"  }
  ]
}
```

---

### POST `/api/v1/cache/lookup/shader`

Probe the vector cache for a shader prompt without triggering generation.

#### Request Body

```json
{
  "prompt":          "Glitchy VHS scanline tears",
  "generation_type": "shader"
}
```

Response schema identical to [`/api/v1/cache/lookup/dsp`](#post-apiv1cachelookup-dsp).

---

## 3. Cache Endpoints

### GET `/api/v1/health/cache`

Return cache index statistics.

#### Response — `200 OK`

```json
{
  "status": "ok",
  "cache": {
    "total_entries":   12,
    "dsp_entries":     7,
    "shader_entries":  5,
    "max_entries":     1000,
    "threshold":       0.92,
    "index_path":      "cache/cache_index.json"
  }
}
```

---

## 4. Health & Diagnostics

### GET `/api/v1/health`

Liveness probe. Returns `200` whenever the process is running. No external calls made.

```json
{ "status": "ok", "version": "0.1.0" }
```

---

### GET `/api/v1/health/ready`

Readiness probe. Verifies cache index is loaded and LLM provider is configured.

```json
{
  "status": "ready",
  "checks": {
    "cache": "ok (12 entries)",
    "llm":   "ok (provider=openai)"
  }
}
```

| `status` value | Meaning |
|---|---|
| `ready` | All subsystems healthy. |
| `not_ready` | At least one subsystem reported an error. |

---

## 5. Common Types

### ParameterMeta

Describes a single runtime-controllable parameter returned in the `parameters` array of generation responses.

```json
{
  "name":       "room_size",
  "type":       "float",
  "range":      [0.1, 1.0],
  "default":    0.35,
  "ui_mapping": "slider",
  "unit":       null
}
```

| Field | Type | Values | Description |
|---|---|---|---|
| `name` | string | snake_case | Identifier used in the algorithm source. |
| `type` | string | `float` · `int` · `bool` | Value type. |
| `range` | array | `[min, max]` | Bounds for numeric parameters. |
| `default` | number | within `range` | Initial value when the algorithm loads. |
| `ui_mapping` | string | `slider` · `knob` · `toggle` · `dropdown` | Suggested frontend control widget. |
| `unit` | string \| null | `Hz` · `ms` · `dB` · `bits` · `null` | Display unit label. |

---

## 6. Error Responses

All 4xx and 5xx responses share a consistent JSON envelope:

```json
{
  "error":   "unprocessable_entity",
  "message": "Generated code failed validation and could not be corrected.",
  "detail":  ["BIBO stability violation: feedback gain >= 1.0"]
}
```

| Field | Type | Description |
|---|---|---|
| `error` | string | Machine-readable error code. |
| `message` | string | Human-readable explanation. |
| `detail` | any \| null | Additional context (list of validation errors, stack trace in debug mode). `null` in production. |

#### Common error codes

| Code | HTTP status | Trigger |
|---|---|---|
| `validation_error` | 422 | Request body failed Pydantic validation. |
| `unprocessable_entity` | 422 | Generated code failed static validation after all correction attempts. |
| `llm_provider_error` | 502 | LLM API returned an error or timed out. |
| `wasm_compilation_error` | 500 | All compiler backends failed. |
| `internal_server_error` | 500 | Unhandled exception. |

---

## 7. Rate Limits & Timeouts

| Setting | Default | Config key |
|---|---|---|
| WASM compilation timeout | 5 000 ms | `wasm_compile_timeout_ms` |
| LLM max tokens | 2 048 | `llm_max_tokens` |
| LLM temperature | 0.2 | `llm_temperature` |
| Cache similarity threshold | 0.92 | `cache_similarity_threshold` |
| Max cache entries | 1 000 | `cache_max_entries` |

HTTP-level rate limiting is not implemented in the application layer. Add it at the reverse proxy (nginx `limit_req`, Cloudflare rate limiting, or an API gateway) before exposing the service publicly.

---

## 8. Quick-Start Examples

### cURL — Generate a DSP algorithm

```bash
curl -X POST http://localhost:8000/api/v1/generate/dsp \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A warm, intimate reverb like a small wooden cabin",
    "sample_rate": 44100,
    "max_parameters": 6
  }'
```

### Python — Generate a shader

```python
import httpx

response = httpx.post(
    "http://localhost:8000/api/v1/generate/shader",
    json={
        "prompt":          "Frosted glass with chromatic aberration",
        "target_platform": "webgpu",
        "max_parameters":  4,
    },
    timeout=60.0,
)
response.raise_for_status()
data = response.json()
print(data["shader_code"][:200])
print(f"Parameters: {[p['name'] for p in data['parameters']]}")
```

### JavaScript — Cache probe before generation

```javascript
const probe = await fetch("/api/v1/cache/lookup/dsp", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ prompt: "spring reverb", generation_type: "dsp" }),
});
const { hit, similarity_score } = await probe.json();

if (hit) {
  console.log(`Cache hit — similarity: ${(similarity_score * 100).toFixed(1)}%`);
} else {
  // proceed to generation
}
```

---

*Interactive API docs are available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc) when the server is running.*
