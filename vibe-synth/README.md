# Vibe-Synth

**Real-time LLM-powered DSP algorithm and shader synthesis engine.**

Describe a sound or visual character in plain English — a "Vibe" — and Vibe-Synth generates a compiled audio DSP algorithm or a GPU fragment shader ready to run in your browser. Instead of producing static `.wav` or `.png` files, it generates *rules*: white-box, parameter-exposed algorithms you can tweak in real time.

```
"Make the sound feel like an empty, freezing ice cave"
        ↓
  Faust reverb algorithm → WebAssembly → AudioWorklet @ 44.1 kHz

"Frosted glass distortion with cyberpunk chromatic aberration"
        ↓
  WGSL fragment shader → WebGPU → 60 fps render loop
```

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Development Guide](#development-guide)
- [Running Tests](#running-tests)
- [Experiments](#experiments)
- [Contributing](#contributing)

---

## Features

- **DSP path** — Generates Faust DSP code compiled to WebAssembly via AudioWorklet; supports reverb, delay, filters, saturation, chorus, tremolo, bitcrusher, compressor, and more.
- **Shader path** — Generates WGSL (WebGPU) or GLSL ES 3.0 (WebGL) fragment shaders; supports chromatic aberration, glitch, blur, grain, vignette, edge detection, pixelate, ripple, and more.
- **Semantic vector cache** — Cosine-similarity cache avoids redundant LLM calls for near-duplicate Vibes; typical cache hit returns in < 25 ms vs ~2 s for fresh generation.
- **Self-correction loop** — Static validator catches common LLM mistakes (unstable feedback, missing entry points, dynamic shader loop bounds); LLM auto-corrects up to 2 rounds before returning an error.
- **Auto-generated parameter panel** — Every algorithm returns a typed parameter manifest; the frontend renders sliders, knobs, toggles, and dropdowns automatically — no UI code required per algorithm.
- **Multi-provider LLM support** — OpenAI, Anthropic, and any local OpenAI-compatible endpoint (Ollama, LM Studio, vLLM).
- **BIBO stability checker** — Heuristic analysis catches feedback gain ≥ 1.0, non-productive recursion, and missing damping in reverb paths before compilation.

---

## Architecture

```
Browser                    FastAPI Backend              External
──────────────────────     ───────────────────────     ─────────────
VibeInput.ts               routes_dsp.py               OpenAI
AudioEngine.ts      ←───→  routes_shader.py     ←───→  Anthropic
ShaderPreview.ts           services/                    Ollama
ParamPanel.ts              │ LLMClient
WasmLoader.ts              │ PromptBuilder
                           │ CacheService           cache/
                           │ Validator              │ cache_index.json
                           generation/
                           │ ASTBuilder             compiler/
                           │ FaustGenerator         │ wasm_compiler.py
                           │ WGSLGenerator          │ sandbox_runner.py
                           │ GLSLGenerator          │ compile_log.csv
```

Full documentation: [`docs/architecture.md`](docs/architecture.md)  
API reference: [`docs/api_reference.md`](docs/api_reference.md)  
DSP mathematics: [`docs/dsp_math_notes.pdf`](docs/dsp_math_notes.pdf)

---

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 20+
- An LLM API key (OpenAI or Anthropic) **or** Ollama running locally

### 1. Clone and configure

```bash
git clone https://github.com/your-org/vibe-synth.git
cd vibe-synth
cp .env.example .env
# Edit .env — set LLM_API_KEY and optionally LLM_PROVIDER / LLM_MODEL
```

### 2. Start the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
# API is now running at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
# Vite dev server at http://localhost:5173
```

### 4. Or use Docker Compose (all services at once)

```bash
cp .env.example .env   # fill in LLM_API_KEY
docker compose up --build
# backend  → http://localhost:8000
# frontend → http://localhost:5173
# ollama   → http://localhost:11434 (optional local LLM)
```

### 5. Try your first Vibe

```bash
curl -X POST http://localhost:8000/api/v1/generate/dsp \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A warm, intimate reverb like playing guitar in a wooden cabin"}'
```

---

## Configuration

All configuration is loaded from `.env` (copied from `.env.example`). Key settings:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` · `anthropic` · `local` |
| `LLM_MODEL` | `gpt-4o` | Model identifier for the chosen provider |
| `LLM_API_KEY` | *(empty)* | Provider API key |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature — keep low for code generation |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity cutoff for cache hits |
| `WASM_COMPILE_TIMEOUT_MS` | `5000` | Faust compiler timeout in milliseconds |
| `API_PORT` | `8000` | FastAPI server port |
| `API_DEBUG` | `false` | Enable Uvicorn auto-reload (dev only) |

See `.env.example` for the full list with descriptions.

---

## Project Structure

```
vibe-synth/
├── backend/
│   ├── app/
│   │   ├── api/            routes_dsp.py · routes_shader.py · routes_health.py
│   │   ├── models/         request_models.py · response_models.py
│   │   ├── services/       llm_client.py · prompt_builder.py · cache_service.py · validator.py
│   │   ├── config.py       centralised settings
│   │   ├── dependencies.py FastAPI DI providers
│   │   └── main.py         application entry point
│   ├── tests/              pytest unit tests
│   ├── requirements.txt
│   └── Dockerfile
│
├── generation/
│   ├── dsp/                ast_builder.py · faust_generator.py · bibo_checker.py · dsp_templates.json
│   ├── shader/             wgsl_generator.py · glsl_generator.py · shader_templates.json
│   └── prompts/            system_dsp.txt · system_shader.txt · few_shot_examples.json
│
├── compiler/               wasm_compiler.py · sandbox_runner.py · error_handler.py · compile_log.csv
│
├── cache/                  vector_store.py · similarity_search.py · cache_index.json
│
├── frontend/
│   └── src/                VibeInput.ts · AudioEngine.ts · ShaderPreview.ts · ParamPanel.ts · WasmLoader.ts
│
├── experiments/            eval notebooks · latency_benchmark.csv · cache_hit_rate.csv · model_comparison.xlsx
│
├── docs/                   architecture.md · api_reference.md · system_diagram.png · dsp_math_notes.pdf
│
├── .env.example
├── docker-compose.yml
└── README.md
```

---

## Development Guide

### Adding a new DSP archetype

1. Add an entry to `generation/dsp/dsp_templates.json` with a unique `id`, descriptive `tags`, a `faust_skeleton` with `{{param}}` placeholders, and `default_parameters`.
2. Verify `ASTBuilder` picks it up by running `ast_builder.inspect_ast("your test vibe")` in the Python REPL.
3. Add BIBO stability coverage in `generation/dsp/bibo_checker.py` if the archetype introduces new feedback patterns.
4. Run `pytest backend/tests/` to confirm no regressions.

### Adding a new shader archetype

1. Add an entry to `generation/shader/shader_templates.json` with both `wgsl_skeleton` and `glsl_skeleton`.
2. Ensure all for-loop bounds are compile-time integer literals in both skeletons.
3. Test the GLSL skeleton compiles by running it through the `Validator` directly.

### Swapping the LLM provider

Set `LLM_PROVIDER` and `LLM_MODEL` in `.env`. The `LLMClient` handles all three backends transparently. For a local Ollama endpoint, set `LLM_PROVIDER=local` and ensure Ollama is running on port 11434.

### Replacing the embedding model

The cache currently uses a deterministic SHA-256 hash fallback (`_simple_embed` in `app/services/cache_service.py`). To use a real embedding model, replace the body of `CacheService._embed()` with an async API call to `text-embedding-3-small` or a `sentence-transformers` model. The rest of the cache pipeline is embedding-agnostic.

---

## Running Tests

```bash
cd backend
pytest tests/ -v

# Run a specific test file
pytest tests/test_llm_client.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing
```

All tests run fully offline via `unittest.mock` — no API keys required.

---

## Experiments

The `experiments/` directory contains Jupyter notebooks for evaluating and tuning the pipeline:

| Notebook | Purpose |
|---|---|
| `eval_dsp_quality.ipynb` | DSP generation success rate, BIBO stability, compile times |
| `eval_shader_quality.ipynb` | Shader success rate by platform, loop safety audit |
| `prompt_ablation.ipynb` | AST hint impact, few-shot token overhead, cache threshold sweep |

Supporting data files:

| File | Contents |
|---|---|
| `latency_benchmark.csv` | Per-run latency breakdown (LLM, validation, compile, total) |
| `cache_hit_rate.csv` | Hourly cache hit rate by generation type and platform |
| `model_comparison.xlsx` | Side-by-side model comparison across success rate, latency, corrections |

---

## Contributing

1. Fork the repository and create a feature branch.
2. Follow the existing code style — all backend code is type-annotated; `ruff` and `mypy` are configured in `requirements.txt`.
3. Add or update tests in `backend/tests/` for any new service or generation logic.
4. Update `docs/architecture.md` if you change the module structure.
5. Open a pull request with a clear description of what changed and why.

---

## License

MIT — see `LICENSE` for details.