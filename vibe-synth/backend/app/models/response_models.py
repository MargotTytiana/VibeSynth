# =============================================================================
# File: backend/app/models/response_models.py
# Purpose: Pydantic response schemas returned by all API endpoints. Defines
#          the exact JSON shape clients receive for DSP generation, shader
#          generation, cache lookups, and error payloads.
# =============================================================================

from pydantic import BaseModel, Field
from enum import Enum
from typing import Any


class GenerationStatus(str, Enum):
    success        = "success"
    cache_hit      = "cache_hit"
    self_corrected = "self_corrected"   # LLM fixed its own compilation error
    error          = "error"


class UIMapping(str, Enum):
    slider   = "slider"
    knob     = "knob"
    toggle   = "toggle"
    dropdown = "dropdown"


class ParameterMeta(BaseModel):
    """
    Describes a single runtime-controllable parameter exposed by the generated
    algorithm. The frontend uses this manifest to auto-render the control panel
    without any hard-coded UI logic.
    """

    name: str = Field(..., description="Snake-case identifier used in the algorithm code.")
    type: str = Field(..., description="Value type: 'float' | 'int' | 'bool'.")
    range: list[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="[min, max] bounds for numeric parameters.",
    )
    default: float = Field(..., description="Initial value when the algorithm loads.")
    ui_mapping: UIMapping = Field(
        default=UIMapping.slider,
        description="Suggested control widget for the frontend to render.",
    )
    unit: str | None = Field(
        default=None,
        description="Human-readable unit label, e.g. 'Hz', 'ms', 'dB'.",
    )


class GenerateDSPResponse(BaseModel):
    """
    Response body for POST /api/v1/generate/dsp.
    Contains the generated Faust source, its WASM-ready compilation artefact
    reference, and the parameter manifest for UI binding.
    """

    status: GenerationStatus
    compile_time_ms: int = Field(..., description="Wall-clock time for the full pipeline in ms.")
    cache_hit: bool = Field(default=False, description="True if result was served from cache.")
    faust_code: str = Field(..., description="Generated Faust DSP source code.")
    wasm_module_id: str = Field(
        ...,
        description="Unique ID referencing the compiled WASM module in the sandbox registry.",
    )
    parameters: list[ParameterMeta] = Field(
        default_factory=list,
        description="Parameter manifest for dynamic UI generation.",
    )
    self_correction_attempts: int = Field(
        default=0,
        description="Number of LLM self-correction rounds needed before successful compilation.",
    )


class GenerateShaderResponse(BaseModel):
    """
    Response body for POST /api/v1/generate/shader.
    Contains the generated WGSL/GLSL source and its parameter manifest.
    """

    status: GenerationStatus
    compile_time_ms: int
    cache_hit: bool = False
    shader_code: str = Field(..., description="Generated WGSL or GLSL fragment shader source.")
    target_platform: str = Field(..., description="'webgpu' or 'webgl'.")
    parameters: list[ParameterMeta] = Field(default_factory=list)
    self_correction_attempts: int = 0


class CacheLookupResponse(BaseModel):
    """
    Response body for POST /api/v1/cache/lookup.
    Reports whether a semantically similar cached result exists and, if so,
    returns its similarity score and the cached payload.
    """

    hit: bool
    similarity_score: float | None = Field(
        default=None,
        description="Cosine similarity between the query prompt and the cached entry (0–1).",
    )
    cached_result: dict[str, Any] | None = Field(
        default=None,
        description="The full cached response payload if a hit was found.",
    )


class HealthResponse(BaseModel):
    """
    Response body for GET /api/v1/health.
    Used by load balancers and monitoring to confirm the service is running.
    """

    status: str = "ok"
    version: str = "0.1.0"


class ErrorDetail(BaseModel):
    """
    Standardised error envelope returned for all 4xx / 5xx responses.
    Wraps FastAPI's HTTPException with a consistent JSON shape.
    """

    error: str = Field(..., description="Short machine-readable error code.")
    message: str = Field(..., description="Human-readable explanation of the error.")
    detail: Any | None = None