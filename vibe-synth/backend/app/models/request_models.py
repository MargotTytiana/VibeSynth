# =============================================================================
# File: backend/app/models/request_models.py
# Purpose: Pydantic request schemas for all API endpoints. Defines the shape
#          and validation rules for incoming JSON payloads — DSP generation,
#          shader generation, and cache lookup requests.
# =============================================================================

from pydantic import BaseModel, Field, field_validator
from enum import Enum


class TargetPlatform(str, Enum):
    webgpu = "webgpu"
    webgl  = "webgl"


class InputType(str, Enum):
    text_only    = "text_only"
    audio_stream = "audio_stream"
    video_stream = "video_stream"


class GenerateDSPRequest(BaseModel):
    """
    Payload for POST /api/v1/generate/dsp.
    The caller describes a desired sound character in natural language;
    the engine maps it to a Faust DSP algorithm.
    """

    prompt: str = Field(
        ...,
        min_length=3,
        max_length=512,
        description="Natural language description of the desired audio effect.",
        examples=["Make the sound feel like being in an empty, freezing ice cave"],
    )
    input_type: InputType = Field(
        default=InputType.audio_stream,
        description="Whether to process an audio stream or generate from text only.",
    )
    sample_rate: int = Field(
        default=44100,
        ge=8000,
        le=192000,
        description="Target sample rate in Hz for the generated DSP algorithm.",
    )
    max_parameters: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Maximum number of exposed UI-controllable parameters.",
    )
    force_refresh: bool = Field(
        default=False,
        description="If True, bypass the vector cache and request a fresh generation.",
    )

    @field_validator("prompt")
    @classmethod
    def strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt must not be blank after stripping whitespace")
        return v


class GenerateShaderRequest(BaseModel):
    """
    Payload for POST /api/v1/generate/shader.
    The caller describes a visual effect; the engine produces WGSL or GLSL
    fragment shader source code with an accompanying parameter manifest.
    """

    prompt: str = Field(
        ...,
        min_length=3,
        max_length=512,
        description="Natural language description of the desired visual effect.",
        examples=["Frosted glass distortion with cyberpunk chromatic aberration"],
    )
    input_type: InputType = Field(
        default=InputType.video_stream,
        description="The type of input the shader will process at runtime.",
    )
    target_platform: TargetPlatform = Field(
        default=TargetPlatform.webgpu,
        description="Compilation target: 'webgpu' emits WGSL, 'webgl' emits GLSL.",
    )
    max_parameters: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Maximum number of exposed UI-controllable parameters.",
    )
    force_refresh: bool = Field(
        default=False,
        description="If True, bypass the vector cache and request a fresh generation.",
    )

    @field_validator("prompt")
    @classmethod
    def strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt must not be blank after stripping whitespace")
        return v


class CacheLookupRequest(BaseModel):
    """
    Payload for POST /api/v1/cache/lookup.
    Used internally (and by tests) to query whether a semantically similar
    prompt already has a compiled result in the vector cache.
    """

    prompt: str = Field(..., min_length=3, max_length=512)
    generation_type: str = Field(
        ...,
        pattern="^(dsp|shader)$",
        description="Must be 'dsp' or 'shader'.",
    )