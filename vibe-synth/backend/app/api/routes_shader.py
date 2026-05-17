# =============================================================================
# File: backend/app/api/routes_shader.py
# Purpose: REST endpoints for shader code generation. Mirrors the DSP pipeline
#          in routes_dsp.py but targets WGSL (WebGPU) or GLSL (WebGL) fragment
#          shaders. Orchestrates: cache lookup → prompt assembly → LLM
#          generation → static validation → self-correction → response.
#          The frontend ShaderPreview.ts calls POST /api/v1/generate/shader
#          whenever a user submits a visual Vibe description.
# =============================================================================

import time
import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_service_bundle, ServiceBundle
from app.models.request_models import GenerateShaderRequest, CacheLookupRequest
from app.models.response_models import (
    GenerateShaderResponse,
    CacheLookupResponse,
    GenerationStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["shader"])

_MAX_CORRECTION_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# POST /api/v1/generate/shader
# ---------------------------------------------------------------------------

@router.post(
    "/generate/shader",
    response_model=GenerateShaderResponse,
    summary="Generate a fragment shader from a Vibe description",
    description=(
        "Accepts a natural language Vibe description and returns a WGSL or "
        "GLSL fragment shader, along with a parameter manifest for automatic "
        "UI control generation. Target language is determined by target_platform."
    ),
)
async def generate_shader(
    payload: GenerateShaderRequest,
    bundle:  ServiceBundle = Depends(get_service_bundle),
) -> GenerateShaderResponse:
    """
    Full shader generation pipeline:

    1. Check the vector cache for a semantically similar prior result.
    2. On a cache miss, build prompts and call the LLM.
    3. Validate the generated shader code for safety and correctness.
    4. If validation fails, invoke LLM self-correction (up to 2 rounds).
    5. Store the validated result in the cache.
    6. Return the shader source and parameter manifest to the caller.

    Note: Unlike the DSP pipeline, shaders are NOT compiled to WASM server-side.
    Compilation happens in the browser via the WebGPU / WebGL runtime.
    The server only validates the source and returns it as a string.
    """
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — cache lookup
    # ------------------------------------------------------------------
    if not payload.force_refresh:
        hit, score, cached = await bundle.cache_service.lookup(
            prompt=payload.prompt,
            generation_type="shader",
        )
        if hit and cached:
            logger.info(
                "Shader cache hit | score=%.4f prompt='%s'", score, payload.prompt[:60]
            )
            return GenerateShaderResponse(
                status=GenerationStatus.cache_hit,
                compile_time_ms=int((time.perf_counter() - t0) * 1000),
                cache_hit=True,
                shader_code=cached.get("shader_code", ""),
                target_platform=cached.get("target_platform", payload.target_platform.value),
                parameters=cached.get("parameters", []),
            )

    # ------------------------------------------------------------------
    # Step 2 — prompt assembly + LLM generation
    # ------------------------------------------------------------------
    system_prompt, user_prompt = bundle.prompt_builder.build_shader_prompts(payload)

    try:
        raw_output = await bundle.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        logger.error("LLM generation failed for shader request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM provider error: {exc}",
        )

    # ------------------------------------------------------------------
    # Step 3 — static validation
    # ------------------------------------------------------------------
    validation = bundle.validator.validate_shader(
        raw_output=raw_output,
        target_platform=payload.target_platform.value,
    )
    correction_attempts = 0

    # ------------------------------------------------------------------
    # Step 4 — self-correction loop
    # ------------------------------------------------------------------
    if not validation.passed:
        logger.warning(
            "Shader validation failed — entering self-correction. Errors: %s",
            validation.errors,
        )
        correction_system = bundle.prompt_builder.build_correction_context("shader")
        try:
            corrected_output, correction_attempts = await bundle.llm_client.self_correct(
                system_prompt=correction_system,
                original_user_prompt=user_prompt,
                failed_code=validation.clean_code,
                error_message="\n".join(validation.errors),
                max_attempts=_MAX_CORRECTION_ATTEMPTS,
            )
        except Exception as exc:
            logger.error("Self-correction exhausted for shader request: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Generated shader failed validation and could not be corrected: "
                    f"{validation.errors}"
                ),
            )

        # Re-validate the corrected output
        validation = bundle.validator.validate_shader(
            raw_output=corrected_output,
            target_platform=payload.target_platform.value,
        )
        if not validation.passed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Corrected shader still failed validation: {validation.errors}",
            )

    compile_time_ms = int((time.perf_counter() - t0) * 1000)

    # ------------------------------------------------------------------
    # Step 5 — cache store
    # ------------------------------------------------------------------
    result_dict = {
        "shader_code":     validation.clean_code,
        "target_platform": payload.target_platform.value,
        "parameters":      validation.parameters,
    }
    await bundle.cache_service.store(
        prompt=payload.prompt,
        generation_type="shader",
        result=result_dict,
    )

    # ------------------------------------------------------------------
    # Step 6 — return response
    # ------------------------------------------------------------------
    final_status = (
        GenerationStatus.self_corrected
        if correction_attempts > 0
        else GenerationStatus.success
    )

    logger.info(
        "Shader generation complete | status=%s platform=%s compile_ms=%d corrections=%d",
        final_status,
        payload.target_platform.value,
        compile_time_ms,
        correction_attempts,
    )

    return GenerateShaderResponse(
        status=final_status,
        compile_time_ms=compile_time_ms,
        cache_hit=False,
        shader_code=validation.clean_code,
        target_platform=payload.target_platform.value,
        parameters=validation.parameters,
        self_correction_attempts=correction_attempts,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/cache/lookup/shader  —  cache probe (shader)
# ---------------------------------------------------------------------------

@router.post(
    "/cache/lookup/shader",
    response_model=CacheLookupResponse,
    summary="Probe the shader vector cache without generating",
    description=(
        "Check whether a semantically similar shader prompt already exists in "
        "the cache. Useful for client-side prefetch and UX indicators."
    ),
)
async def lookup_shader_cache(
    payload: CacheLookupRequest,
    bundle:  ServiceBundle = Depends(get_service_bundle),
) -> CacheLookupResponse:
    """
    Exposes CacheService.lookup() for shader entries as a standalone endpoint.
    The frontend can call this on keystroke to show a 'cached result available'
    indicator before the user commits to a full generation request.
    """
    hit, score, cached = await bundle.cache_service.lookup(
        prompt=payload.prompt,
        generation_type="shader",
    )
    return CacheLookupResponse(
        hit=hit,
        similarity_score=round(score, 4),
        cached_result=cached,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/generate/shader/platforms  —  supported platforms
# ---------------------------------------------------------------------------

@router.get(
    "/generate/shader/platforms",
    summary="List supported shader target platforms",
    description="Returns the available compilation targets and their output languages.",
)
async def list_platforms() -> dict:
    """
    Informational endpoint consumed by the frontend VibeInput platform selector.
    No auth or heavy computation — returns a static manifest.
    """
    return {
        "platforms": [
            {
                "id":       "webgpu",
                "language": "WGSL",
                "description": "WebGPU shading language — modern, preferred for Chrome/Edge.",
            },
            {
                "id":       "webgl",
                "language": "GLSL ES 3.0",
                "description": "OpenGL ES shading language — wider browser compatibility.",
            },
        ]
    }