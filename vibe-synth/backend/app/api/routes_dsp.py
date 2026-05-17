# =============================================================================
# File: backend/app/api/routes_dsp.py
# Purpose: REST endpoints for DSP algorithm generation. Orchestrates the full
#          pipeline: cache lookup → prompt assembly → LLM generation →
#          static validation → self-correction (if needed) → WASM compilation
#          → cache store → response. This is the primary work-file route that
#          the frontend VibeInput calls when a user submits an audio Vibe.
# =============================================================================

import time
import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_service_bundle, ServiceBundle
from app.models.request_models import GenerateDSPRequest, CacheLookupRequest
from app.models.response_models import (
    GenerateDSPResponse,
    CacheLookupResponse,
    GenerationStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["dsp"])

# Maximum self-correction attempts before returning a hard error
_MAX_CORRECTION_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# POST /api/v1/generate/dsp
# ---------------------------------------------------------------------------

@router.post(
    "/generate/dsp",
    response_model=GenerateDSPResponse,
    summary="Generate a DSP algorithm from a Vibe description",
    description=(
        "Accepts a natural language Vibe description and returns a compiled "
        "Faust DSP algorithm as a WASM module ID, along with a parameter "
        "manifest for automatic UI control generation."
    ),
)
async def generate_dsp(
    payload: GenerateDSPRequest,
    bundle:  ServiceBundle = Depends(get_service_bundle),
) -> GenerateDSPResponse:
    """
    Full DSP generation pipeline:

    1. Check the vector cache for a semantically similar prior result.
    2. On a cache miss, build prompts and call the LLM.
    3. Validate the generated Faust code for safety and correctness.
    4. If validation fails, invoke LLM self-correction (up to 2 rounds).
    5. Send validated code to the WASM compilation sandbox.
    6. Store the result in the cache.
    7. Return the WASM module ID and parameter manifest to the caller.
    """
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — cache lookup
    # ------------------------------------------------------------------
    if not payload.force_refresh:
        hit, score, cached = await bundle.cache_service.lookup(
            prompt=payload.prompt,
            generation_type="dsp",
        )
        if hit and cached:
            logger.info(
                "DSP cache hit | score=%.4f prompt='%s'", score, payload.prompt[:60]
            )
            return GenerateDSPResponse(
                status=GenerationStatus.cache_hit,
                compile_time_ms=int((time.perf_counter() - t0) * 1000),
                cache_hit=True,
                faust_code=cached.get("faust_code", ""),
                wasm_module_id=cached.get("wasm_module_id", ""),
                parameters=cached.get("parameters", []),
            )

    # ------------------------------------------------------------------
    # Step 2 — prompt assembly + LLM generation
    # ------------------------------------------------------------------
    system_prompt, user_prompt = bundle.prompt_builder.build_dsp_prompts(payload)

    try:
        raw_output = await bundle.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        logger.error("LLM generation failed for DSP request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM provider error: {exc}",
        )

    # ------------------------------------------------------------------
    # Step 3 — static validation
    # ------------------------------------------------------------------
    validation = bundle.validator.validate_dsp(raw_output)
    correction_attempts = 0

    # ------------------------------------------------------------------
    # Step 4 — self-correction loop
    # ------------------------------------------------------------------
    if not validation.passed:
        logger.warning(
            "DSP validation failed — entering self-correction. Errors: %s",
            validation.errors,
        )
        correction_system = bundle.prompt_builder.build_correction_context("dsp")
        try:
            corrected_output, correction_attempts = await bundle.llm_client.self_correct(
                system_prompt=correction_system,
                original_user_prompt=user_prompt,
                failed_code=validation.clean_code,
                error_message="\n".join(validation.errors),
                max_attempts=_MAX_CORRECTION_ATTEMPTS,
            )
        except Exception as exc:
            logger.error("Self-correction exhausted for DSP request: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Generated code failed validation and could not be corrected: "
                    f"{validation.errors}"
                ),
            )
        # Re-validate corrected output
        validation = bundle.validator.validate_dsp(corrected_output)
        if not validation.passed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Corrected code still failed validation: {validation.errors}"
                ),
            )

    # ------------------------------------------------------------------
    # Step 5 — WASM compilation (delegated to compiler module)
    # ------------------------------------------------------------------
    try:
        from compiler.wasm_compiler import WasmCompiler
        compiler = WasmCompiler(settings=bundle.settings)
        wasm_module_id = await compiler.compile_faust(
            faust_code=validation.clean_code,
            timeout_ms=bundle.settings.wasm_compile_timeout_ms,
        )
    except Exception as exc:
        logger.error("WASM compilation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"WASM compilation error: {exc}",
        )

    compile_time_ms = int((time.perf_counter() - t0) * 1000)

    # ------------------------------------------------------------------
    # Step 6 — cache store
    # ------------------------------------------------------------------
    result_dict = {
        "faust_code":     validation.clean_code,
        "wasm_module_id": wasm_module_id,
        "parameters":     validation.parameters,
    }
    await bundle.cache_service.store(
        prompt=payload.prompt,
        generation_type="dsp",
        result=result_dict,
    )

    # ------------------------------------------------------------------
    # Step 7 — return response
    # ------------------------------------------------------------------
    final_status = (
        GenerationStatus.self_corrected
        if correction_attempts > 0
        else GenerationStatus.success
    )

    logger.info(
        "DSP generation complete | status=%s compile_ms=%d corrections=%d",
        final_status, compile_time_ms, correction_attempts,
    )

    return GenerateDSPResponse(
        status=final_status,
        compile_time_ms=compile_time_ms,
        cache_hit=False,
        faust_code=validation.clean_code,
        wasm_module_id=wasm_module_id,
        parameters=validation.parameters,
        self_correction_attempts=correction_attempts,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/cache/lookup  —  cache probe (DSP)
# ---------------------------------------------------------------------------

@router.post(
    "/cache/lookup/dsp",
    response_model=CacheLookupResponse,
    summary="Probe the DSP vector cache without generating",
    description=(
        "Check whether a semantically similar DSP prompt already exists in "
        "the cache. Useful for client-side prefetch decisions."
    ),
)
async def lookup_dsp_cache(
    payload: CacheLookupRequest,
    bundle:  ServiceBundle = Depends(get_service_bundle),
) -> CacheLookupResponse:
    """
    Thin wrapper around CacheService.lookup() exposed as a standalone endpoint
    so callers can check cache status before committing to a generation call.
    """
    hit, score, cached = await bundle.cache_service.lookup(
        prompt=payload.prompt,
        generation_type="dsp",
    )
    return CacheLookupResponse(
        hit=hit,
        similarity_score=round(score, 4),
        cached_result=cached,
    )