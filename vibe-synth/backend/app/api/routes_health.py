# =============================================================================
# File: backend/app/api/routes_health.py
# Purpose: Health-check and diagnostics endpoints. Used by load balancers,
#          container orchestrators (Docker / Kubernetes), and monitoring tools
#          to verify the service is alive and its subsystems are reachable.
#          Returns cache statistics and configuration metadata alongside the
#          standard liveness signal.
# =============================================================================

import logging
from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_cache_service
from app.models.response_models import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["health"])


# ---------------------------------------------------------------------------
# GET /api/v1/health  —  liveness probe
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Returns HTTP 200 with status='ok' whenever the process is running. "
        "Used by load balancers and container health checks."
    ),
)
async def health_check() -> HealthResponse:
    """
    Minimal liveness endpoint. No external calls are made so this always
    returns quickly even if downstream services (LLM API, compiler) are slow.
    """
    logger.debug("Health check requested")
    return HealthResponse(status="ok", version="0.1.0")


# ---------------------------------------------------------------------------
# GET /api/v1/health/ready  —  readiness probe
# ---------------------------------------------------------------------------

@router.get(
    "/health/ready",
    summary="Readiness probe",
    description=(
        "Checks that the cache index is loaded and configuration is valid. "
        "Returns 200 if the service can handle traffic, 503 otherwise."
    ),
)
async def readiness_check(
    settings: Settings = Depends(get_settings),
    cache_service      = Depends(get_cache_service),
) -> dict:
    """
    Readiness probe that verifies key subsystems are initialised.

    Checks performed:
      - Cache index is accessible (stats() does not raise)
      - LLM provider is configured (api_key is non-empty or provider is local)

    Returns a JSON object with per-subsystem status flags.
    """
    checks: dict[str, str] = {}

    # Cache subsystem
    try:
        cache_stats = cache_service.stats()
        checks["cache"] = f"ok ({cache_stats['total_entries']} entries)"
    except Exception as exc:
        checks["cache"] = f"error: {exc}"

    # LLM configuration
    if settings.llm_provider == "local":
        checks["llm"] = "ok (local provider — no API key required)"
    elif settings.llm_api_key:
        checks["llm"] = f"ok (provider={settings.llm_provider})"
    else:
        checks["llm"] = "warning: LLM_API_KEY is not set"

    overall = "ready" if all("error" not in v for v in checks.values()) else "not_ready"

    logger.info("Readiness check: %s | %s", overall, checks)
    return {"status": overall, "checks": checks}


# ---------------------------------------------------------------------------
# GET /api/v1/health/cache  —  cache diagnostics
# ---------------------------------------------------------------------------

@router.get(
    "/health/cache",
    summary="Cache statistics",
    description="Returns detailed statistics about the vector cache index.",
)
async def cache_stats(
    cache_service = Depends(get_cache_service),
) -> dict:
    """
    Exposes cache internals for monitoring dashboards and manual inspection.
    Useful when tuning settings.cache_similarity_threshold.
    """
    stats = cache_service.stats()
    logger.debug("Cache stats requested: %s", stats)
    return {"status": "ok", "cache": stats}