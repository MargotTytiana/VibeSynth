# =============================================================================
# File: backend/app/dependencies.py
# Purpose: FastAPI dependency-injection providers. Every shared resource —
#          the LLM client, cache service, and validator — is instantiated once
#          and injected into route handlers via Depends(). This keeps routes
#          thin and makes unit-testing straightforward (swap the dependency).
# =============================================================================

from functools import lru_cache
from fastapi import Depends

from app.config import Settings, get_settings


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm_client(settings: Settings = Depends(get_settings)):
    """
    Return a cached LLMClient singleton.

    The client is constructed once on first request and reused for the
    lifetime of the process. Importing lazily here (inside the function)
    avoids a circular-import cycle because llm_client.py itself imports
    from config.py, not from dependencies.py.
    """
    from app.services.llm_client import LLMClient
    return LLMClient(settings=settings)


# ---------------------------------------------------------------------------
# CacheService
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_cache_service(settings: Settings = Depends(get_settings)):
    """
    Return a cached CacheService singleton backed by the on-disk vector index
    path declared in settings.cache_index_path.
    """
    from app.services.cache_service import CacheService
    return CacheService(settings=settings)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_validator(settings: Settings = Depends(get_settings)):
    """
    Return a cached Validator singleton used to static-check generated code
    before it is handed to the WASM compilation sandbox.
    """
    from app.services.validator import Validator
    return Validator(settings=settings)


# ---------------------------------------------------------------------------
# Convenience bundle — injects all three at once for routes that need them
# ---------------------------------------------------------------------------

class ServiceBundle:
    """
    Aggregates the three core services into a single injectable object.
    Route handlers that need all services can declare one dependency instead
    of three:

        async def my_route(bundle: ServiceBundle = Depends(get_service_bundle)):
            bundle.llm_client.generate(...)
    """

    def __init__(
        self,
        settings: Settings = Depends(get_settings),
        llm_client=Depends(get_llm_client),
        cache_service=Depends(get_cache_service),
        validator=Depends(get_validator),
    ):
        self.settings      = settings
        self.llm_client    = llm_client
        self.cache_service = cache_service
        self.validator     = validator


def get_service_bundle(bundle: ServiceBundle = Depends(ServiceBundle)) -> ServiceBundle:
    """
    FastAPI-compatible provider for ServiceBundle.
    Usage in a route:

        @router.post("/generate/dsp")
        async def generate_dsp(
            payload: GenerateDSPRequest,
            bundle: ServiceBundle = Depends(get_service_bundle),
        ):
            ...
    """
    return bundle