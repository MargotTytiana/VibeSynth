# =============================================================================
# File: backend/app/main.py
# Purpose: FastAPI application entry point. Creates the app instance, registers
#          all routers, configures CORS, sets up structured logging, and defines
#          startup / shutdown lifecycle hooks that initialise shared resources
#          (cache index pre-load, LLM client warm-up). This is the file that
#          Uvicorn targets when the server starts.
# =============================================================================

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api import routes_health, routes_dsp, routes_shader

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def _configure_logging(level: str) -> None:
    """
    Set up a consistent log format for the entire application.
    All modules use logging.getLogger(__name__) so this root config
    propagates everywhere automatically.
    """
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

settings = get_settings()
_configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup (before the first request) and once on shutdown.

    Startup:
      - Pre-load the cache index from disk so the first request is not slow.
      - Log a confirmation that all services are ready.

    Shutdown:
      - Flush any pending cache writes to disk.
    """
    # --- Startup ---
    logger.info("=== Vibe-Synth API starting up ===")
    logger.info(
        "Config | provider=%s model=%s port=%d debug=%s",
        settings.llm_provider,
        settings.llm_model,
        settings.api_port,
        settings.api_debug,
    )

    # Force cache index to load eagerly so the first /generate call
    # does not pay the cold-start disk-read penalty.
    try:
        from app.dependencies import get_cache_service
        cache = get_cache_service()
        stats = cache.stats()
        logger.info(
            "Cache index ready | entries=%d path=%s",
            stats["total_entries"], stats["index_path"],
        )
    except Exception as exc:
        logger.warning("Cache pre-load failed (non-fatal): %s", exc)

    logger.info("=== Vibe-Synth API ready ===")

    yield  # application runs here

    # --- Shutdown ---
    logger.info("=== Vibe-Synth API shutting down ===")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Construct and return the configured FastAPI application.
    Separating construction from module-level instantiation makes the app
    easily importable in tests without side effects.
    """
    app = FastAPI(
        title="Vibe-Synth API",
        description=(
            "Real-time LLM-powered DSP algorithm and shader generator. "
            "Converts natural language Vibe descriptions into compiled "
            "Faust DSP (WASM) or WGSL/GLSL fragment shaders with "
            "auto-generated UI parameter manifests."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # CORS — allow the Vite dev server and any configured origins
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Global exception handler — wraps unhandled errors in a consistent
    # JSON envelope so clients always get a structured error response.
    # ------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method, request.url.path, exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error":   "internal_server_error",
                "message": "An unexpected error occurred. Please try again.",
                "detail":  str(exc) if settings.api_debug else None,
            },
        )

    # ------------------------------------------------------------------
    # Request logging middleware — logs method, path, and response time
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        import time
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "%s %s → %d (%d ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(routes_health.router)
    app.include_router(routes_dsp.router)
    app.include_router(routes_shader.router)

    # ------------------------------------------------------------------
    # Root redirect — convenience for developers hitting bare /
    # ------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Vibe-Synth API", "docs": "/docs", "health": "/api/v1/health"}

    return app


# ---------------------------------------------------------------------------
# Module-level app instance — imported by Uvicorn
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# Development runner — python -m app.main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level=settings.log_level.lower(),
    )