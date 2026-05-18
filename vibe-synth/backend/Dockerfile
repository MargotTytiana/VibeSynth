# =============================================================================
# File: backend/Dockerfile
# Purpose: Multi-stage Docker image for the Vibe-Synth backend service.
#          Stage 1 (builder) installs all dependencies including dev tools into
#          a virtual environment. Stage 2 (runtime) copies only the venv and
#          application code, producing a lean production image without build
#          tooling or test packages. Final image runs as a non-root user.
# =============================================================================


# ---------------------------------------------------------------------------
# Stage 1 — builder
# Installs Python dependencies into an isolated virtual environment so
# nothing leaks into the system Python in the runtime stage.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Prevent .pyc files and enable unbuffered stdout/stderr for clean logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install build tools needed to compile any C-extension wheels (e.g. wasmtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies first (leverages Docker layer cache —
# this layer only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2 — runtime
# Copies only the venv and application source; no build tools included.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Default values — override via docker run -e or docker-compose environment:
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_DEBUG=false \
    LOG_LEVEL=INFO

# Runtime system libraries only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy application source
COPY backend/app/       ./app/
COPY compiler/          ./compiler/
COPY generation/        ./generation/
COPY cache/             ./cache/

# Create a non-root user and hand over ownership
RUN useradd --no-create-home --shell /bin/false vibesynth \
    && chown -R vibesynth:vibesynth /app

USER vibesynth

# Expose the API port (informational — actual binding set by API_PORT env var)
EXPOSE 8000

# Health check — Docker marks the container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" \
    || exit 1

# Default command — Uvicorn with worker count auto-set to 1 for predictable
# single-process behaviour. Scale horizontally via container replicas rather
# than multiple workers per container (simpler cache state management).
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]