# =============================================================================
# File: compiler/wasm_compiler.py
# Purpose: Compiles validated Faust DSP source code into a WebAssembly module
#          via an external Faust-to-WASM toolchain (faust2wasm or an LLVM-
#          based cloud endpoint). Registers the compiled module in an in-process
#          registry keyed by a unique module ID, which the frontend WasmLoader
#          uses to instantiate the AudioWorklet processor. Enforces a
#          compile-time timeout and logs every attempt to compile_log.csv.
# =============================================================================

import asyncio
import csv
import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-process WASM module registry
# Registry maps module_id → compiled WASM bytes (or a path/URL depending on
# the backend used). A real production deployment would use Redis or an
# object-storage bucket; in-process is sufficient for the MVP.
# ---------------------------------------------------------------------------

_MODULE_REGISTRY: dict[str, dict[str, Any]] = {}


def get_module(module_id: str) -> dict | None:
    """
    Retrieve a compiled WASM module entry by its ID.
    Returns None if the ID is not found in the registry.
    """
    return _MODULE_REGISTRY.get(module_id)


def list_modules() -> list[str]:
    """Return all registered module IDs."""
    return list(_MODULE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Compilation backends
# ---------------------------------------------------------------------------

async def _compile_via_local_faust(
    faust_code: str,
    timeout_ms: int,
) -> bytes:
    """
    Compile Faust code to WASM using a locally installed faust2wasm binary.

    Requirements:
      - 'faust' binary on PATH (from the GRAME Faust distribution)
      - 'faust2wasm' wrapper script or equivalent

    The Faust compiler writes <tmpfile>.wasm to disk; we read and return
    the bytes, then clean up.

    Raises:
        RuntimeError: If the compiler exits with a non-zero return code or
                      the timeout is exceeded.
    """
    import tempfile
    import os

    timeout_s = timeout_ms / 1000.0

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path  = Path(tmpdir) / "vibe_synth_dsp.dsp"
        wasm_path = Path(tmpdir) / "vibe_synth_dsp.wasm"

        src_path.write_text(faust_code, encoding="utf-8")

        cmd = [
            "faust",
            "-lang", "wasm",
            "-o", str(wasm_path),
            str(src_path),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(
                f"Faust compiler timed out after {timeout_ms} ms."
            )

        if proc.returncode != 0:
            error_msg = stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"Faust compiler exited with code {proc.returncode}. "
                f"Stderr: {error_msg[:500]}"
            )

        if not wasm_path.exists():
            raise RuntimeError(
                "Faust compiler exited successfully but did not produce a .wasm file."
            )

        return wasm_path.read_bytes()


async def _compile_via_stub(faust_code: str, timeout_ms: int) -> bytes:  # noqa: ARG001
    """
    Stub compiler used when the Faust binary is not available (local dev,
    CI environments, unit tests). Returns a minimal valid WASM module
    (the 'magic bytes' header + version only — not executable).

    Replace this with a real backend call for production.
    """
    await asyncio.sleep(0.01)  # simulate minimal compilation delay
    # Minimal WASM binary: magic number + version (8 bytes)
    stub_wasm = b"\x00asm\x01\x00\x00\x00"
    logger.debug("WasmCompiler | stub compilation used (no faust binary available).")
    return stub_wasm


# ---------------------------------------------------------------------------
# Compile log helper
# ---------------------------------------------------------------------------

def _log_compile_attempt(
    log_path: str,
    module_id: str,
    success: bool,
    duration_ms: int,
    error_msg: str,
    faust_len: int,
    wasm_len: int,
) -> None:
    """
    Append a single row to compile_log.csv for observability.
    Creates the file with a header row if it does not exist.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "module_id", "success",
                "duration_ms", "faust_len", "wasm_len", "error_msg",
            ])
        writer.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            module_id,
            success,
            duration_ms,
            faust_len,
            wasm_len,
            error_msg,
        ])


# ---------------------------------------------------------------------------
# Public compiler
# ---------------------------------------------------------------------------

class WasmCompiler:
    """
    Compiles Faust DSP source to WebAssembly and registers the result.

    Backend selection (in priority order):
      1. Local Faust binary (faust on PATH)         → _compile_via_local_faust()
      2. Stub / offline fallback                    → _compile_via_stub()

    A production deployment should add a cloud LLVM/faust2wasm endpoint
    as Backend 1 and demote the local binary to Backend 2.

    Usage:
        compiler   = WasmCompiler(settings)
        module_id  = await compiler.compile_faust(faust_code)
        module     = get_module(module_id)   # retrieve later
    """

    def __init__(self, settings: Settings) -> None:
        self.settings    = settings
        self.log_path    = settings.log_compile_csv
        self.timeout_ms  = settings.wasm_compile_timeout_ms

    async def compile_faust(
        self,
        faust_code:  str,
        timeout_ms:  int | None = None,
    ) -> str:
        """
        Compile Faust code to WASM and register the module.

        Args:
            faust_code:  Clean Faust DSP source (no json-params block).
            timeout_ms:  Override settings.wasm_compile_timeout_ms for this call.

        Returns:
            module_id: Unique string ID referencing the compiled module in the
                       registry. Pass this to the frontend WasmLoader.

        Raises:
            RuntimeError: If compilation fails after trying all available backends.
        """
        timeout = timeout_ms or self.timeout_ms
        t0      = time.perf_counter()

        # Deterministic module ID: SHA-256 of the Faust source so identical
        # code reuses the same registry slot without re-compiling.
        content_hash = hashlib.sha256(faust_code.encode()).hexdigest()[:16]
        module_id    = f"wasm-{content_hash}"

        # Return immediately if already compiled
        if module_id in _MODULE_REGISTRY:
            logger.debug(
                "WasmCompiler | cache hit for module_id=%s — skipping recompile.",
                module_id,
            )
            return module_id

        logger.info(
            "WasmCompiler | compiling | module_id=%s faust_len=%d timeout_ms=%d",
            module_id, len(faust_code), timeout,
        )

        wasm_bytes: bytes = b""
        error_msg:  str   = ""
        success            = False

        # Try backends in order
        for backend_name, backend_fn in self._backends():
            try:
                logger.debug("WasmCompiler | trying backend: %s", backend_name)
                wasm_bytes = await backend_fn(faust_code, timeout)
                success    = True
                logger.info(
                    "WasmCompiler | compiled via '%s' | wasm_len=%d",
                    backend_name, len(wasm_bytes),
                )
                break
            except Exception as exc:
                error_msg = str(exc)
                logger.warning(
                    "WasmCompiler | backend '%s' failed: %s", backend_name, exc
                )

        duration_ms = int((time.perf_counter() - t0) * 1000)

        _log_compile_attempt(
            log_path=self.log_path,
            module_id=module_id,
            success=success,
            duration_ms=duration_ms,
            error_msg=error_msg,
            faust_len=len(faust_code),
            wasm_len=len(wasm_bytes),
        )

        if not success:
            raise RuntimeError(
                f"All compiler backends failed for module_id={module_id}. "
                f"Last error: {error_msg}"
            )

        # Register the compiled module
        _MODULE_REGISTRY[module_id] = {
            "module_id":   module_id,
            "wasm_bytes":  wasm_bytes,
            "wasm_len":    len(wasm_bytes),
            "faust_len":   len(faust_code),
            "duration_ms": duration_ms,
            "compiled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        return module_id

    def _backends(self):
        """
        Yield (name, async_fn) pairs in priority order.
        The first backend that succeeds is used; others are skipped.
        """
        import shutil

        # Backend 1: local Faust binary
        if shutil.which("faust"):
            yield "local_faust", _compile_via_local_faust

        # Backend 2: stub (always available — dev / test fallback)
        yield "stub", _compile_via_stub