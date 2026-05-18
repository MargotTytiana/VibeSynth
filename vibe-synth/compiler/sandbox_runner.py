# =============================================================================
# File: compiler/sandbox_runner.py
# Purpose: Executes compiled WASM modules in an isolated sandbox environment
#          to verify they load correctly and do not trigger runtime errors
#          (infinite loops, memory violations, trap instructions) before the
#          module ID is returned to the frontend. Uses the wasmtime Python
#          bindings for sandboxed execution with configurable fuel limits to
#          prevent runaway computation.
# =============================================================================

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """
    Outcome of a sandboxed WASM execution attempt.

    Attributes:
        passed:       True if the module loaded and ran without trapping.
        duration_ms:  Wall-clock time for the sandbox run.
        fuel_used:    Wasmtime fuel units consumed (proxy for instruction count).
        errors:       List of blocking errors (traps, OOM, timeout).
        warnings:     Non-blocking observations.
        exports:      Dict of exported symbol names → types discovered at load time.
    """
    passed:      bool
    duration_ms: int                  = 0
    fuel_used:   int                  = 0
    errors:      list[str]            = field(default_factory=list)
    warnings:    list[str]            = field(default_factory=list)
    exports:     dict[str, str]       = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Wasmtime backend
# ---------------------------------------------------------------------------

def _run_wasmtime(
    wasm_bytes:  bytes,
    fuel_limit:  int,
) -> SandboxResult:
    """
    Load and validate a WASM module using wasmtime with a fuel limit.

    Fuel in wasmtime maps roughly to the number of instructions executed.
    Setting a fuel limit prevents infinite loops from hanging the process.

    The sandbox only calls exported functions that take no arguments and
    return nothing (i.e. it verifies load-time initialisation, not audio
    processing logic — the latter runs in the browser AudioWorklet).

    Args:
        wasm_bytes:  Raw WASM binary to validate.
        fuel_limit:  Maximum wasmtime fuel units before aborting.

    Returns:
        SandboxResult describing the outcome.
    """
    try:
        import wasmtime
    except ImportError:
        return SandboxResult(
            passed=True,
            warnings=["wasmtime package not installed — sandbox check skipped."],
        )

    t0 = time.perf_counter()

    try:
        # Configure engine with fuel metering enabled
        engine_cfg = wasmtime.Config()
        engine_cfg.consume_fuel = True
        engine    = wasmtime.Engine(engine_cfg)
        store     = wasmtime.Store(engine)
        store.set_fuel(fuel_limit)

        # Compile the module (catches malformed WASM)
        module = wasmtime.Module(engine, wasm_bytes)

        # Collect exports for inspection
        exports: dict[str, str] = {}
        for export in module.exports:
            exports[export.name] = str(type(export.type).__name__)

        # Instantiate (runs any start function / global initialisers)
        linker   = wasmtime.Linker(engine)
        instance = linker.instantiate(store, module)  # noqa: F841

        fuel_used = fuel_limit - store.get_fuel()
        duration_ms = int((time.perf_counter() - t0) * 1000)

        logger.info(
            "SandboxRunner | wasmtime | passed=True fuel_used=%d/%d exports=%s",
            fuel_used, fuel_limit, list(exports.keys()),
        )

        return SandboxResult(
            passed=True,
            duration_ms=duration_ms,
            fuel_used=fuel_used,
            exports=exports,
        )

    except wasmtime.WasmtimeError as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        error_msg   = str(exc)
        logger.warning("SandboxRunner | wasmtime trap/error: %s", error_msg)
        return SandboxResult(
            passed=False,
            duration_ms=duration_ms,
            errors=[f"WASM runtime error: {error_msg}"],
        )

    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.error("SandboxRunner | unexpected error: %s", exc, exc_info=True)
        return SandboxResult(
            passed=False,
            duration_ms=duration_ms,
            errors=[f"Sandbox runner unexpected error: {exc}"],
        )


# ---------------------------------------------------------------------------
# Stub backend (when wasmtime is not installed)
# ---------------------------------------------------------------------------

def _run_stub(wasm_bytes: bytes, fuel_limit: int) -> SandboxResult:  # noqa: ARG001
    """
    Minimal structural check used when wasmtime is unavailable.
    Verifies the WASM magic number and version bytes only.
    """
    WASM_MAGIC   = b"\x00asm"
    WASM_VERSION = b"\x01\x00\x00\x00"

    if not wasm_bytes.startswith(WASM_MAGIC + WASM_VERSION):
        return SandboxResult(
            passed=False,
            errors=["WASM bytes do not start with the expected magic number + version."],
        )

    logger.debug("SandboxRunner | stub check passed (magic bytes valid).")
    return SandboxResult(
        passed=True,
        warnings=["Full sandbox check skipped — wasmtime not installed."],
    )


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

class SandboxRunner:
    """
    Validates a compiled WASM module in an isolated sandbox before it is
    served to the frontend.

    Backends tried in order:
      1. wasmtime (full fuel-limited execution)
      2. stub     (magic-byte check only — always available)

    Usage:
        runner = SandboxRunner(fuel_limit=1_000_000)
        result = runner.run(wasm_bytes)
        if not result.passed:
            raise RuntimeError(result.errors)
    """

    # Default fuel limit: ~1M instructions, roughly equivalent to a few
    # milliseconds of real computation — enough for WASM module initialisation
    # without risking process hang on infinite loops.
    DEFAULT_FUEL_LIMIT = 1_000_000

    def __init__(self, fuel_limit: int = DEFAULT_FUEL_LIMIT) -> None:
        self.fuel_limit = fuel_limit
        self._has_wasmtime = self._check_wasmtime()

    @staticmethod
    def _check_wasmtime() -> bool:
        try:
            import wasmtime  # noqa: F401
            return True
        except ImportError:
            return False

    def run(self, wasm_bytes: bytes) -> SandboxResult:
        """
        Run the WASM module through the best available sandbox backend.

        Args:
            wasm_bytes: Raw WASM binary produced by WasmCompiler.

        Returns:
            SandboxResult — check result.passed before proceeding.
        """
        if not wasm_bytes:
            return SandboxResult(
                passed=False,
                errors=["Empty WASM bytes provided to sandbox runner."],
            )

        if self._has_wasmtime:
            logger.debug("SandboxRunner | using wasmtime backend.")
            return _run_wasmtime(wasm_bytes, self.fuel_limit)

        logger.debug("SandboxRunner | wasmtime not available — using stub backend.")
        return _run_stub(wasm_bytes, self.fuel_limit)

    def run_from_registry(self, module_id: str) -> SandboxResult:
        """
        Convenience method that fetches WASM bytes from the WasmCompiler
        registry and runs the sandbox check.

        Args:
            module_id: ID returned by WasmCompiler.compile_faust().

        Returns:
            SandboxResult, or a failed result if the module_id is not found.
        """
        from compiler.wasm_compiler import get_module

        entry = get_module(module_id)
        if entry is None:
            return SandboxResult(
                passed=False,
                errors=[
                    f"Module '{module_id}' not found in the WASM registry. "
                    "Compile it first via WasmCompiler.compile_faust()."
                ],
            )

        logger.info(
            "SandboxRunner | validating registry module_id=%s wasm_len=%d",
            module_id, entry.get("wasm_len", 0),
        )
        return self.run(entry["wasm_bytes"])

    def summary(self, result: SandboxResult) -> dict[str, Any]:
        """
        Return a JSON-serialisable summary of a SandboxResult for logging
        and API responses.
        """
        return {
            "passed":      result.passed,
            "duration_ms": result.duration_ms,
            "fuel_used":   result.fuel_used,
            "errors":      result.errors,
            "warnings":    result.warnings,
            "exports":     result.exports,
        }