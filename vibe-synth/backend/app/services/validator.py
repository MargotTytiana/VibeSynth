# =============================================================================
# File: backend/app/services/validator.py
# Purpose: Static analysis layer that inspects LLM-generated code before it
#          reaches the compilation sandbox. Catches common safety violations
#          (infinite loops, unbounded recursion, banned constructs) and
#          extracts the parameter manifest JSON block from the raw LLM output.
#          A failed validation triggers the self-correction loop in LLMClient
#          rather than a hard error to the caller.
# =============================================================================

import re
import json
import logging
from dataclasses import dataclass, field

from app.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Outcome of a single validation run.

    Attributes:
        passed:      True if the code is safe to send to the compiler.
        errors:      List of blocking issues that must be fixed.
        warnings:    List of non-blocking observations (logged, not returned to caller).
        clean_code:  The code with the ```json-params``` block stripped out,
                     ready for compilation.
        parameters:  Parsed parameter manifest extracted from the LLM output.
    """
    passed:     bool
    errors:     list[str] = field(default_factory=list)
    warnings:   list[str] = field(default_factory=list)
    clean_code: str        = ""
    parameters: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Banned pattern definitions
# ---------------------------------------------------------------------------

# Faust DSP — patterns that indicate an unsafe or unstable algorithm
_FAUST_BANNED: list[tuple[str, str]] = [
    (r"\bwhile\s*\(",          "Unbounded 'while' loop detected in Faust code"),
    (r"\bfor\s*\([^;]*;[^;]*;[^)]*\b[a-zA-Z_]\w*\s*(?!\d)",
                               "Dynamic loop bound detected — all loops must be constant"),
    (r"letrec\b.*letrec\b",    "Nested 'letrec' may cause unbounded recursion"),
]

# WGSL shader — WebGPU shading language unsafe patterns
_WGSL_BANNED: list[tuple[str, str]] = [
    (r"\bloop\s*\{",           "Unbounded 'loop {}' construct — use 'for' with constant bound"),
    (r"\bcontinuing\s*\{",     "'continuing' block without constant-bound loop"),
    (r"\bwhile\s*\(",          "WGSL does not support 'while' — use 'for' with constant bound"),
]

# GLSL shader — OpenGL ES 3.0 unsafe patterns
_GLSL_BANNED: list[tuple[str, str]] = [
    (r"\bwhile\s*\(",          "Unbounded 'while' loop in GLSL — use 'for' with constant bound"),
    (r"\bdo\s*\{",             "'do-while' loop detected — use 'for' with constant bound"),
    (r"#extension\s+GL_",     "GL extension usage requires explicit allowlisting"),
]

# Shared across all languages — security-sensitive patterns
_UNIVERSAL_BANNED: list[tuple[str, str]] = [
    (r"__asm__",               "Inline assembly is not permitted"),
    (r"system\s*\(",           "System calls are not permitted"),
    (r"exec\s*\(",             "exec() calls are not permitted"),
    (r"import\s+os\b",         "OS module import is not permitted in generated code"),
    (r"import\s+subprocess\b", "subprocess import is not permitted in generated code"),
]

# Regex to locate the ```json-params``` block the LLM appends after the code
_PARAMS_BLOCK_RE = re.compile(
    r"```json-params\s*([\s\S]*?)```",
    re.IGNORECASE,
)

# Regex to locate the primary code block (Faust / WGSL / GLSL)
_CODE_BLOCK_RE = re.compile(
    r"```(?:faust|wgsl|glsl|hlsl)?\s*([\s\S]*?)```",
    re.IGNORECASE,
)


class Validator:
    """
    Static-analysis validator for LLM-generated DSP and shader code.

    Usage:
        result = validator.validate_dsp(raw_llm_output)
        if not result.passed:
            # trigger self-correction with result.errors
        else:
            compile(result.clean_code, result.parameters)
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def validate_dsp(self, raw_output: str) -> ValidationResult:
        """
        Validate raw LLM output for a Faust DSP generation request.

        Steps:
          1. Extract the Faust code block and the json-params block.
          2. Run universal + Faust-specific banned-pattern checks.
          3. Validate and parse the parameter manifest JSON.
          4. Return a ValidationResult.
        """
        return self._run(raw_output, language="faust")

    def validate_shader(self, raw_output: str, target_platform: str = "webgpu") -> ValidationResult:
        """
        Validate raw LLM output for a shader generation request.

        Args:
            raw_output:      The full text returned by the LLM.
            target_platform: 'webgpu' (WGSL) or 'webgl' (GLSL).
        """
        language = "wgsl" if target_platform == "webgpu" else "glsl"
        return self._run(raw_output, language=language)

    # ------------------------------------------------------------------
    # Internal validation pipeline
    # ------------------------------------------------------------------

    def _run(self, raw_output: str, language: str) -> ValidationResult:
        errors:   list[str] = []
        warnings: list[str] = []

        # Step 1 — extract code block
        clean_code, code_warnings = self._extract_code(raw_output, language)
        warnings.extend(code_warnings)
        if not clean_code.strip():
            errors.append(
                f"No {language.upper()} code block found in LLM output. "
                "Expected a fenced code block tagged with the language name."
            )
            return ValidationResult(passed=False, errors=errors, warnings=warnings)

        # Step 2 — run banned-pattern checks
        pattern_errors = self._check_banned_patterns(clean_code, language)
        errors.extend(pattern_errors)

        # Step 3 — extract and validate parameter manifest
        parameters, param_errors, param_warnings = self._extract_parameters(raw_output)
        errors.extend(param_errors)
        warnings.extend(param_warnings)

        # Step 4 — basic structural checks
        struct_errors = self._check_structure(clean_code, language)
        errors.extend(struct_errors)

        passed = len(errors) == 0

        if passed:
            logger.info("Validation passed | language=%s params=%d", language, len(parameters))
        else:
            logger.warning(
                "Validation failed | language=%s errors=%d: %s",
                language, len(errors), "; ".join(errors),
            )

        return ValidationResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
            clean_code=clean_code,
            parameters=parameters,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_code(self, raw: str, language: str) -> tuple[str, list[str]]:
        """
        Pull the primary code block out of the raw LLM output.
        Strips the json-params block first so it does not end up in the
        code sent to the compiler.
        """
        warnings: list[str] = []

        # Remove the json-params block before looking for the code block
        without_params = _PARAMS_BLOCK_RE.sub("", raw)

        match = _CODE_BLOCK_RE.search(without_params)
        if match:
            return match.group(1).strip(), warnings

        # Fallback: if the LLM forgot the fenced block, treat the entire
        # output (minus the params block) as code and warn.
        warnings.append(
            "No fenced code block found — using entire LLM output as code. "
            "Ask the model to wrap the code in a fenced block."
        )
        return without_params.strip(), warnings

    def _check_banned_patterns(self, code: str, language: str) -> list[str]:
        """Run all applicable banned-pattern regexes against the code."""
        errors: list[str] = []
        pattern_sets = [_UNIVERSAL_BANNED]

        if language == "faust":
            pattern_sets.append(_FAUST_BANNED)
        elif language == "wgsl":
            pattern_sets.append(_WGSL_BANNED)
        elif language == "glsl":
            pattern_sets.append(_GLSL_BANNED)

        for pattern_set in pattern_sets:
            for pattern, message in pattern_set:
                if re.search(pattern, code, re.IGNORECASE):
                    errors.append(message)

        return errors

    def _extract_parameters(
        self, raw: str
    ) -> tuple[list[dict], list[str], list[str]]:
        """
        Locate the ```json-params``` block and parse its contents.

        Returns:
            (parameters, errors, warnings)
            - parameters: list of parameter dicts (may be empty on failure)
            - errors:     blocking issues (malformed JSON, missing required keys)
            - warnings:   non-blocking observations
        """
        errors:     list[str] = []
        warnings:   list[str] = []
        parameters: list[dict] = []

        match = _PARAMS_BLOCK_RE.search(raw)
        if not match:
            warnings.append(
                "No ```json-params``` block found in LLM output — "
                "UI will have no auto-generated controls."
            )
            return parameters, errors, warnings

        try:
            data = json.loads(match.group(1).strip())
        except json.JSONDecodeError as exc:
            errors.append(f"Parameter manifest JSON is malformed: {exc}")
            return parameters, errors, warnings

        if not isinstance(data, list):
            errors.append(
                "Parameter manifest must be a JSON array of parameter objects."
            )
            return parameters, errors, warnings

        required_keys = {"name", "type", "range", "default"}
        for i, param in enumerate(data):
            missing = required_keys - set(param.keys())
            if missing:
                errors.append(
                    f"Parameter [{i}] is missing required keys: {missing}"
                )
            if "range" in param and (
                not isinstance(param["range"], list) or len(param["range"]) != 2
            ):
                errors.append(
                    f"Parameter [{i}] 'range' must be a two-element array [min, max]."
                )

        if not errors:
            parameters = data

        return parameters, errors, warnings

    def _check_structure(self, code: str, language: str) -> list[str]:
        """
        Light structural sanity checks specific to each language.
        These complement the banned-pattern checks with positive assertions
        (i.e. things that MUST be present).
        """
        errors: list[str] = []

        if language == "faust":
            if "process" not in code:
                errors.append(
                    "Faust code must define a 'process' expression as its entry point."
                )

        elif language == "wgsl":
            if "@fragment" not in code:
                errors.append(
                    "WGSL shader must contain a @fragment entry point function."
                )

        elif language == "glsl":
            if "void main(" not in code and "void main (" not in code:
                errors.append(
                    "GLSL shader must define a 'void main()' entry point."
                )

        return errors