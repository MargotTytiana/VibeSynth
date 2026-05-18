# =============================================================================
# File: compiler/error_handler.py
# Purpose: Parses and classifies raw compiler and sandbox error messages into
#          structured, actionable error objects. Translates cryptic Faust
#          compiler stderr, wasmtime trap messages, and WASM validation errors
#          into human-readable descriptions and LLM-ready correction hints.
#          Used by both wasm_compiler.py and sandbox_runner.py to produce
#          consistent error payloads for the self-correction loop.
# =============================================================================

import re
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class ErrorCategory(str, Enum):
    """
    Top-level category of a compiler or runtime error.
    Used to select the right correction strategy in the self-correction loop.
    """
    SYNTAX          = "syntax"           # Faust / WGSL / GLSL parse error
    STABILITY       = "stability"        # BIBO violation, unbounded feedback
    UNDEFINED_REF   = "undefined_ref"    # Unknown symbol or missing import
    TYPE_MISMATCH   = "type_mismatch"    # Signal arity or type error
    LOOP_VIOLATION  = "loop_violation"   # Dynamic loop bound in shader
    RUNTIME_TRAP    = "runtime_trap"     # WASM trap at instantiation
    TIMEOUT         = "timeout"          # Compilation exceeded time limit
    MEMORY          = "memory"           # OOM or excessive allocation
    UNKNOWN         = "unknown"          # Catch-all


@dataclass
class CompilerError:
    """
    Structured representation of a single compiler or runtime error.

    Attributes:
        category:     ErrorCategory enum value.
        raw_message:  Original error string from the compiler / runtime.
        summary:      Short human-readable description (1 sentence).
        correction_hint: Prompt fragment telling the LLM how to fix this class
                         of error. Injected into the self-correction user message.
        line_number:  Source line number if extractable, else None.
        severity:     'error' | 'warning'
    """
    category:         ErrorCategory
    raw_message:      str
    summary:          str
    correction_hint:  str
    line_number:      int | None  = None
    severity:         str         = "error"


@dataclass
class ErrorReport:
    """
    Aggregated result of parsing one or more raw error strings.

    Attributes:
        errors:          List of structured CompilerError objects.
        dominant_category: Most frequent ErrorCategory across all errors.
        llm_hint_block:  Ready-to-inject string for the self-correction prompt.
        is_recoverable:  True if the errors are the kind the LLM can fix.
    """
    errors:             list[CompilerError] = field(default_factory=list)
    dominant_category:  ErrorCategory       = ErrorCategory.UNKNOWN
    llm_hint_block:     str                 = ""
    is_recoverable:     bool                = True


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

# Each entry: (regex pattern, ErrorCategory, summary_template, correction_hint)
_FAUST_PATTERNS: list[tuple[re.Pattern, ErrorCategory, str, str]] = [
    (
        re.compile(r"syntax error", re.I),
        ErrorCategory.SYNTAX,
        "Faust syntax error — the generated code is not valid Faust 2 syntax.",
        "Fix all syntax errors. Ensure the code uses valid Faust 2 syntax: "
        "correct operator precedence, matching parentheses, and a top-level 'process' definition.",
    ),
    (
        re.compile(r"undefined symbol|undefined variable|not defined", re.I),
        ErrorCategory.UNDEFINED_REF,
        "Undefined symbol — a function or variable was used before being declared.",
        "Check that all functions are defined or imported before use. "
        "Add 'import(\"stdfaust.lib\");' at the top if using standard library functions "
        "such as fi, ef, os, ba, ma, de.",
    ),
    (
        re.compile(r"process\s+is\s+not\s+defined|missing\s+process", re.I),
        ErrorCategory.UNDEFINED_REF,
        "Missing 'process' definition — Faust requires a top-level 'process' expression.",
        "Add a top-level 'process = ...' definition. Every Faust program must define "
        "'process' as its entry point.",
    ),
    (
        re.compile(r"wrong number of inputs|wrong number of outputs|arity", re.I),
        ErrorCategory.TYPE_MISMATCH,
        "Signal arity mismatch — a processor received the wrong number of input or output signals.",
        "Check the number of input and output signals at each composition point. "
        "Use ':>' to merge multiple signals and '<:' to split one signal into multiple. "
        "Verify that all operators have matching signal counts.",
    ),
    (
        re.compile(r"type error|cannot apply|incompatible type", re.I),
        ErrorCategory.TYPE_MISMATCH,
        "Type error — a value of the wrong type was passed to a Faust operator.",
        "Check that numeric literals match the expected type (float vs int) and "
        "that signal-processing operators are not applied to non-signal values.",
    ),
    (
        re.compile(r"infinite loop|recursion.*not.*productive|non-productive", re.I),
        ErrorCategory.STABILITY,
        "Non-productive recursion — the Faust compiler detected an infinite feedback loop.",
        "Ensure all recursive signal paths include at least one unit delay (mem or @(1)). "
        "In Faust, the '~' operator requires the feedback path to be delayed by at least "
        "one sample to be productive.",
    ),
    (
        re.compile(r"timed out|timeout", re.I),
        ErrorCategory.TIMEOUT,
        "Compilation timed out — the Faust compiler exceeded the allowed time limit.",
        "Simplify the algorithm. Avoid deeply nested expressions, very large delay lines, "
        "or polynomial combinatorial expansions. Break complex signal chains into named "
        "intermediate variables.",
    ),
]

_WASM_PATTERNS: list[tuple[re.Pattern, ErrorCategory, str, str]] = [
    (
        re.compile(r"trap|unreachable|integer overflow|out of bounds", re.I),
        ErrorCategory.RUNTIME_TRAP,
        "WASM runtime trap — the module triggered an unreachable instruction or memory violation.",
        "Check for division by zero, out-of-bounds array access, or integer overflow in "
        "the generated code. Ensure all delay line indices are within declared bounds.",
    ),
    (
        re.compile(r"memory.*limit|allocation.*failed|OOM|out of memory", re.I),
        ErrorCategory.MEMORY,
        "Memory allocation failure — the WASM module requested more memory than allowed.",
        "Reduce delay line sizes. The sum of all de.delay(max_size, ...) values should not "
        "exceed approximately 4 million samples total.",
    ),
    (
        re.compile(r"magic.*number|invalid.*wasm|malformed", re.I),
        ErrorCategory.SYNTAX,
        "Malformed WASM binary — the output is not a valid WebAssembly module.",
        "The Faust compiler produced invalid output. Simplify the Faust code and check "
        "for any non-standard syntax that may have confused the backend.",
    ),
]

_SHADER_PATTERNS: list[tuple[re.Pattern, ErrorCategory, str, str]] = [
    (
        re.compile(r"dynamic.*loop|non-constant.*bound|loop.*variable.*bound", re.I),
        ErrorCategory.LOOP_VIOLATION,
        "Dynamic loop bound — all for-loop iteration counts must be compile-time constants.",
        "Replace all dynamic loop bounds with compile-time integer literals. "
        "For example: 'for (int i = 0; i < 16; i++)' not 'for (int i = 0; i < u_count; i++)'.",
    ),
    (
        re.compile(r"undeclared identifier|undefined.*variable|not declared", re.I),
        ErrorCategory.UNDEFINED_REF,
        "Undeclared identifier — a variable or function was used without being declared.",
        "Declare all variables before use. In WGSL, use 'let' or 'var'. "
        "In GLSL ES 3.0, declare variables at the correct scope.",
    ),
    (
        re.compile(r"type mismatch|cannot.*convert|implicit.*conversion", re.I),
        ErrorCategory.TYPE_MISMATCH,
        "Type mismatch — scalar/vector type conversion error in the shader.",
        "Use explicit constructors for type conversions: vec2f(x, y), vec4f(rgb, 1.0). "
        "In WGSL, f32 and i32 are not implicitly convertible.",
    ),
    (
        re.compile(r"missing.*@fragment|no.*fragment.*entry|entry.*point", re.I),
        ErrorCategory.SYNTAX,
        "Missing @fragment entry point in WGSL shader.",
        "Add a function decorated with @fragment that returns @location(0) vec4f. "
        "Example: '@fragment fn fs_main(in: FragInput) -> @location(0) vec4f { ... }'",
    ),
    (
        re.compile(r"missing.*void main|no.*main.*function", re.I),
        ErrorCategory.SYNTAX,
        "Missing void main() in GLSL shader.",
        "Add 'void main() { ... }' as the entry point. "
        "Assign the output colour to 'frag_color' (declared as 'out vec4 frag_color;').",
    ),
]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ErrorHandler:
    """
    Parses raw error strings from the Faust compiler, WASM runtime, or shader
    validator into structured ErrorReport objects ready for the self-correction
    loop.

    Usage:
        handler = ErrorHandler()
        report  = handler.parse_faust_errors(stderr_text)
        if not report.is_recoverable:
            raise RuntimeError("Unrecoverable compilation failure")
        corrected = await llm_client.self_correct(..., error_message=report.llm_hint_block)
    """

    # Error categories that the LLM can realistically fix
    _RECOVERABLE = {
        ErrorCategory.SYNTAX,
        ErrorCategory.UNDEFINED_REF,
        ErrorCategory.TYPE_MISMATCH,
        ErrorCategory.LOOP_VIOLATION,
        ErrorCategory.STABILITY,
    }

    # Error categories where retrying is unlikely to help
    _UNRECOVERABLE = {
        ErrorCategory.TIMEOUT,
        ErrorCategory.MEMORY,
        ErrorCategory.RUNTIME_TRAP,
    }

    def parse_faust_errors(self, raw: str) -> ErrorReport:
        """Parse Faust compiler stderr into a structured ErrorReport."""
        return self._parse(raw, _FAUST_PATTERNS)

    def parse_wasm_errors(self, raw: str) -> ErrorReport:
        """Parse WASM runtime / validator errors into a structured ErrorReport."""
        return self._parse(raw, _WASM_PATTERNS)

    def parse_shader_errors(self, raw: str) -> ErrorReport:
        """Parse shader validation errors into a structured ErrorReport."""
        return self._parse(raw, _SHADER_PATTERNS)

    def parse_auto(self, raw: str) -> ErrorReport:
        """
        Auto-detect the error source and parse accordingly.
        Tries Faust, WASM, and shader patterns in order and returns
        the report with the most matches.
        """
        candidates = [
            self._parse(raw, _FAUST_PATTERNS),
            self._parse(raw, _WASM_PATTERNS),
            self._parse(raw, _SHADER_PATTERNS),
        ]
        # Pick the report with the most matched errors
        best = max(candidates, key=lambda r: len(r.errors))
        if not best.errors:
            # Nothing matched — return a generic unknown error
            return self._make_unknown_report(raw)
        return best

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse(
        self,
        raw: str,
        patterns: list[tuple[re.Pattern, ErrorCategory, str, str]],
    ) -> ErrorReport:
        """Apply a pattern table to a raw error string."""
        errors: list[CompilerError] = []

        for pattern, category, summary, hint in patterns:
            for match in pattern.finditer(raw):
                line_num = self._extract_line_number(raw, match.start())
                errors.append(CompilerError(
                    category=category,
                    raw_message=match.group(0),
                    summary=summary,
                    correction_hint=hint,
                    line_number=line_num,
                ))

        if not errors:
            return self._make_unknown_report(raw)

        dominant = self._dominant_category(errors)
        recoverable = dominant not in self._UNRECOVERABLE
        hint_block = self._build_hint_block(errors)

        return ErrorReport(
            errors=errors,
            dominant_category=dominant,
            llm_hint_block=hint_block,
            is_recoverable=recoverable,
        )

    @staticmethod
    def _extract_line_number(text: str, match_pos: int) -> int | None:
        """
        Try to extract a line number from the text near the match position.
        Looks for patterns like 'line 42', ':42:', 'error at 42'.
        """
        window = text[max(0, match_pos - 100): match_pos + 100]
        m = re.search(r"(?:line|at|:)\s*(\d+)", window, re.I)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return None

    @staticmethod
    def _dominant_category(errors: list[CompilerError]) -> ErrorCategory:
        """Return the most frequently occurring ErrorCategory."""
        from collections import Counter
        counts = Counter(e.category for e in errors)
        return counts.most_common(1)[0][0]

    @staticmethod
    def _build_hint_block(errors: list[CompilerError]) -> str:
        """
        Assemble a compact correction-hint block suitable for injection
        into an LLM self-correction user message.
        """
        seen_hints: set[str] = set()
        lines = ["The following errors must be fixed:"]
        for i, err in enumerate(errors, start=1):
            lines.append(f"\n[Error {i}] {err.summary}")
            if err.line_number:
                lines.append(f"  Location: line {err.line_number}")
            if err.correction_hint not in seen_hints:
                lines.append(f"  Fix: {err.correction_hint}")
                seen_hints.add(err.correction_hint)
        return "\n".join(lines)

    @staticmethod
    def _make_unknown_report(raw: str) -> ErrorReport:
        """Fallback report for unrecognised error messages."""
        summary = raw.strip()[:200] or "Unknown compiler error."
        error = CompilerError(
            category=ErrorCategory.UNKNOWN,
            raw_message=raw,
            summary=summary,
            correction_hint=(
                "Review the full error message carefully and correct any "
                "syntax, type, or structural issues in the generated code."
            ),
        )
        return ErrorReport(
            errors=[error],
            dominant_category=ErrorCategory.UNKNOWN,
            llm_hint_block=f"The following error must be fixed:\n\n{summary}",
            is_recoverable=True,
        )