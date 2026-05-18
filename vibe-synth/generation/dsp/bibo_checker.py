# =============================================================================
# File: generation/dsp/bibo_checker.py
# Purpose: Heuristic BIBO (Bounded-Input Bounded-Output) stability checker for
#          LLM-generated Faust DSP code. Scans the source for known instability
#          patterns — unbounded feedback gain, missing damping in recursive
#          structures, and unstable filter configurations — before the code
#          reaches the WASM compiler. Complements the Validator's banned-pattern
#          checks with DSP-domain-specific analysis.
# =============================================================================

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BIBOCheckResult:
    """
    Outcome of a BIBO stability analysis run.

    Attributes:
        stable:   True if no instability patterns were detected.
        warnings: Non-blocking observations (possible instability, not certain).
        errors:   Blocking issues — code is very likely unstable.
        details:  Dict of check_name → list of findings for debugging.
    """
    stable:   bool
    warnings: list[str]          = field(default_factory=list)
    errors:   list[str]          = field(default_factory=list)
    details:  dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def _check_feedback_gain(code: str) -> tuple[list[str], list[str]]:
    """
    Detect feedback loops whose gain coefficient is >= 1.0.

    In Faust, recursive signal flow is written as:
        signal ~ (delay_line * gain)
    A gain of 1.0 or greater causes exponential amplitude growth (instability).

    Heuristic: find all numeric literals that appear after a '*' operator
    inside a '~' expression context. Flag values >= 1.0 as errors,
    values in [0.9, 1.0) as warnings (borderline).
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # Locate ~ expressions (recursive feedback operator in Faust)
    feedback_blocks = re.findall(r"~\s*\(([^)]+)\)", code)

    for block in feedback_blocks:
        # Find all float/int literals multiplied inside the feedback block
        literals = re.findall(r"\*\s*([\d.]+)", block)
        for lit in literals:
            try:
                val = float(lit)
            except ValueError:
                continue
            if val >= 1.0:
                errors.append(
                    f"Feedback gain literal {val} >= 1.0 detected inside '~' expression "
                    f"'{block.strip()[:60]}'. This will cause exponential amplitude growth."
                )
            elif val >= 0.9:
                warnings.append(
                    f"Feedback gain literal {val} is close to 1.0 in '~' expression "
                    f"'{block.strip()[:60]}'. Borderline instability risk at high settings."
                )

    return errors, warnings


def _check_recursive_process(code: str) -> tuple[list[str], list[str]]:
    """
    Detect self-referential 'process' definitions without a damping path.

    A Faust program that defines 'process' in terms of itself without any
    explicit attenuation factor is almost certainly unstable. This is distinct
    from valid recursive structures using the '~' operator.

    Heuristic: flag if 'process' appears on both the left and right sides of
    an assignment without a visible multiplication by a value < 1.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # Match: process = ... process ...  (self-reference without ~)
    if re.search(r"process\s*=.*\bprocess\b", code, re.DOTALL):
        # Check if the self-reference is mediated by ~ (valid recursion)
        if "~" not in code:
            errors.append(
                "Direct self-referential 'process' definition detected without "
                "the '~' feedback operator. This is not valid Faust recursion syntax."
            )
        else:
            warnings.append(
                "'process' appears to reference itself — verify that the "
                "feedback path includes explicit damping."
            )

    return errors, warnings


def _check_missing_damping_in_reverb(code: str) -> tuple[list[str], list[str]]:
    """
    Check that reverb-like feedback comb filter structures include a
    low-pass or attenuation stage in the feedback path.

    A comb filter without damping (pure feedback delay) at gain close to 1
    will ring indefinitely. The presence of fi.lowpass, fi.resonlp, or a
    multiplication by a value < 1 inside the feedback loop is a good sign.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # Only applies if there is a feedback operator
    if "~" not in code:
        return errors, warnings

    feedback_blocks = re.findall(r"~\s*\(([^)]+)\)", code)
    for block in feedback_blocks:
        has_filter    = bool(re.search(r"\bfi\.\w+|lowpass|highpass|resonlp|resonhp\b", block))
        has_damping   = bool(re.search(r"\*\s*0\.\d+", block))  # literal < 1

        if not has_filter and not has_damping:
            warnings.append(
                f"Feedback block '{block.strip()[:60]}' contains neither a filter "
                "nor an explicit damping coefficient. Consider adding fi.lowpass() "
                "or a gain multiplier < 1.0 to ensure finite decay."
            )

    return errors, warnings


def _check_filter_q_stability(code: str) -> tuple[list[str], list[str]]:
    """
    Detect resonant filter Q values that may cause ringing or instability.

    Very high Q values (> 50) in resonant filters make the filter behave
    like an oscillator at the cutoff frequency — practically unstable for
    most audio DSP purposes.

    Heuristic: find hslider / vslider definitions whose label contains
    'resonance' or 'q' and inspect their max value.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # Match slider definitions that look like: hslider("resonance", default, min, max, step)
    slider_pattern = re.compile(
        r'[hv]slider\s*\(\s*"(?:resonance|q|q_factor|bandwidth)"'
        r'\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*([\d.]+)',
        re.IGNORECASE,
    )
    for match in slider_pattern.finditer(code):
        try:
            max_q = float(match.group(1))
        except ValueError:
            continue
        if max_q > 50.0:
            errors.append(
                f"Resonant filter maximum Q value {max_q} exceeds 50. "
                "This will cause extreme resonance and near-oscillation at the cutoff frequency."
            )
        elif max_q > 20.0:
            warnings.append(
                f"Resonant filter maximum Q value {max_q} is high (> 20). "
                "This may cause audible ringing artefacts at extreme settings."
            )

    return errors, warnings


def _check_delay_length_bounds(code: str) -> tuple[list[str], list[str]]:
    """
    Verify that delay line allocations use reasonable, bounded sizes.

    Faust delay lines are declared with de.delay(max_size, delay_samples).
    A max_size that is extremely large (> 8 192 000 samples ≈ 3 min at 44.1 kHz)
    indicates a likely typo and risks excessive memory allocation.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    delay_calls = re.findall(r"de\.delay\s*\(\s*(\d+)", code)
    for size_str in delay_calls:
        size = int(size_str)
        if size > 8_192_000:
            errors.append(
                f"Delay line max_size {size} samples is extremely large "
                f"(> 8 192 000). This is likely a typo and will cause "
                "excessive memory allocation."
            )
        elif size > 2_048_000:
            warnings.append(
                f"Delay line max_size {size} samples is unusually large. "
                "Verify this is intentional."
            )

    return errors, warnings


def _check_output_amplitude(code: str) -> tuple[list[str], list[str]]:
    """
    Look for obvious fixed gain values > 1.0 at the output stage.

    Output amplitudes > 1.0 will clip and may indicate a missing normalisation
    step after a mixing or accumulation operation.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # Match: :> * <literal>  or  *(literal)  near the end of the process definition
    # This is a broad heuristic — only flag large constants like * 10 or * 100
    large_gains = re.findall(r"\*\s*([1-9]\d+\.?\d*)\b", code)
    for gain_str in large_gains:
        try:
            gain = float(gain_str)
        except ValueError:
            continue
        if gain > 100.0:
            errors.append(
                f"Fixed gain multiplier {gain} detected. Values > 100 will "
                "produce extreme amplitude and almost certainly cause clipping or overflow."
            )
        elif gain > 10.0:
            warnings.append(
                f"Fixed gain multiplier {gain} is large (> 10). "
                "Ensure this is intentional and that output is normalised."
            )

    return errors, warnings


# ---------------------------------------------------------------------------
# Public checker
# ---------------------------------------------------------------------------

class BIBOChecker:
    """
    Runs all heuristic BIBO stability checks against a Faust code string.

    Usage:
        checker = BIBOChecker()
        result  = checker.check(faust_code)
        if not result.stable:
            # surface result.errors to the self-correction loop
    """

    _CHECKS = [
        ("feedback_gain",        _check_feedback_gain),
        ("recursive_process",    _check_recursive_process),
        ("reverb_damping",       _check_missing_damping_in_reverb),
        ("filter_q_stability",   _check_filter_q_stability),
        ("delay_length_bounds",  _check_delay_length_bounds),
        ("output_amplitude",     _check_output_amplitude),
    ]

    def check(self, faust_code: str) -> BIBOCheckResult:
        """
        Run all stability checks and return a consolidated BIBOCheckResult.

        Args:
            faust_code: Clean Faust DSP source code (no json-params block).

        Returns:
            BIBOCheckResult with stable=True if no blocking errors were found.
        """
        all_errors:   list[str] = []
        all_warnings: list[str] = []
        details:      dict[str, list[str]] = {}

        for check_name, check_fn in self._CHECKS:
            try:
                errors, warnings = check_fn(faust_code)
            except Exception as exc:
                logger.warning(
                    "BIBOChecker: check '%s' raised an exception: %s", check_name, exc
                )
                errors, warnings = [], []

            if errors or warnings:
                details[check_name] = errors + warnings

            all_errors.extend(errors)
            all_warnings.extend(warnings)

        stable = len(all_errors) == 0

        if stable and not all_warnings:
            logger.debug("BIBOChecker: all checks passed — code appears BIBO stable.")
        elif stable:
            logger.info(
                "BIBOChecker: %d warning(s) — code is borderline stable. Warnings: %s",
                len(all_warnings), all_warnings,
            )
        else:
            logger.warning(
                "BIBOChecker: %d error(s) detected — code is likely unstable. Errors: %s",
                len(all_errors), all_errors,
            )

        return BIBOCheckResult(
            stable=stable,
            warnings=all_warnings,
            errors=all_errors,
            details=details,
        )