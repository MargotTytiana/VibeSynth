# =============================================================================
# File: generation/dsp/ast_builder.py
# Purpose: Constructs a Signal Processing Abstract Syntax Tree (AST) from a
#          parsed Vibe intent. The AST represents the high-level processing
#          chain (e.g. filter → saturation → reverb) before any Faust code is
#          generated. It drives template selection from dsp_templates.json and
#          provides the LLM with a structured starting point, reducing
#          hallucination and improving compilation success rates.
# =============================================================================

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DSP_TEMPLATES_PATH = Path("generation/dsp/dsp_templates.json")


# ---------------------------------------------------------------------------
# AST node and tree data structures
# ---------------------------------------------------------------------------

@dataclass
class ASTNode:
    """
    A single processing stage in the signal chain.

    Attributes:
        node_type:   Canonical type string matching dsp_templates ast_node_type
                     (e.g. 'reverb', 'filter_lp', 'distortion_soft').
        template_id: ID of the matched template from dsp_templates.json.
        parameters:  Dict of parameter name → initial value for this stage.
        tags_matched: Vibe words that triggered selection of this node.
        faust_fragment: Faust code skeleton with placeholders filled in.
    """
    node_type:      str
    template_id:    str
    parameters:     dict[str, float]        = field(default_factory=dict)
    tags_matched:   list[str]               = field(default_factory=list)
    faust_fragment: str                     = ""
    metadata:       dict[str, Any]          = field(default_factory=dict)


@dataclass
class SignalAST:
    """
    Ordered list of ASTNodes representing the complete signal processing chain.
    Nodes are applied left-to-right (series composition in Faust: A : B : C).

    Attributes:
        nodes:       Ordered processing stages.
        vibe_text:   Original Vibe description that produced this AST.
        confidence:  0–1 score reflecting how well the templates matched the Vibe.
    """
    nodes:      list[ASTNode] = field(default_factory=list)
    vibe_text:  str           = ""
    confidence: float         = 0.0

    def to_dict(self) -> dict:
        return {
            "vibe_text":  self.vibe_text,
            "confidence": self.confidence,
            "chain": [
                {
                    "node_type":    n.node_type,
                    "template_id":  n.template_id,
                    "parameters":   n.parameters,
                    "tags_matched": n.tags_matched,
                }
                for n in self.nodes
            ],
        }

    def to_faust_hint(self) -> str:
        """
        Render a plain-text chain description used as additional context
        in the LLM user prompt, e.g.:
            'Suggested chain: lowpass_filter → saturation_soft → reverb_room'
        """
        if not self.nodes:
            return ""
        chain = " → ".join(n.template_id for n in self.nodes)
        return f"Suggested processing chain (refine as needed): {chain}"


# ---------------------------------------------------------------------------
# Template loader
# ---------------------------------------------------------------------------

def _load_templates() -> list[dict]:
    """Load and return the DSP template list from dsp_templates.json."""
    if not DSP_TEMPLATES_PATH.exists():
        logger.warning(
            "DSP templates file not found at %s — AST builder will return empty trees.",
            DSP_TEMPLATES_PATH,
        )
        return []
    with DSP_TEMPLATES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("templates", [])


# ---------------------------------------------------------------------------
# Vibe tokeniser
# ---------------------------------------------------------------------------

# Common English stop-words to ignore during tag matching
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "with", "like",
    "feel", "feels", "feeling", "sound", "sounds", "sounding",
    "make", "makes", "is", "it", "to", "that", "this", "be",
    "very", "quite", "really", "bit", "little", "more", "less",
}


def _tokenise(vibe: str) -> list[str]:
    """
    Split a Vibe string into lowercase alpha tokens, removing stop-words
    and punctuation. Used for tag matching against template tag lists.

    Example:
        "Make it feel like a freezing, empty cave" →
        ["freezing", "empty", "cave"]
    """
    tokens = []
    for word in vibe.lower().split():
        cleaned = "".join(c for c in word if c.isalpha() or c == "-")
        if cleaned and cleaned not in _STOP_WORDS and len(cleaned) > 2:
            tokens.append(cleaned)
    return tokens


# ---------------------------------------------------------------------------
# Tag scorer
# ---------------------------------------------------------------------------

def _score_template(template: dict, tokens: list[str]) -> tuple[float, list[str]]:
    """
    Score a template against the tokenised Vibe.

    Strategy:
      - Exact token match against template tags: +2.0 per match
      - Substring match (token is part of a tag phrase): +0.5 per match
      - Normalise by number of tokens so short Vibes are not penalised

    Returns:
        (score, matched_tags) — score in [0, ∞), matched_tags for provenance.
    """
    tags          = [t.lower() for t in template.get("tags", [])]
    matched: list[str] = []
    raw_score     = 0.0

    for token in tokens:
        for tag in tags:
            if token == tag:
                raw_score += 2.0
                if tag not in matched:
                    matched.append(tag)
                break
            elif token in tag or tag in token:
                raw_score += 0.5
                if tag not in matched:
                    matched.append(tag)

    normalised = raw_score / max(len(tokens), 1)
    return normalised, matched


# ---------------------------------------------------------------------------
# Chain composition rules
# ---------------------------------------------------------------------------

# Maps canonical node types to their recommended position in a signal chain.
# Lower index = earlier in the chain (pre-processing before post-processing).
_CHAIN_ORDER: dict[str, int] = {
    "filter_hp":            0,   # high-pass first — remove DC / rumble
    "filter_lp":            1,   # tonal shaping
    "distortion_soft":      2,   # saturation before time-based effects
    "distortion_digital":   2,
    "modulation_am":        3,   # tremolo / chorus before reverb
    "modulation_chorus":    3,
    "dynamics_gate":        4,   # gate before compressor
    "dynamics_compressor":  4,
    "delay":                5,   # delay before reverb (in standard routing)
    "reverb":               6,   # reverb last — global space
}

_DEFAULT_ORDER = 5  # fallback for unknown node types


def _sort_chain(nodes: list[ASTNode]) -> list[ASTNode]:
    """Sort nodes by their canonical chain position."""
    return sorted(nodes, key=lambda n: _CHAIN_ORDER.get(n.node_type, _DEFAULT_ORDER))


# ---------------------------------------------------------------------------
# Parameter interpolation
# ---------------------------------------------------------------------------

def _fill_skeleton(skeleton: str, parameters: dict[str, float]) -> str:
    """
    Replace {{param_name}} placeholders in a Faust skeleton string with
    the corresponding float values from the parameters dict.

    Example:
        skeleton = "hslider(..., {{room_size}}, ...)"
        parameters = {"room_size": 0.7}
        → "hslider(..., 0.7, ...)"
    """
    result = skeleton
    for name, value in parameters.items():
        placeholder = "{{" + name + "}}"
        result = result.replace(placeholder, str(round(float(value), 4)))
    return result


# ---------------------------------------------------------------------------
# Acoustic feature adjustments
# ---------------------------------------------------------------------------

# Modifier words and how they shift parameter values from their defaults.
# Format: { keyword: { template_id: { param: multiplier } } }
_MODIFIERS: dict[str, dict[str, dict[str, float]]] = {
    "large":     {"reverb_room": {"room_size": 1.6, "wet_mix": 1.3}},
    "huge":      {"reverb_room": {"room_size": 2.0, "wet_mix": 1.5}},
    "small":     {"reverb_room": {"room_size": 0.4, "wet_mix": 0.7}},
    "long":      {"delay_echo":  {"delay_ms":  1.8, "feedback": 1.2}},
    "short":     {"delay_echo":  {"delay_ms":  0.3, "feedback": 0.7}},
    "bright":    {"lowpass_filter": {"cutoff": 1.8}},
    "dark":      {"lowpass_filter": {"cutoff": 0.4}},
    "heavy":     {"saturation_soft": {"drive": 1.5}},
    "gentle":    {"saturation_soft": {"drive": 0.4}},
    "fast":      {"tremolo": {"rate": 2.5}},
    "slow":      {"tremolo": {"rate": 0.3}},
    "freezing":  {"reverb_room": {"damping": 0.3, "room_size": 1.4}},
    "warm":      {"lowpass_filter": {"cutoff": 0.6}, "saturation_soft": {"drive": 0.5}},
}


def _apply_modifiers(
    nodes: list[ASTNode], tokens: list[str]
) -> list[ASTNode]:
    """
    Adjust parameter values based on modifier keywords found in the Vibe tokens.
    Multipliers are clamped to parameter hint ranges where available.
    """
    for node in nodes:
        for token in tokens:
            if token in _MODIFIERS:
                adjustments = _MODIFIERS[token].get(node.template_id, {})
                for param, multiplier in adjustments.items():
                    if param in node.parameters:
                        node.parameters[param] = round(
                            node.parameters[param] * multiplier, 4
                        )
    return nodes


# ---------------------------------------------------------------------------
# Public AST builder
# ---------------------------------------------------------------------------

class ASTBuilder:
    """
    Builds a SignalAST from a natural language Vibe description by:
      1. Tokenising the Vibe into searchable keywords.
      2. Scoring all DSP templates against those keywords.
      3. Selecting the top-N highest-scoring templates.
      4. Applying acoustic modifier adjustments to their default parameters.
      5. Sorting the selected nodes into a standard signal-chain order.
      6. Filling Faust code skeletons with the adjusted parameter values.
    """

    # Maximum number of processing stages to include in a single chain.
    MAX_NODES = 3
    # Minimum score for a template to be included (avoids noise templates).
    SCORE_THRESHOLD = 0.15

    def __init__(self) -> None:
        self._templates = _load_templates()
        logger.info("ASTBuilder loaded %d DSP templates.", len(self._templates))

    def build(self, vibe: str) -> SignalAST:
        """
        Build and return a SignalAST for the given Vibe description.

        Args:
            vibe: Natural language Vibe string from the user.

        Returns:
            SignalAST with ordered nodes and a confidence score.
            Returns an empty-node AST (confidence=0) if no templates match.
        """
        tokens = _tokenise(vibe)
        logger.debug("ASTBuilder tokens: %s", tokens)

        if not tokens or not self._templates:
            return SignalAST(vibe_text=vibe, confidence=0.0)

        # Score all templates
        scored: list[tuple[float, list[str], dict]] = []
        for template in self._templates:
            score, matched = _score_template(template, tokens)
            if score >= self.SCORE_THRESHOLD:
                scored.append((score, matched, template))

        if not scored:
            logger.debug("ASTBuilder: no templates met score threshold for vibe='%s'", vibe)
            return SignalAST(vibe_text=vibe, confidence=0.0)

        # Sort by score descending, take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[: self.MAX_NODES]

        # Build ASTNodes
        nodes: list[ASTNode] = []
        for score, matched, template in selected:
            params = dict(template.get("default_parameters", {}))
            skeleton = _fill_skeleton(
                template.get("faust_skeleton", ""),
                params,
            )
            node = ASTNode(
                node_type=template.get("ast_node_type", "unknown"),
                template_id=template["id"],
                parameters=params,
                tags_matched=matched,
                faust_fragment=skeleton,
                metadata={"score": round(score, 4)},
            )
            nodes.append(node)

        # Apply modifier adjustments
        nodes = _apply_modifiers(nodes, tokens)

        # Re-fill skeletons with modified parameters
        for node in nodes:
            template = next(
                (t for t in self._templates if t["id"] == node.template_id), None
            )
            if template:
                node.faust_fragment = _fill_skeleton(
                    template.get("faust_skeleton", ""),
                    node.parameters,
                )

        # Sort into canonical chain order
        nodes = _sort_chain(nodes)

        # Overall confidence = mean of top-N scores
        confidence = sum(s for s, _, _ in selected) / len(selected)

        ast = SignalAST(
            nodes=nodes,
            vibe_text=vibe,
            confidence=round(confidence, 4),
        )

        logger.info(
            "ASTBuilder result | vibe='%s' nodes=%d confidence=%.3f chain=%s",
            vibe[:60],
            len(nodes),
            confidence,
            ast.to_faust_hint(),
        )
        return ast