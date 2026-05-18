# =============================================================================
# File: backend/app/services/cache_service.py
# Purpose: Vector-based semantic cache for generated DSP and shader results.
#          Embeds prompt text into a vector space and performs cosine similarity
#          search so that near-duplicate Vibe descriptions reuse an existing
#          compiled result instead of triggering a fresh LLM generation.
#          The index is persisted to disk as cache/cache_index.json.
# =============================================================================

import json
import math
import time
import logging
import hashlib
from pathlib import Path
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight pure-Python cosine similarity (no numpy required at import time)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two equal-length vectors.
    Returns a value in [-1, 1]; 1.0 means identical direction.
    """
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _simple_embed(text: str, dim: int = 128) -> list[float]:
    """
    Deterministic character-level hash embedding used as a fallback when no
    embedding API is available (e.g. local development without API keys).

    NOT suitable for production — replace with a real embedding model
    (text-embedding-3-small, sentence-transformers, etc.) via _embed().

    Strategy: SHA-256 hash of lowercased text, repeated to fill `dim` floats,
    each normalised to [-1, 1].
    """
    digest = hashlib.sha256(text.lower().encode()).digest()
    # Repeat digest bytes to cover dim floats
    raw = (digest * (dim // len(digest) + 1))[:dim]
    vec = [(b / 127.5) - 1.0 for b in raw]
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]


class CacheService:
    """
    Semantic cache that stores (prompt_embedding, generation_result) pairs
    and retrieves the most similar cached result for a new prompt.

    Index structure (cache_index.json):
    {
      "entries": [
        {
          "id":             "<sha256 of prompt>",
          "prompt":         "<original prompt text>",
          "generation_type": "dsp" | "shader",
          "embedding":      [<float>, ...],
          "result":         { <full API response dict> },
          "created_at":     <unix timestamp>
        },
        ...
      ]
    }
    """

    def __init__(self, settings: Settings) -> None:
        self.settings   = settings
        self.index_path = Path(settings.cache_index_path)
        self.threshold  = settings.cache_similarity_threshold
        self.max_entries = settings.cache_max_entries
        self._entries: list[dict] = []
        self._load_index()

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Load existing cache entries from disk, or start with an empty index."""
        if self.index_path.exists():
            try:
                with self.index_path.open(encoding="utf-8") as f:
                    data = json.load(f)
                self._entries = data.get("entries", [])
                logger.info(
                    "Cache index loaded: %d entries from %s",
                    len(self._entries), self.index_path,
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load cache index (%s) — starting fresh.", exc)
                self._entries = []
        else:
            logger.info("No cache index found at %s — starting fresh.", self.index_path)
            self._entries = []

    def _save_index(self) -> None:
        """Persist the current in-memory index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump({"entries": self._entries}, f, indent=2)
        logger.debug("Cache index saved: %d entries.", len(self._entries))

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def _embed(self, text: str) -> list[float]:
        """
        Embed a prompt string into a float vector.

        Production path: call an embedding API (OpenAI, Cohere, etc.).
        Fallback: use the deterministic hash embedding for offline / test use.

        To switch to a real embedding model, replace the body below with an
        async API call and return the resulting vector.
        """
        return _simple_embed(text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def lookup(
        self, prompt: str, generation_type: str
    ) -> tuple[bool, float, dict | None]:
        """
        Search for a cached result whose prompt is semantically similar to
        the given prompt.

        Args:
            prompt:          The incoming Vibe description.
            generation_type: 'dsp' or 'shader' — only entries of the same
                             type are considered.

        Returns:
            (hit, similarity_score, cached_result)
            - hit: True if similarity_score >= settings.cache_similarity_threshold
            - similarity_score: highest cosine similarity found (0.0 if no entries)
            - cached_result: the stored result dict, or None on a miss
        """
        if not self._entries:
            return False, 0.0, None

        query_vec = await self._embed(prompt)
        best_score  = -1.0
        best_result = None

        for entry in self._entries:
            if entry.get("generation_type") != generation_type:
                continue
            score = _cosine_similarity(query_vec, entry["embedding"])
            if score > best_score:
                best_score  = score
                best_result = entry["result"]

        hit = best_score >= self.threshold
        logger.debug(
            "Cache lookup | type=%s score=%.4f hit=%s", generation_type, best_score, hit
        )
        return hit, best_score, best_result if hit else None

    async def store(
        self, prompt: str, generation_type: str, result: dict[str, Any]
    ) -> None:
        """
        Add a new entry to the cache index and persist it to disk.

        If the cache has reached settings.cache_max_entries, the oldest entry
        is evicted before inserting the new one (FIFO eviction).

        Args:
            prompt:          The Vibe description that produced the result.
            generation_type: 'dsp' or 'shader'.
            result:          The full API response dict to store.
        """
        entry_id  = hashlib.sha256(
            f"{generation_type}:{prompt}".encode()
        ).hexdigest()[:16]
        embedding = await self._embed(prompt)

        new_entry = {
            "id":              entry_id,
            "prompt":          prompt,
            "generation_type": generation_type,
            "embedding":       embedding,
            "result":          result,
            "created_at":      time.time(),
        }

        # Evict oldest entry if at capacity
        if len(self._entries) >= self.max_entries:
            evicted = self._entries.pop(0)
            logger.debug("Cache evicted oldest entry id=%s", evicted.get("id"))

        self._entries.append(new_entry)
        self._save_index()
        logger.info(
            "Cache stored | id=%s type=%s total_entries=%d",
            entry_id, generation_type, len(self._entries),
        )

    def stats(self) -> dict:
        """
        Return a summary of the current cache state — useful for the
        /health endpoint and monitoring dashboards.
        """
        dsp_count    = sum(1 for e in self._entries if e.get("generation_type") == "dsp")
        shader_count = sum(1 for e in self._entries if e.get("generation_type") == "shader")
        return {
            "total_entries":  len(self._entries),
            "dsp_entries":    dsp_count,
            "shader_entries": shader_count,
            "max_entries":    self.max_entries,
            "threshold":      self.threshold,
            "index_path":     str(self.index_path),
        }