# =============================================================================
# File: cache/vector_store.py
# Purpose: Low-level vector storage layer for the semantic cache. Manages the
#          in-memory and on-disk representation of prompt embeddings. Provides
#          CRUD operations (add, get, delete, list) on embedding entries
#          independent of similarity search logic. The CacheService in
#          app/services/cache_service.py sits above this layer; similarity
#          search is handled by cache/similarity_search.py.
# =============================================================================

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VectorEntry:
    """
    A single stored embedding with its associated metadata.

    Attributes:
        entry_id:        Unique identifier (SHA-256 prefix of prompt+type).
        prompt:          Original natural language Vibe description.
        generation_type: 'dsp' or 'shader'.
        embedding:       Float vector representing the prompt in latent space.
        result:          Serialisable dict of the cached API response.
        created_at:      Unix timestamp of insertion.
        hit_count:       Number of times this entry has been returned as a hit.
        last_hit_at:     Unix timestamp of the most recent cache hit, or None.
    """
    entry_id:        str
    prompt:          str
    generation_type: str
    embedding:       list[float]
    result:          dict
    created_at:      float        = field(default_factory=time.time)
    hit_count:       int          = 0
    last_hit_at:     float | None = None

    def record_hit(self) -> None:
        """Increment the hit counter and update last_hit_at."""
        self.hit_count  += 1
        self.last_hit_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VectorEntry":
        return cls(**data)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class VectorStore:
    """
    In-memory key-value store for VectorEntry objects with JSON persistence.

    Operations are O(n) for iteration (small n expected — cache_max_entries
    is typically ≤ 1000) and O(1) for id-keyed lookup.

    The store does NOT perform similarity search — that is delegated to
    cache/similarity_search.py. This class only manages storage and I/O.

    Usage:
        store = VectorStore(index_path="cache/cache_index.json")
        store.add(entry)
        entry = store.get("some-id")
        store.save()
    """

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self._entries:  dict[str, VectorEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load entries from the JSON index file into memory."""
        if not self.index_path.exists():
            logger.info(
                "VectorStore | no index file at %s — starting empty.", self.index_path
            )
            return

        try:
            with self.index_path.open(encoding="utf-8") as f:
                data = json.load(f)

            raw_entries = data.get("entries", [])
            for raw in raw_entries:
                try:
                    entry = VectorEntry.from_dict(raw)
                    self._entries[entry.entry_id] = entry
                except (TypeError, KeyError) as exc:
                    logger.warning(
                        "VectorStore | skipping malformed entry: %s", exc
                    )

            logger.info(
                "VectorStore | loaded %d entries from %s",
                len(self._entries), self.index_path,
            )

        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "VectorStore | could not load index (%s) — starting empty.", exc
            )
            self._entries = {}

    def save(self) -> None:
        """Persist all in-memory entries to the JSON index file."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "saved_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entry_count":    len(self._entries),
            "entries":        [e.to_dict() for e in self._entries.values()],
        }
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.debug(
            "VectorStore | saved %d entries to %s", len(self._entries), self.index_path
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, entry: VectorEntry, *, evict_oldest: bool = True, max_size: int = 1000) -> None:
        """
        Insert or replace a VectorEntry in the store.

        If max_size is reached and evict_oldest=True, the entry with the
        oldest created_at timestamp is removed before insertion (FIFO).

        Args:
            entry:        The VectorEntry to add.
            evict_oldest: Whether to evict the oldest entry when at capacity.
            max_size:     Maximum number of entries to keep.
        """
        if len(self._entries) >= max_size and evict_oldest:
            oldest_id = min(
                self._entries, key=lambda k: self._entries[k].created_at
            )
            removed = self._entries.pop(oldest_id)
            logger.debug(
                "VectorStore | evicted oldest entry id=%s created_at=%.0f",
                removed.entry_id, removed.created_at,
            )

        self._entries[entry.entry_id] = entry
        logger.debug("VectorStore | added entry id=%s", entry.entry_id)

    def get(self, entry_id: str) -> VectorEntry | None:
        """Return the VectorEntry for the given ID, or None if not found."""
        return self._entries.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        """
        Remove an entry by ID.

        Returns:
            True if the entry existed and was removed, False otherwise.
        """
        if entry_id in self._entries:
            del self._entries[entry_id]
            logger.debug("VectorStore | deleted entry id=%s", entry_id)
            return True
        return False

    def update_hit(self, entry_id: str) -> None:
        """Increment the hit counter for an entry and persist."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.record_hit()
            logger.debug(
                "VectorStore | hit recorded for id=%s total_hits=%d",
                entry_id, entry.hit_count,
            )

    # ------------------------------------------------------------------
    # Iteration and queries
    # ------------------------------------------------------------------

    def all(self) -> list[VectorEntry]:
        """Return all stored entries as a list (order not guaranteed)."""
        return list(self._entries.values())

    def by_type(self, generation_type: str) -> list[VectorEntry]:
        """Return all entries of a specific generation type ('dsp' or 'shader')."""
        return [
            e for e in self._entries.values()
            if e.generation_type == generation_type
        ]

    def iter_embeddings(
        self, generation_type: str | None = None
    ) -> Iterator[tuple[str, list[float]]]:
        """
        Yield (entry_id, embedding) pairs, optionally filtered by type.
        Used by the similarity search layer to avoid materialising all entries.
        """
        for entry_id, entry in self._entries.items():
            if generation_type is None or entry.generation_type == generation_type:
                yield entry_id, entry.embedding

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, entry_id: str) -> bool:
        return entry_id in self._entries

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary dict for monitoring and health endpoints."""
        all_entries = list(self._entries.values())
        dsp_count    = sum(1 for e in all_entries if e.generation_type == "dsp")
        shader_count = sum(1 for e in all_entries if e.generation_type == "shader")
        total_hits   = sum(e.hit_count for e in all_entries)
        most_hit     = max(all_entries, key=lambda e: e.hit_count, default=None)

        return {
            "total_entries":   len(all_entries),
            "dsp_entries":     dsp_count,
            "shader_entries":  shader_count,
            "total_hits":      total_hits,
            "most_hit_id":     most_hit.entry_id if most_hit else None,
            "most_hit_count":  most_hit.hit_count if most_hit else 0,
            "index_path":      str(self.index_path),
        }