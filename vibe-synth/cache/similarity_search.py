# =============================================================================
# File: cache/similarity_search.py
# Purpose: Cosine similarity search over the VectorStore. Given a query
#          embedding, finds the most semantically similar cached entry above
#          a configurable threshold. Supports batch queries, ranked results,
#          and threshold sweeping for offline evaluation in the experiments
#          notebooks. Designed to be swapped out for an ANN library (faiss,
#          hnswlib) when the cache grows beyond a few thousand entries.
# =============================================================================

import logging
import math
from dataclasses import dataclass, field

from cache.vector_store import VectorEntry, VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SearchHit:
    """
    A single similarity search result.

    Attributes:
        entry:       The matched VectorEntry.
        score:       Cosine similarity score in [-1, 1]; higher is more similar.
        rank:        1-indexed position in the result list (1 = best match).
    """
    entry: VectorEntry
    score: float
    rank:  int = 1


@dataclass
class SearchResult:
    """
    Aggregated output of a similarity search query.

    Attributes:
        hit:         True if the top result exceeds the similarity threshold.
        best:        The top-ranked SearchHit, or None if the store is empty.
        all_hits:    All results above the threshold, sorted by score descending.
        query_time_ms: Wall-clock time for the search in milliseconds.
    """
    hit:           bool
    best:          SearchHit | None         = None
    all_hits:      list[SearchHit]          = field(default_factory=list)
    query_time_ms: int                      = 0


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute the cosine similarity between two equal-length float vectors.

    Returns a value in [-1, 1]:
      1.0  → identical direction (semantically equivalent prompts)
      0.0  → orthogonal (unrelated prompts)
     -1.0  → opposite direction (rare for text embeddings)

    Args:
        a: Query embedding vector.
        b: Stored embedding vector.

    Returns:
        Cosine similarity score, or 0.0 if either vector is the zero vector.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vector dimension mismatch: query has {len(a)} dims, "
            f"stored entry has {len(b)} dims."
        )
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 0.0
    return dot / (mag_a * mag_b)


def dot_product(a: list[float], b: list[float]) -> float:
    """
    Raw dot product of two vectors. Faster than cosine when both vectors
    are unit-normalised (as produced by _simple_embed in cache_service.py).
    """
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Similarity searcher
# ---------------------------------------------------------------------------

class SimilaritySearch:
    """
    Brute-force cosine similarity search over a VectorStore.

    Complexity: O(n · d) per query, where n = number of cached entries and
    d = embedding dimensionality. Acceptable for n ≤ 5000; above that,
    replace the inner loop with a faiss or hnswlib index.

    Usage:
        searcher = SimilaritySearch(store, threshold=0.92)
        result   = searcher.search(query_embedding, generation_type="dsp")
        if result.hit:
            return result.best.entry.result
    """

    def __init__(
        self,
        store:     VectorStore,
        threshold: float = 0.92,
    ) -> None:
        """
        Args:
            store:     VectorStore instance to search.
            threshold: Minimum cosine similarity to count as a cache hit.
        """
        self.store     = store
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Primary search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding:  list[float],
        generation_type:  str,
        top_k:            int = 5,
    ) -> SearchResult:
        """
        Find the top-k most similar cached entries for a query embedding.

        Args:
            query_embedding: Float vector of the incoming prompt.
            generation_type: 'dsp' or 'shader' — only entries of this type
                             are considered.
            top_k:           Maximum number of results to return in all_hits.

        Returns:
            SearchResult with hit=True if the best score >= self.threshold.
        """
        import time
        t0 = time.perf_counter()

        scored: list[tuple[float, VectorEntry]] = []

        for entry_id, embedding in self.store.iter_embeddings(generation_type):
            entry = self.store.get(entry_id)
            if entry is None:
                continue
            try:
                score = cosine_similarity(query_embedding, embedding)
            except ValueError as exc:
                logger.warning("SimilaritySearch | dimension mismatch: %s", exc)
                continue
            scored.append((score, entry))

        query_time_ms = int((time.perf_counter() - t0) * 1000)

        if not scored:
            logger.debug(
                "SimilaritySearch | no entries for type='%s'", generation_type
            )
            return SearchResult(hit=False, query_time_ms=query_time_ms)

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build ranked hits above threshold
        all_hits: list[SearchHit] = []
        for rank, (score, entry) in enumerate(scored[:top_k], start=1):
            if score >= self.threshold:
                all_hits.append(SearchHit(entry=entry, score=score, rank=rank))

        best_score, best_entry = scored[0]
        hit = best_score >= self.threshold

        best_hit = SearchHit(entry=best_entry, score=best_score, rank=1) if hit else None

        logger.debug(
            "SimilaritySearch | type=%s entries_scanned=%d best_score=%.4f hit=%s time_ms=%d",
            generation_type, len(scored), best_score, hit, query_time_ms,
        )

        if hit:
            self.store.update_hit(best_entry.entry_id)

        return SearchResult(
            hit=hit,
            best=best_hit,
            all_hits=all_hits,
            query_time_ms=query_time_ms,
        )

    # ------------------------------------------------------------------
    # Batch search
    # ------------------------------------------------------------------

    def search_batch(
        self,
        query_embeddings: list[list[float]],
        generation_type:  str,
        top_k:            int = 1,
    ) -> list[SearchResult]:
        """
        Run multiple similarity queries in sequence.

        Args:
            query_embeddings: List of query vectors.
            generation_type:  'dsp' or 'shader'.
            top_k:            Max results per query.

        Returns:
            List of SearchResult objects in the same order as the input queries.
        """
        results = []
        for i, embedding in enumerate(query_embeddings):
            logger.debug(
                "SimilaritySearch | batch query %d/%d", i + 1, len(query_embeddings)
            )
            results.append(self.search(embedding, generation_type, top_k=top_k))
        return results

    # ------------------------------------------------------------------
    # Threshold sweep  (used in experiments notebooks)
    # ------------------------------------------------------------------

    def sweep_thresholds(
        self,
        query_embedding: list[float],
        generation_type: str,
        thresholds:      list[float] | None = None,
    ) -> dict[float, bool]:
        """
        Evaluate the search result across a range of threshold values.
        Useful for tuning settings.cache_similarity_threshold in the
        experiments/prompt_ablation.ipynb notebook.

        Args:
            query_embedding: Query vector.
            generation_type: 'dsp' or 'shader'.
            thresholds:      List of threshold values to test.
                             Defaults to [0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.99].

        Returns:
            Dict mapping threshold → hit (True/False).
        """
        if thresholds is None:
            thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.99]

        original_threshold = self.threshold
        sweep_results: dict[float, bool] = {}

        for t in thresholds:
            self.threshold = t
            result = self.search(query_embedding, generation_type, top_k=1)
            sweep_results[t] = result.hit

        # Restore original threshold
        self.threshold = original_threshold
        return sweep_results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def nearest_neighbours(
        self,
        query_embedding: list[float],
        generation_type: str,
        n:               int = 10,
    ) -> list[tuple[float, str, str]]:
        """
        Return the n nearest neighbours regardless of threshold.
        Used for debugging and cache quality inspection in notebooks.

        Returns:
            List of (score, entry_id, prompt_preview) tuples, sorted
            by score descending.
        """
        scored: list[tuple[float, str, str]] = []

        for entry_id, embedding in self.store.iter_embeddings(generation_type):
            try:
                score = cosine_similarity(query_embedding, embedding)
            except ValueError:
                continue
            entry = self.store.get(entry_id)
            prompt_preview = entry.prompt[:60] if entry else ""
            scored.append((score, entry_id, prompt_preview))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:n]

    def stats(self) -> dict:
        """
        Return combined stats from the store and the current threshold setting.
        """
        store_stats = self.store.stats()
        store_stats["search_threshold"] = self.threshold
        return store_stats