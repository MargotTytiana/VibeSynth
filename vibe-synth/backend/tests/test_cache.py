# =============================================================================
# File: backend/tests/test_cache.py
# Purpose: Unit tests for CacheService. Uses a temporary directory for the
#          index file so tests never touch the real cache on disk. Covers:
#          empty-cache misses, store and retrieve round-trips, similarity
#          threshold behaviour, FIFO eviction, cross-type isolation, and
#          index persistence across service restarts.
# =============================================================================

import json
import pytest
import tempfile
from pathlib import Path

from app.config import Settings
from app.services.cache_service import CacheService, _cosine_similarity, _simple_embed


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

def _make_settings(tmp_path: Path, **overrides) -> Settings:
    """Return a Settings instance pointing the cache index at a temp file."""
    index_file = str(tmp_path / "cache_index.json")
    defaults = dict(
        llm_provider="local",
        llm_api_key="x",
        cache_index_path=index_file,
        cache_similarity_threshold=0.92,
        cache_max_entries=5,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _make_cache(tmp_path: Path, **setting_overrides) -> CacheService:
    settings = _make_settings(tmp_path, **setting_overrides)
    return CacheService(settings=settings)


_SAMPLE_RESULT_DSP = {
    "faust_code":     "process = _;",
    "wasm_module_id": "wasm-abc123",
    "parameters":     [{"name": "gain", "type": "float", "range": [0, 2], "default": 1.0}],
}

_SAMPLE_RESULT_SHADER = {
    "shader_code":     "@fragment fn main() -> @location(0) vec4f { return vec4f(1.0); }",
    "target_platform": "webgpu",
    "parameters":      [],
}


# ---------------------------------------------------------------------------
# Pure-function unit tests (no I/O)
# ---------------------------------------------------------------------------

class TestCosineSimilarity:

    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_opposite_vectors_return_negative_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-9

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_symmetry(self):
        a = [0.6, 0.8]
        b = [0.8, 0.6]
        assert abs(_cosine_similarity(a, b) - _cosine_similarity(b, a)) < 1e-9


class TestSimpleEmbed:

    def test_returns_list_of_floats(self):
        vec = _simple_embed("test prompt")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_default_dimension_is_128(self):
        assert len(_simple_embed("anything")) == 128

    def test_custom_dimension_respected(self):
        assert len(_simple_embed("anything", dim=64)) == 64

    def test_deterministic_for_same_input(self):
        assert _simple_embed("hello world") == _simple_embed("hello world")

    def test_different_inputs_differ(self):
        assert _simple_embed("reverb") != _simple_embed("distortion")

    def test_vector_is_unit_normalised(self):
        import math
        vec = _simple_embed("normalised")
        mag = math.sqrt(sum(v * v for v in vec))
        assert abs(mag - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# CacheService integration tests (with temp filesystem)
# ---------------------------------------------------------------------------

class TestCacheServiceLookup:

    @pytest.mark.asyncio
    async def test_empty_cache_returns_miss(self, tmp_path):
        cache = _make_cache(tmp_path)
        hit, score, result = await cache.lookup("reverb", "dsp")
        assert hit   is False
        assert score == 0.0
        assert result is None

    @pytest.mark.asyncio
    async def test_exact_prompt_returns_hit(self, tmp_path):
        """Storing and immediately looking up the same prompt should be a hit."""
        cache = _make_cache(tmp_path, cache_similarity_threshold=0.5)
        await cache.store("spring reverb", "dsp", _SAMPLE_RESULT_DSP)
        hit, score, result = await cache.lookup("spring reverb", "dsp")
        assert hit is True
        assert score > 0.5
        assert result == _SAMPLE_RESULT_DSP

    @pytest.mark.asyncio
    async def test_miss_below_threshold(self, tmp_path):
        """A very different prompt should fall below the threshold."""
        cache = _make_cache(tmp_path, cache_similarity_threshold=0.99)
        await cache.store("spring reverb", "dsp", _SAMPLE_RESULT_DSP)
        hit, score, result = await cache.lookup(
            "completely unrelated shader prompt xyz", "dsp"
        )
        assert hit is False
        assert result is None

    @pytest.mark.asyncio
    async def test_cross_type_isolation(self, tmp_path):
        """A DSP cache entry must not match a shader lookup."""
        cache = _make_cache(tmp_path, cache_similarity_threshold=0.5)
        await cache.store("spring reverb", "dsp", _SAMPLE_RESULT_DSP)
        hit, _, _ = await cache.lookup("spring reverb", "shader")
        assert hit is False

    @pytest.mark.asyncio
    async def test_multiple_entries_returns_best_match(self, tmp_path):
        """lookup() returns the entry with the highest similarity score."""
        cache = _make_cache(tmp_path, cache_similarity_threshold=0.0)
        await cache.store("warm reverb",   "dsp", {"label": "warm"})
        await cache.store("spring reverb", "dsp", {"label": "spring"})
        _, _, result = await cache.lookup("spring reverb", "dsp")
        assert result == {"label": "spring"}


class TestCacheServiceStore:

    @pytest.mark.asyncio
    async def test_store_increments_entry_count(self, tmp_path):
        cache = _make_cache(tmp_path)
        assert cache.stats()["total_entries"] == 0
        await cache.store("reverb", "dsp", _SAMPLE_RESULT_DSP)
        assert cache.stats()["total_entries"] == 1

    @pytest.mark.asyncio
    async def test_store_persists_to_disk(self, tmp_path):
        """After store(), the index file must exist and contain valid JSON."""
        cache = _make_cache(tmp_path)
        await cache.store("reverb", "dsp", _SAMPLE_RESULT_DSP)
        index_path = Path(cache.index_path)
        assert index_path.exists()
        with index_path.open() as f:
            data = json.load(f)
        assert len(data["entries"]) == 1

    @pytest.mark.asyncio
    async def test_fifo_eviction_at_max_entries(self, tmp_path):
        """When max_entries is reached the oldest entry is evicted."""
        cache = _make_cache(tmp_path, cache_max_entries=3)
        for i in range(3):
            await cache.store(f"prompt_{i}", "dsp", {"index": i})

        # All three entries are present
        assert cache.stats()["total_entries"] == 3

        # Adding a fourth evicts the first
        await cache.store("prompt_new", "dsp", {"index": 99})
        assert cache.stats()["total_entries"] == 3

        # The entry at index 0 should be gone; index 99 should be present
        prompts_stored = [e["prompt"] for e in cache._entries]
        assert "prompt_0" not in prompts_stored
        assert "prompt_new" in prompts_stored

    @pytest.mark.asyncio
    async def test_stats_tracks_type_counts(self, tmp_path):
        cache = _make_cache(tmp_path)
        await cache.store("reverb",  "dsp",    _SAMPLE_RESULT_DSP)
        await cache.store("bloom",   "shader",  _SAMPLE_RESULT_SHADER)
        await cache.store("eq",      "dsp",    _SAMPLE_RESULT_DSP)
        stats = cache.stats()
        assert stats["dsp_entries"]    == 2
        assert stats["shader_entries"] == 1
        assert stats["total_entries"]  == 3


class TestCacheServicePersistence:

    @pytest.mark.asyncio
    async def test_index_survives_service_restart(self, tmp_path):
        """
        A new CacheService instance pointing at the same index file should
        load the entries stored by the previous instance.
        """
        # First instance — store an entry
        cache_a = _make_cache(tmp_path, cache_similarity_threshold=0.5)
        await cache_a.store("spring reverb", "dsp", _SAMPLE_RESULT_DSP)

        # Second instance — load from same file
        cache_b = _make_cache(tmp_path, cache_similarity_threshold=0.5)
        assert cache_b.stats()["total_entries"] == 1

        hit, score, result = await cache_b.lookup("spring reverb", "dsp")
        assert hit is True
        assert result == _SAMPLE_RESULT_DSP

    @pytest.mark.asyncio
    async def test_corrupted_index_file_starts_fresh(self, tmp_path):
        """A malformed JSON index file should be silently ignored."""
        index_path = tmp_path / "cache_index.json"
        index_path.write_text("{ this is not valid json }", encoding="utf-8")

        cache = _make_cache(tmp_path)
        assert cache.stats()["total_entries"] == 0