"""
tests/unit/test_image_cache.py

Unit tests for core/image_cache.py (_LRUImageCache).

Covers
------
put / get         – basic round-trip, returns correct array and meta
LRU eviction      – oldest item evicted when item limit reached
byte-limit evict  – oldest item evicted when byte limit reached
invalidate        – removes a single key
clear             – empties the cache
re-put            – updating an existing key refreshes its MRU position
"""

import numpy as np
import pytest


def _make_array(shape=(4, 4), fill=1.0):
    return np.full(shape, fill, dtype=np.float32)


def _make_cache(max_items=8, max_bytes=100 * 1024 * 1024):
    """Return a fresh _LRUImageCache instance with custom limits."""
    from core.image_cache import _LRUImageCache
    return _LRUImageCache(max_bytes=max_bytes, max_items=max_items)


# ─────────────────────────────────────────────────────────────────────────────
# Basic put / get
# ─────────────────────────────────────────────────────────────────────────────

class TestPutGet:
    def test_get_after_put_returns_array(self):
        cache = _make_cache()
        arr = _make_array(fill=42.0)
        cache.put("key1", arr, {"info": "test"})
        result = cache.get("key1")
        assert result is not None
        got_arr, got_meta = result
        np.testing.assert_array_equal(got_arr, arr)

    def test_get_after_put_returns_metadata(self):
        cache = _make_cache()
        arr = _make_array()
        meta = {"modality": "MRI", "shape": [4, 4]}
        cache.put("key1", arr, meta)
        _, got_meta = cache.get("key1")
        assert got_meta == meta

    def test_get_missing_key_returns_none(self):
        cache = _make_cache()
        assert cache.get("nonexistent") is None

    def test_current_items_increments(self):
        cache = _make_cache()
        assert cache.current_items == 0
        cache.put("a", _make_array(), {})
        assert cache.current_items == 1
        cache.put("b", _make_array(), {})
        assert cache.current_items == 2

    def test_current_bytes_tracked(self):
        cache = _make_cache()
        arr = _make_array(shape=(10, 10))
        cache.put("k", arr, {})
        assert cache.current_bytes == arr.nbytes


# ─────────────────────────────────────────────────────────────────────────────
# LRU eviction by item count
# ─────────────────────────────────────────────────────────────────────────────

class TestItemLimitEviction:
    def test_oldest_item_evicted_at_limit(self):
        cache = _make_cache(max_items=3)
        cache.put("first", _make_array(fill=1.0), {})
        cache.put("second", _make_array(fill=2.0), {})
        cache.put("third", _make_array(fill=3.0), {})
        # Cache is full; adding a fourth should evict "first" (oldest)
        cache.put("fourth", _make_array(fill=4.0), {})
        assert cache.get("first") is None
        assert cache.get("fourth") is not None

    def test_item_count_does_not_exceed_max(self):
        cache = _make_cache(max_items=3)
        for i in range(10):
            cache.put(f"key{i}", _make_array(), {})
        assert cache.current_items <= 3

    def test_recently_used_item_not_evicted(self):
        """Accessing 'first' promotes it to MRU, so 'second' should be evicted next."""
        cache = _make_cache(max_items=3)
        cache.put("first", _make_array(fill=1.0), {})
        cache.put("second", _make_array(fill=2.0), {})
        cache.put("third", _make_array(fill=3.0), {})
        # Access "first" to promote it
        cache.get("first")
        # Now "second" is the LRU; adding "fourth" should evict "second"
        cache.put("fourth", _make_array(fill=4.0), {})
        assert cache.get("second") is None
        assert cache.get("first") is not None


# ─────────────────────────────────────────────────────────────────────────────
# LRU eviction by byte budget
# ─────────────────────────────────────────────────────────────────────────────

class TestByteLimitEviction:
    def test_evicts_when_byte_budget_exceeded(self):
        # Each 4×4 float32 array is 64 bytes; limit to 96 bytes → max 1.5 arrays
        arr_bytes = 4 * 4 * 4  # 64 bytes
        cache = _make_cache(max_items=100, max_bytes=arr_bytes + 10)
        cache.put("a", _make_array(), {})
        # Adding "b" would exceed budget, so "a" must be evicted
        cache.put("b", _make_array(), {})
        assert cache.get("a") is None
        assert cache.get("b") is not None

    def test_bytes_stay_within_budget(self):
        arr_bytes = 4 * 4 * 4  # 64 bytes per array
        budget = arr_bytes * 3 + 10
        cache = _make_cache(max_items=100, max_bytes=budget)
        for i in range(10):
            cache.put(f"k{i}", _make_array(), {})
        assert cache.current_bytes <= budget


# ─────────────────────────────────────────────────────────────────────────────
# invalidate
# ─────────────────────────────────────────────────────────────────────────────

class TestInvalidate:
    def test_invalidated_key_returns_none(self):
        cache = _make_cache()
        cache.put("x", _make_array(), {})
        cache.invalidate("x")
        assert cache.get("x") is None

    def test_invalidate_missing_key_is_safe(self):
        cache = _make_cache()
        cache.invalidate("does_not_exist")  # should not raise

    def test_byte_accounting_updated_after_invalidate(self):
        cache = _make_cache()
        arr = _make_array(shape=(10, 10))
        cache.put("z", arr, {})
        expected_bytes = arr.nbytes
        assert cache.current_bytes == expected_bytes
        cache.invalidate("z")
        assert cache.current_bytes == 0


# ─────────────────────────────────────────────────────────────────────────────
# clear
# ─────────────────────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_empties_cache(self):
        cache = _make_cache()
        for i in range(5):
            cache.put(f"k{i}", _make_array(), {})
        cache.clear()
        assert cache.current_items == 0

    def test_clear_resets_byte_count(self):
        cache = _make_cache()
        cache.put("a", _make_array(shape=(100, 100)), {})
        cache.clear()
        assert cache.current_bytes == 0

    def test_get_after_clear_returns_none(self):
        cache = _make_cache()
        cache.put("a", _make_array(), {})
        cache.clear()
        assert cache.get("a") is None


# ─────────────────────────────────────────────────────────────────────────────
# re-put (update existing key)
# ─────────────────────────────────────────────────────────────────────────────

class TestRePut:
    def test_reput_updates_value(self):
        cache = _make_cache()
        cache.put("key", _make_array(fill=1.0), {"v": 1})
        cache.put("key", _make_array(fill=2.0), {"v": 2})
        arr, meta = cache.get("key")
        assert meta["v"] == 2
        assert arr[0, 0] == pytest.approx(2.0)

    def test_reput_does_not_duplicate_item_count(self):
        cache = _make_cache()
        cache.put("key", _make_array(), {})
        cache.put("key", _make_array(), {})
        assert cache.current_items == 1

    def test_reput_promotes_to_mru(self):
        """Re-putting the oldest key should protect it from the next eviction."""
        cache = _make_cache(max_items=2)
        cache.put("a", _make_array(fill=1.0), {})
        cache.put("b", _make_array(fill=2.0), {})
        # Re-put "a" → now "b" is LRU
        cache.put("a", _make_array(fill=1.5), {})
        # Adding "c" should evict "b"
        cache.put("c", _make_array(fill=3.0), {})
        assert cache.get("b") is None
        assert cache.get("a") is not None
