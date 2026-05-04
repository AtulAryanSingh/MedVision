"""
core/image_cache.py

What this module does:
  Provides a thread-safe in-memory LRU (Least-Recently-Used) cache for
  loaded medical image arrays and their metadata.

Why it exists:
  Loading NIfTI / DICOM volumes from disk is CPU-heavy and I/O-bound.
  Caching prevents re-reading and re-decoding the same file on every API
  request while keeping total resident memory bounded by both item count
  and aggregate byte size.

Performance notes:
  - Cache key is the file path as returned by find_uploaded_file() / the
    upload save path (both rooted at the same real directory).
  - Stored values are raw NumPy arrays; no serialization overhead.
  - LRU eviction (oldest-used first) ensures large volumes do not accumulate.
  - Limits are configurable via environment variables:
      MEDVISION_CACHE_MB    – max total cache size in MiB (default: 2048)
      MEDVISION_CACHE_ITEMS – max number of cached entries   (default:   64)

Thread safety:
  All public methods acquire a threading.Lock before mutating internal
  state.  This is correct for the asyncio + thread-pool usage pattern:
  the blocking image load runs in a thread-pool worker via
  loop.run_in_executor(), while cache lookups and insertions happen in the
  event-loop coroutine (before / after the await).  Lock contention is
  therefore minimized: the only concurrent accessors are multiple
  simultaneous coroutines performing a lookup before their respective
  executor calls, which is a fast, non-blocking critical section.
"""

import os
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ── Configuration (overridable via environment variables) ─────────────────────

_DEFAULT_MAX_BYTES: int = (
    int(os.environ.get("MEDVISION_CACHE_MB", "2048")) * 1024 * 1024
)
_DEFAULT_MAX_ITEMS: int = int(os.environ.get("MEDVISION_CACHE_ITEMS", "64"))


# ── Cache implementation ───────────────────────────────────────────────────────

class _LRUImageCache:
    """
    Thread-safe in-memory LRU cache for medical image arrays.

    Eviction policy
    ---------------
    When inserting a new item would cause total bytes to exceed *max_bytes*
    OR the item count to reach *max_items*, the least-recently-used entry is
    evicted first.  Eviction continues until both limits are satisfied.
    A single item whose size alone exceeds *max_bytes* is still cached (it
    becomes the sole entry until it is itself evicted by a later put()).

    Diagnostics
    -----------
    current_bytes  – total NumPy array bytes held
    current_items  – number of entries in the cache
    max_bytes      – configured byte limit
    max_items      – configured item limit
    """

    def __init__(
        self,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        max_items: int = _DEFAULT_MAX_ITEMS,
    ) -> None:
        self._max_bytes = max_bytes
        self._max_items = max_items
        # OrderedDict: insertion / move_to_end order encodes recency (oldest first)
        self._store: OrderedDict[
            str, Tuple[np.ndarray, Dict[str, Any], int]
        ] = OrderedDict()
        self._total_bytes: int = 0
        self._lock = threading.Lock()

    # ── Public interface ───────────────────────────────────────────────────

    def get(
        self, key: str
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Return *(array, metadata)* for *key* and promote it to MRU position.

        Returns ``None`` on a cache miss.
        """
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)       # mark as most-recently used
            arr, meta, _ = self._store[key]
            return arr, meta

    def put(
        self, key: str, arr: np.ndarray, meta: Dict[str, Any]
    ) -> None:
        """
        Insert *(arr, meta)* under *key*, evicting LRU entries as needed.

        Re-inserting an existing key refreshes its position and updates
        its size accounting.
        """
        nbytes = arr.nbytes
        with self._lock:
            # Remove existing entry first to handle re-puts and size updates
            if key in self._store:
                _, _, old_bytes = self._store.pop(key)
                self._total_bytes -= old_bytes

            # Evict least-recently-used items until limits can be satisfied
            while self._store and (
                self._total_bytes + nbytes > self._max_bytes
                or len(self._store) >= self._max_items
            ):
                _, (_, _, evicted_bytes) = self._store.popitem(last=False)
                self._total_bytes -= evicted_bytes

            self._store[key] = (arr, meta, nbytes)
            self._total_bytes += nbytes

    def invalidate(self, key: str) -> None:
        """
        Remove *key* from the cache.

        Call this after a file on disk has been replaced so that the next
        request reloads the updated data rather than serving a stale entry.
        """
        with self._lock:
            if key in self._store:
                _, _, nbytes = self._store.pop(key)
                self._total_bytes -= nbytes

    def clear(self) -> None:
        """Evict every cached entry and reset byte accounting."""
        with self._lock:
            self._store.clear()
            self._total_bytes = 0

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def current_bytes(self) -> int:
        """Total NumPy array bytes currently held by the cache."""
        with self._lock:
            return self._total_bytes

    @property
    def current_items(self) -> int:
        """Number of entries currently in the cache."""
        with self._lock:
            return len(self._store)

    @property
    def max_bytes(self) -> int:
        """Configured byte capacity limit."""
        return self._max_bytes

    @property
    def max_items(self) -> int:
        """Configured item count limit."""
        return self._max_items


# ── Module-level singleton ─────────────────────────────────────────────────────

#: Shared cache instance used by the entire application.
#: Import this object wherever you need cached image access.
image_cache = _LRUImageCache()
