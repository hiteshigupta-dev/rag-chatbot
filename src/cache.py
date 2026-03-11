"""
Cache module.
Implements caching for query responses.
"""

import hashlib
import json
import time
import logging
from pathlib import Path
from typing import Any, Optional, Dict

from .config import CACHE_DIR, QUERY_CACHE_TTL, MAX_CACHE_SIZE


logger = logging.getLogger(__name__)


class Cache:
    """
    Cache system with memory + disk persistence.
    """

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        ttl: int = QUERY_CACHE_TTL,
        max_size: int = MAX_CACHE_SIZE
    ):

        self.cache_dir = cache_dir
        self.ttl = ttl
        self.max_size = max_size

        self.memory_cache: Dict[str, Dict[str, Any]] = {}

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_query_hash(self, query: str) -> str:
        """
        Generate deterministic query hash.
        """

        normalized = query.lower().strip()

        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _get_cache_path(self, query_hash: str) -> Path:
        """
        Path for cache file.
        """

        return self.cache_dir / f"{query_hash}.json"

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result.
        """

        query_hash = self._get_query_hash(query)

        # --------------------------
        # Memory cache
        # --------------------------

        entry = self.memory_cache.get(query_hash)

        if entry:

            if time.time() - entry["timestamp"] < self.ttl:
                logger.info("Cache hit (memory)")
                return entry["data"]

            else:
                del self.memory_cache[query_hash]

        # --------------------------
        # Disk cache
        # --------------------------

        cache_path = self._get_cache_path(query_hash)

        if cache_path.exists():

            try:

                with open(cache_path, "r", encoding="utf-8") as f:
                    entry = json.load(f)

                if time.time() - entry["timestamp"] < self.ttl:

                    self.memory_cache[query_hash] = entry

                    logger.info("Cache hit (disk)")

                    return entry["data"]

                else:

                    cache_path.unlink(missing_ok=True)

            except Exception as e:

                logger.warning("Cache read failed: %s", e)

        return None

    def set(self, query: str, data: Dict[str, Any]) -> None:
        """
        Store cache entry.
        """

        query_hash = self._get_query_hash(query)

        entry = {
            "query": query,
            "data": data,
            "timestamp": time.time()
        }

        self.memory_cache[query_hash] = entry

        # --------------------------
        # Memory eviction
        # --------------------------

        if len(self.memory_cache) > self.max_size:

            oldest_key = min(
                self.memory_cache,
                key=lambda k: self.memory_cache[k]["timestamp"]
            )

            del self.memory_cache[oldest_key]

        # --------------------------
        # Disk write
        # --------------------------

        cache_path = self._get_cache_path(query_hash)

        try:

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)

            logger.info("Response cached")

        except Exception as e:

            logger.warning("Cache write failed: %s", e)

    def clear(self) -> None:
        """
        Clear all cache (query cache and LLM response cache).
        """

        self.memory_cache.clear()

        for cache_file in self.cache_dir.glob("*.json"):

            try:
                cache_file.unlink()
            except Exception:
                pass

        try:
            from .llm import clear_llm_response_cache
            clear_llm_response_cache()
        except Exception:
            pass
        try:
            from .langchain_rag import clear_langchain_llm_cache
            clear_langchain_llm_cache()
        except Exception:
            pass

        logger.info("Cache cleared")


# ---------------------------------------------------
# Global singleton
# ---------------------------------------------------

_cache = None


def get_cache() -> Cache:
    """
    Get global cache instance.
    """

    global _cache

    if _cache is None:
        _cache = Cache()

    return _cache


def get_cached_response(query: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached query result.
    """

    cache = get_cache()

    return cache.get(query)


def cache_response(query: str, data: Dict[str, Any]) -> None:
    """
    Store cached query result.
    """

    cache = get_cache()

    cache.set(query, data)


def clear_cache() -> None:
    """
    Clear all cache.
    """

    cache = get_cache()

    cache.clear()