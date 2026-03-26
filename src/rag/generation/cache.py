from cachetools import TTLCache

from rag.core.config import Settings
from rag.observability.logging import get_logger

logger = get_logger(__name__)

_query_cache: TTLCache | None = None


def get_query_cache(settings: Settings) -> TTLCache:
    global _query_cache
    if _query_cache is None:
        _query_cache = TTLCache(
            maxsize=settings.cache_max_size,
            ttl=settings.cache_ttl_seconds,
        )
    return _query_cache


def cached_query(cache: TTLCache, question: str, k: int, search_type: str):
    key = f"{question}:{k}:{search_type}"
    result = cache.get(key)
    if result is not None:
        logger.info("cache_hit", question_length=len(question))
    return result


def store_query(cache: TTLCache, question: str, k: int, search_type: str, result: dict):
    key = f"{question}:{k}:{search_type}"
    cache[key] = result
    logger.info("cache_stored", question_length=len(question))
