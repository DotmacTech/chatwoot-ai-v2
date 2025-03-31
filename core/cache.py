from functools import wraps
import json
from datetime import timedelta
from typing import Optional, Any, Callable
import redis
from core.config import settings
from services.redis_service import RedisService
import logging

logger = logging.getLogger(__name__)

class Cache:
    """Redis-based cache with in-memory fallback"""

    def __init__(self, redis_service: RedisService):
        """Initialize Cache with a RedisService instance."""
        if not redis_service or not redis_service.client:
            logger.error("RedisService (or its client) not available for CacheManager. Caching will be disabled.")
            # Consider raising an error if Redis is essential
            # raise ValueError("Valid RedisService instance is required for CacheManager")
        self.redis_service = redis_service

    def __call__(self, ttl: int = 300, key_prefix: str = "", key_builder: Optional[Callable] = None):
        """Cache decorator
        
        Args:
            ttl (int): Time to live in seconds (default: 300)
            key_prefix (str): Prefix for cache key
            key_builder (callable): Function to build cache key from arguments
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Build cache key
                if key_builder:
                    key = f"{key_prefix}:{key_builder(*args, **kwargs)}"
                else:
                    key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"

                # Try to get from cache
                cached = await self.redis_service.get(key)
                if cached:
                    try:
                        loaded_data = json.loads(cached)
                        return loaded_data
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to load cached data for key {key}: {e}", exc_info=True)

                # If not in cache, execute function
                result = await func(*args, **kwargs)

                # Cache the result
                try:
                    await self.redis_service.set(key, json.dumps(result), ex=ttl)
                except Exception as e:
                    logger.error(f"Failed to cache result for key {key}: {e}", exc_info=True)

                return result
            return wrapper
        return decorator

def cached(
    expire: timedelta = timedelta(minutes=5),
    key_prefix: str = "",
    key_builder: Optional[Callable] = None
) -> Callable:
    """Cache decorator that uses the global cache instance
    
    Args:
        expire (timedelta): Time to live
        key_prefix (str): Prefix for cache key
        key_builder (callable): Function to build cache key from arguments
    """
    return Cache(RedisService())(ttl=int(expire.total_seconds()), key_prefix=key_prefix, key_builder=key_builder)
