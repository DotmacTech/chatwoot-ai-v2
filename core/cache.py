from functools import wraps
import json
from datetime import timedelta
from typing import Optional, Any, Callable
import redis
from core.config import settings

class Cache:
    """Redis-based cache with in-memory fallback"""

    def __init__(self):
        """Initialize Redis connection"""
        self.redis = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )

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
                cached = self.redis.get(key)
                if cached:
                    return json.loads(cached)

                # If not in cache, execute function
                result = await func(*args, **kwargs)

                # Cache the result
                self.redis.setex(
                    key,
                    ttl,
                    json.dumps(result)
                )

                return result
            return wrapper
        return decorator

# Global cache instance
cache = Cache()

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
    return cache(ttl=int(expire.total_seconds()), key_prefix=key_prefix, key_builder=key_builder)
