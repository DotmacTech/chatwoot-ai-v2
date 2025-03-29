from functools import wraps
from typing import Optional
import redis
from fastapi import HTTPException
from core.config import settings

class RateLimiter:
    """Rate limiter using Redis"""

    def __init__(self):
        """Initialize Redis connection"""
        self.redis = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )

    def limit(self, rate: str) -> callable:
        """Rate limiting decorator
        
        Args:
            rate (str): Rate limit in format "number/unit" (e.g., "5/minute")
        """
        number, unit = rate.split('/')
        number = int(number)
        seconds = {
            'second': 1,
            'minute': 60,
            'hour': 3600,
            'day': 86400
        }[unit]

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Get client IP from request
                request = next((arg for arg in args if hasattr(arg, 'client')), None)
                if not request:
                    return await func(*args, **kwargs)

                key = f"rate_limit:{request.client.host}:{func.__name__}"
                current = self.redis.get(key)

                if current is None:
                    self.redis.setex(key, seconds, 1)
                elif int(current) >= number:
                    raise HTTPException(
                        status_code=429,
                        detail="Too many requests"
                    )
                else:
                    self.redis.incr(key)

                return await func(*args, **kwargs)
            return wrapper
        return decorator

# Global rate limiter instance
rate_limiter = RateLimiter()

# Expose the rate limit decorator directly
def rate_limit(key_prefix: str, max_requests: str) -> callable:
    """Rate limiting decorator that uses the global rate limiter instance
    
    Args:
        key_prefix (str): Prefix for the rate limit key
        max_requests (str): Rate limit in format "number/unit" (e.g., "5/minute")
    """
    return rate_limiter.limit(max_requests)
