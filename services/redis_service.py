"""
Redis Service Module

Provides a centralized service for interacting with the Redis instance,
managing the connection pool and offering common Redis operations.
"""

import redis.asyncio as redis
from typing import Optional, Union, Any, Mapping
from datetime import timedelta

from core.config import settings
from utils.logging import AppLogger

logger = AppLogger(__name__)

class RedisService:
    """Service to manage Redis connection and operations."""

    def __init__(self):
        """Initialize the Redis connection pool using settings."""
        self.connection_pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._connect()

    def _connect(self):
        """Establish the connection pool."""
        try:
            logger.info(f"Initializing Redis connection pool for {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            self.connection_pool = redis.ConnectionPool(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB, # Use configured DB
                decode_responses=True # Decode responses to strings by default
            )
            # Create a client instance from the pool for immediate use/testing
            self._client = redis.Redis(connection_pool=self.connection_pool)
            logger.info("Redis connection pool initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection pool: {e}", exc_info=True)
            self.connection_pool = None
            self._client = None
            # Depending on how critical Redis is, you might raise an error here
            # raise ConnectionError(f"Failed to connect to Redis: {e}") from e

    @property
    def client(self) -> Optional[redis.Redis]:
        """Provides access to the Redis client instance."""
        if not self._client:
            logger.warning("Attempted to access Redis client, but connection is not established.")
        return self._client

    async def is_connected(self) -> bool:
        """Check if the Redis connection is alive."""
        if not self.client:
            return False
        try:
            return await self.client.ping()
        except redis.RedisError as e:
            logger.error(f"Redis connection check (ping) failed: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Redis ping: {e}", exc_info=True)
            return False

    async def get(self, key: str) -> Optional[str]:
        """Get a value from Redis by key."""
        if not self.client:
            return None
        try:
            return await self.client.get(key)
        except redis.RedisError as e:
            logger.error(f"Redis GET error for key '{key}': {e}", exc_info=True)
            return None

    async def set(
        self, 
        key: str, 
        value: Any, 
        ex: Optional[Union[int, timedelta]] = None, # Expiry time in seconds or timedelta
        nx: bool = False, # Set only if key does not exist
        xx: bool = False, # Set only if key exists
    ) -> bool:
        """Set a value in Redis."""
        if not self.client:
            return False
        try:
            result = await self.client.set(key, value, ex=ex, nx=nx, xx=xx)
            # SET command returns True on success, None if nx/xx conditions not met
            return result is True
        except redis.RedisError as e:
            logger.error(f"Redis SET error for key '{key}': {e}", exc_info=True)
            return False

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys from Redis. Returns the number of keys deleted."""
        if not self.client or not keys:
            return 0
        try:
            return await self.client.delete(*keys)
        except redis.RedisError as e:
            logger.error(f"Redis DEL error for keys '{keys}': {e}", exc_info=True)
            return 0

    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a key's integer value."""
        if not self.client:
            return None
        try:
            return await self.client.incrby(key, amount)
        except redis.RedisError as e:
            logger.error(f"Redis INCRBY error for key '{key}': {e}", exc_info=True)
            return None

    async def exists(self, *keys: str) -> int:
        """Check if one or more keys exist. Returns the number of existing keys."""
        if not self.client or not keys:
            return 0
        try:
            return await self.client.exists(*keys)
        except redis.RedisError as e:
            logger.error(f"Redis EXISTS error for keys '{keys}': {e}", exc_info=True)
            return 0

    # --- Add other common Redis operations as needed ---
    # async def hget(self, name: str, key: str) -> Optional[str]: ...
    # async def hset(self, name: str, mapping: Mapping) -> int: ...
    # async def publish(self, channel: str, message: Any) -> int: ...

    async def close(self):
        """Close the Redis connection pool."""
        if self._client:
             # Close the individual client connection first if necessary (depends on library version)
             # await self._client.close() 
             pass # Usually closing the pool is sufficient
        if self.connection_pool:
            logger.info("Closing Redis connection pool...")
            await self.connection_pool.disconnect()
            logger.info("Redis connection pool closed.")
            self.connection_pool = None
            self._client = None
