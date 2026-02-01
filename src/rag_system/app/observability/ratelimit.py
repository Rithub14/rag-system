import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from fastapi import HTTPException, status


def _get_redis_client():
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    try:
        import redis

        client = redis.Redis.from_url(url, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


class RateLimiter:
    def __init__(self) -> None:
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._redis = _get_redis_client()

    def check(self, scope: str, key: str, *, limit: int, window_seconds: int) -> None:
        if self._redis is not None:
            self._check_redis(scope, key, limit, window_seconds)
            return
        self._check_memory(scope, key, limit, window_seconds)

    def _check_memory(self, scope: str, key: str, limit: int, window_seconds: int) -> None:
        now = time.time()
        bucket = self._events[f"{scope}:{key}"]
        cutoff = now - window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {scope}. Try again later.",
            )
        bucket.append(now)

    def _check_redis(self, scope: str, key: str, limit: int, window_seconds: int) -> None:
        assert self._redis is not None
        now = int(time.time())
        bucket = now // window_seconds
        redis_key = f"ratelimit:{scope}:{key}:{bucket}"
        count = self._redis.incr(redis_key)
        if count == 1:
            self._redis.expire(redis_key, window_seconds)
        if count > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {scope}. Try again later.",
            )


rate_limiter = RateLimiter()
