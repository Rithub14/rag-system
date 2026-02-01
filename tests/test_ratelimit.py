import pytest
from fastapi import HTTPException

from rag_system.app.observability.ratelimit import RateLimiter


def test_rate_limiter_blocks_after_limit():
    limiter = RateLimiter()
    for _ in range(2):
        limiter.check("query", "user1", limit=2, window_seconds=60)
    with pytest.raises(HTTPException) as exc:
        limiter.check("query", "user1", limit=2, window_seconds=60)
    assert exc.value.status_code == 429
