"""Simple token-bucket rate limiter for shared LLM clients."""
import threading
import time
from functools import wraps
from typing import Callable

try:
    from groq import RateLimitError
except ImportError:
    RateLimitError = Exception  # type: ignore


class TokenBucket:
    """N tokens replenished over a period. Thread-safe."""

    def __init__(self, capacity: int, period_sec: float):
        self.capacity = capacity
        self.period = period_sec
        self.tokens = float(capacity)
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last
                self.tokens = min(self.capacity, self.tokens + elapsed * (self.capacity / self.period))
                self.last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) * (self.period / self.capacity)
            time.sleep(wait)


# Conservative limit: 25 RPM (5 under Groq free-tier 30 RPM to leave safety margin)
_default_bucket = TokenBucket(capacity=25, period_sec=60.0)


def with_rate_limit_and_retry(fn: Callable):
    """Decorator: enforce the token bucket and retry on 429 with the API's retry-after."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        attempts = 0
        while True:
            _default_bucket.acquire()
            try:
                return fn(*args, **kwargs)
            except RateLimitError as e:
                attempts += 1
                if attempts > 8:
                    raise
                # Respect Retry-After header if present, else exponential backoff
                retry_after = 2.0 * (2 ** (attempts - 1))
                try:
                    hdr = getattr(e, "response", None)
                    if hdr is not None:
                        ra = hdr.headers.get("retry-after")
                        if ra:
                            retry_after = float(ra) + 0.5
                except Exception:
                    pass
                time.sleep(min(retry_after, 30))

    return wrapper
