"""Simple retry-with-backoff decorator used by external API wrappers."""

import functools
import time
from typing import Callable, TypeVar

from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def retry(max_attempts: int = 3, base_delay: float = 0.5, backoff: float = 2.0) -> Callable:
    """Retry a function on exception with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts before giving up.
        base_delay: Delay in seconds before the first retry.
        backoff: Multiplier applied to the delay after each failed attempt.

    Returns:
        A decorator that wraps the target function with retry logic.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = base_delay
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - intentionally broad for external calls
                    last_exc = exc
                    logger.warning(
                        "Attempt %s/%s failed for %s: %s", attempt, max_attempts, func.__name__, exc
                    )
                    if attempt < max_attempts:
                        time.sleep(delay)
                        delay *= backoff
            logger.error("All %s attempts failed for %s", max_attempts, func.__name__)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
