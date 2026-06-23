"""Bounded exponential-backoff retry for transient failures.

Centralizes the retry loop that several services need (Supabase storage/DB
calls, and the same shape as the LLM/embedding retries). The default predicate
retries only transient network/DNS/socket blips — a momentary hiccup shouldn't
turn into a hard failure, but a real error must surface immediately rather than
being masked behind several seconds of pointless retries.

Synchronous on purpose: callers run it on a worker thread (``asyncio.to_thread``)
or inside the RQ worker, so the ``time.sleep`` backoff never blocks an event loop.
"""

from __future__ import annotations

import logging
import socket
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

DEFAULT_MAX_RETRIES = 4
DEFAULT_BASE_DELAY = 1.0

# httpx / supabase transport errors raised on a network blip, matched by class
# name so this module doesn't need to import httpx.
_RETRYABLE_ERROR_NAMES = {
    "ConnectError", "ConnectTimeout", "ConnectionError", "ReadTimeout",
    "ReadError", "WriteTimeout", "WriteError", "PoolTimeout",
    "RemoteProtocolError", "Timeout",
}
_RETRYABLE_ERROR_PHRASES = (
    "getaddrinfo failed",            # DNS didn't resolve (Errno 11001)
    "non-blocking socket",           # WinError 10035 (WSAEWOULDBLOCK)
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
    "connection refused",
    "connection timed out",
    "timed out",
    "timeout",
    "name or service not known",
    "failed to establish a new connection",
    "server disconnected",
    "max retries exceeded",
)


def is_retryable_network_error(exc: BaseException) -> bool:
    """True for transient network/DNS/socket errors worth retrying.

    Validation and other application errors fall through to a hard failure.
    """
    if isinstance(exc, (socket.gaierror, socket.timeout, ConnectionError, TimeoutError, OSError)):
        return True
    if type(exc).__name__ in _RETRYABLE_ERROR_NAMES:
        return True
    message = str(exc).lower()
    return any(phrase in message for phrase in _RETRYABLE_ERROR_PHRASES)


def with_retry(
    operation: Callable[[], _T],
    description: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    is_retryable: Callable[[BaseException], bool] = is_retryable_network_error,
) -> _T:
    """Run ``operation`` with bounded exponential backoff on retryable errors.

    Only errors for which ``is_retryable`` returns True are retried; anything
    else raises immediately so real failures aren't masked. Delays grow as
    ``base_delay * 2**attempt`` (default 1s, 2s, 4s across four attempts).
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries - 1 or not is_retryable(exc):
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                "%s failed (attempt %s/%s), retrying in %.1fs: %s",
                description, attempt + 1, max_retries, delay, exc,
            )
            time.sleep(delay)
    raise last_error or RuntimeError(f"{description} failed")
