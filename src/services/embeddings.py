"""Embedding generation via LiteLLM (Bedrock Titan by default).

Used by the RAG pipeline to embed image/file chunks at ingest time and to embed
queries at search time. The model, dimensionality, and credentials come from
``.env`` (see ``src/core/config.py``).
"""

from __future__ import annotations

import logging
import time

import litellm

from src.core.config import (
    AWS_REGION,
    EMBEDDING_API_BASE,
    EMBEDDING_API_KEY,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_SEND_DIMENSIONS,
)
from src.services.llm_analysis import is_retryable_llm_error

logger = logging.getLogger(__name__)

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 2.0


def _embedding_kwargs() -> dict:
    kwargs: dict = {
        "model": EMBEDDING_MODEL,
        "api_key": EMBEDDING_API_KEY or None,
        "aws_region_name": AWS_REGION,
    }
    if EMBEDDING_API_BASE:
        kwargs["api_base"] = EMBEDDING_API_BASE
    if EMBEDDING_SEND_DIMENSIONS:
        kwargs["dimensions"] = EMBEDDING_DIM
    return kwargs


def _embed(inputs: list[str]) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = litellm.embedding(**_embedding_kwargs(), input=inputs)
            # litellm normalises to {"data": [{"embedding": [...]}, ...]}
            return [item["embedding"] for item in response["data"]]
        except Exception as exc:
            last_error = exc
            if attempt >= _MAX_RETRIES - 1 or not is_retryable_llm_error(exc):
                raise
            delay = _RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                "Embedding call failed (attempt %s/%s), retrying in %.1fs: %s",
                attempt + 1,
                _MAX_RETRIES,
                delay,
                exc,
            )
            time.sleep(delay)
    raise last_error or RuntimeError("Embedding call failed")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Some providers (Titan) embed one input at a time,
    so we send them individually and collect the vectors in order."""
    vectors: list[list[float]] = []
    for text in texts:
        cleaned = (text or "").strip()
        if not cleaned:
            cleaned = " "
        vectors.extend(_embed([cleaned]))
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
