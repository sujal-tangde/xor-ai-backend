"""RQ (Redis Queue) wiring for background processing jobs.

The upload API enqueues image/document processing here and returns immediately;
a separate ``rq worker`` process (see ``backend/worker.py``) runs the jobs.
Job functions are referenced by dotted path so the worker can import them
without the FastAPI app being loaded.
"""

from __future__ import annotations

import logging

from redis import Redis
from rq import Queue, Retry

from src.core.config import REDIS_URL

logger = logging.getLogger(__name__)

QUEUE_NAME = "xor-processing"
# Generous timeout: a document map/reduce can involve several LLM calls.
JOB_TIMEOUT = 1800
# Automatically retry a failed job with growing back-off instead of letting it
# die in the failed registry. A KB recompute that hit a transient statement /
# lock timeout (the cause of the historical processed-count drift) gets three
# more chances before it's considered dead. Back-off is in seconds.
DEFAULT_RETRY_INTERVALS = [30, 90, 300]

_redis: Redis | None = None
_queue: Queue | None = None


def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(REDIS_URL)
    return _redis


def get_queue() -> Queue:
    global _queue
    if _queue is None:
        _queue = Queue(QUEUE_NAME, connection=get_redis())
    return _queue


def _enqueue(func_path: str, *args, retry: bool = True) -> None:
    try:
        retry_policy = (
            Retry(max=len(DEFAULT_RETRY_INTERVALS), interval=DEFAULT_RETRY_INTERVALS)
            if retry
            else None
        )
        get_queue().enqueue(func_path, *args, job_timeout=JOB_TIMEOUT, retry=retry_policy)
    except Exception:
        # If Redis is down we don't want the upload request to fail; the file is
        # already stored and can be reprocessed. Log and move on. The reconcile
        # sweep (knowledge_base.reconcile_unprocessed_insights) is the backstop
        # for an insight whose recompute was never enqueued because of this.
        logger.exception("Failed to enqueue %s", func_path)


def enqueue_image_processing(
    file_id: str, project_id: str, user_id: str, compressed_jpeg: bytes, file_name: str
) -> None:
    _enqueue(
        "src.services.file_processing.process_image",
        file_id,
        project_id,
        user_id,
        compressed_jpeg,
        file_name,
    )


def enqueue_document_processing(
    file_id: str, project_id: str, user_id: str, raw_bytes: bytes, ext: str, file_name: str
) -> None:
    _enqueue(
        "src.services.file_processing.process_document",
        file_id,
        project_id,
        user_id,
        raw_bytes,
        ext,
        file_name,
    )


def enqueue_knowledge_base_recompute(project_id: str, insight_id: str) -> None:
    _enqueue(
        "src.services.knowledge_base.recompute_knowledge_base",
        project_id,
        insight_id,
    )


def enqueue_qa_insight(
    project_id: str, user_id: str | None, qa_pairs: list
) -> None:
    _enqueue(
        "src.services.report_qa_ingest.ingest_qa_insight",
        project_id,
        user_id,
        qa_pairs,
    )
