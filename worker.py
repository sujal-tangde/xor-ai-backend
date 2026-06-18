"""RQ worker entrypoint for background upload processing.

Run alongside the API:

    cd backend
    python worker.py            # or: rq worker -u $REDIS_URL xor-processing

Processes image/document analysis, embedding, insight, and knowledge-base jobs
enqueued by the upload API (see src/services/queue.py).
"""

from __future__ import annotations

import logging

from rq import SimpleWorker

from src.services.queue import QUEUE_NAME, get_queue, get_redis

logging.basicConfig(level=logging.INFO)


def main() -> None:
    # SimpleWorker runs jobs in-process (no os.fork), which works on Windows.
    worker = SimpleWorker([get_queue()], connection=get_redis())
    worker.work()


if __name__ == "__main__":
    main()
