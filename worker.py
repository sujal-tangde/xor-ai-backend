"""RQ worker entrypoint for background upload processing.

Run alongside the API:

    cd backend
    python worker.py                  # WORKER_COUNT workers (default 1)
    WORKER_COUNT=4 python worker.py   # four workers analyzing in parallel

Each worker is an independent process consuming the same ``xor-processing``
queue; Redis hands each enqueued job to whichever worker is free, so N workers
analyze N files concurrently. Scale by setting ``WORKER_COUNT`` in the env.

Processes image/document analysis, embedding, insight, and knowledge-base jobs
enqueued by the upload API (see src/services/queue.py).
"""

from __future__ import annotations

import logging
import multiprocessing
import signal

from rq import SimpleWorker

from src.core.config import WORKER_COUNT
from src.services.queue import QUEUE_NAME, get_queue, get_redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _run_worker() -> None:
    # SimpleWorker runs jobs in-process (no os.fork), which works on Windows.
    # Each OS process gets its own Redis connection and queue handle.
    worker = SimpleWorker([get_queue()], connection=get_redis())
    worker.work()


def main() -> None:
    count = max(1, WORKER_COUNT)

    # Single worker: run in this process so logs and Ctrl-C are direct.
    if count == 1:
        _run_worker()
        return

    logger.info("Starting %d parallel workers on queue %r", count, QUEUE_NAME)
    procs = [
        multiprocessing.Process(target=_run_worker, name=f"worker-{i}")
        for i in range(count)
    ]
    for p in procs:
        p.start()

    # Forward an explicit shutdown signal (e.g. `docker stop`/systemd SIGTERM)
    # to the children so each RQ worker can finish its current job and exit
    # cleanly. A terminal Ctrl-C already reaches the whole process group.
    def _shutdown(signum, _frame):
        logger.info("Received signal %s, stopping %d workers...", signum, len(procs))
        for p in procs:
            if p.is_alive():
                p.terminate()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
