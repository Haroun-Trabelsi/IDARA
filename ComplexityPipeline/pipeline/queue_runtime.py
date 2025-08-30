"""
Tiny in-memory queue runtime used by main.py.
- One asyncio.Queue per user_id
- One asyncio.Task (worker) per user_id that consumes that queue in FIFO order
- Each user is always processed serially; different users process in parallel
"""

from __future__ import annotations
import asyncio
from collections import defaultdict
from typing import Awaitable, Callable, Dict

# user_id -> asyncio.Queue[job_id]
_user_queues: Dict[str, asyncio.Queue[str]] = defaultdict(asyncio.Queue)

# user_id -> asyncio.Task (long-lived per-user worker)
_user_workers: Dict[str, asyncio.Task] = {}

async def enqueue_job(user_id: str, job_id: str) -> None:
    """Push a job id into this user's queue (FIFO)."""
    await _user_queues[user_id].put(job_id)

async def ensure_worker(
    user_id: str,
    handler: Callable[[str, str], Awaitable[None]],
) -> None:
    """Start a worker for this user if there isn't one or it finished.

    handler(user_id, job_id) is a coroutine that does the actual processing
    of the given job_id.
    """
    # Recreate a worker if it doesn't exist or it crashed/finished
    if user_id not in _user_workers or _user_workers[user_id].done():
        _user_workers[user_id] = asyncio.create_task(_worker_loop(user_id, handler))

async def _worker_loop(
    user_id: str,
    handler: Callable[[str, str], Awaitable[None]],
) -> None:
    """Consume this user's queue forever. One job at a time, FIFO."""
    q = _user_queues[user_id]
    while True:
        job_id = await q.get()  # wait for the next job
        try:
            await handler(user_id, job_id)
        finally:
            # Mark the item as processed so Queue.join() (if used) can proceed
            q.task_done()