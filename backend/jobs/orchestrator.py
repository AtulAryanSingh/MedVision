import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from api import get_cache_dir

TERMINAL_STATES = {"succeeded", "failed", "canceled"}

_TASKS: dict[str, asyncio.Task] = {}
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jobs_dir() -> str:
    path = os.path.join(get_cache_dir(), "jobs")
    os.makedirs(path, exist_ok=True)
    return path


def _job_path(job_id: str) -> str:
    # Validate and canonicalize UUID so user-provided IDs cannot alter paths.
    safe_job_id = str(uuid.UUID(job_id))
    return os.path.join(_jobs_dir(), f"{safe_job_id}.json")


def _read_job(job_id: str) -> dict[str, Any]:
    try:
        path = _job_path(job_id)
    except (ValueError, AttributeError) as exc:
        raise FileNotFoundError(job_id) from exc
    if not os.path.isfile(path):
        raise FileNotFoundError(job_id)
    with open(path) as fh:
        return json.load(fh)


def _write_job(job: dict[str, Any]) -> None:
    job["updated_at"] = _now_iso()
    with open(_job_path(job["job_id"]), "w") as fh:
        json.dump(job, fh)


def _mark(job_id: str, **fields: Any) -> dict[str, Any]:
    job = _read_job(job_id)
    job.update(fields)
    _write_job(job)
    return job


def create_job(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    now = _now_iso()
    job = {
        "job_id": job_id,
        "operation": operation,
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "cancel_requested": False,
        "payload": payload,
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
    }
    _write_job(job)
    return job


def get_job(job_id: str) -> dict[str, Any]:
    return _read_job(job_id)


def get_result(job_id: str) -> dict[str, Any]:
    job = _read_job(job_id)
    if job["status"] != "succeeded":
        raise ValueError(f"Job {job_id} is not succeeded (status={job['status']}).")
    return job.get("result") or {}


def request_cancel(job_id: str) -> dict[str, Any]:
    job = _read_job(job_id)
    if job["status"] in TERMINAL_STATES:
        return job

    job["cancel_requested"] = True
    job["message"] = "Cancel requested"

    # If still queued, transition immediately.
    if job["status"] == "queued":
        job["status"] = "canceled"
        job["progress"] = 0
        job["finished_at"] = _now_iso()
        job["message"] = "Canceled before execution"

    _write_job(job)

    task = _TASKS.get(job_id)
    if task and not task.done():
        task.cancel()

    return _read_job(job_id)


def start_job(
    job_id: str,
    worker: Callable[
        [Callable[[int, str], None], Callable[[], bool]],
        Awaitable[dict[str, Any]],
    ],
) -> None:
    async def _runner() -> None:
        try:
            job = _read_job(job_id)
        except FileNotFoundError:
            return

        if job["status"] in TERMINAL_STATES:
            return

        _mark(job_id, status="running", progress=1, message="Running", started_at=_now_iso())

        def update_progress(progress: int, message: str = "Running") -> None:
            p = max(0, min(100, int(progress)))
            _mark(job_id, progress=p, message=message)

        def is_canceled() -> bool:
            j = _read_job(job_id)
            return bool(j.get("cancel_requested")) or j.get("status") == "canceled"

        try:
            result = await worker(update_progress, is_canceled)
            latest = _read_job(job_id)
            if latest.get("cancel_requested"):
                _mark(
                    job_id,
                    status="canceled",
                    message="Canceled",
                    finished_at=_now_iso(),
                )
                return
            _mark(
                job_id,
                status="succeeded",
                progress=100,
                message="Succeeded",
                result=result,
                finished_at=_now_iso(),
            )
        except asyncio.CancelledError:
            _mark(
                job_id,
                status="canceled",
                message="Canceled",
                finished_at=_now_iso(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Async job %s failed", job_id)
            _mark(
                job_id,
                status="failed",
                message="Failed",
                error={"type": type(exc).__name__, "detail": str(exc)},
                finished_at=_now_iso(),
            )
        finally:
            _TASKS.pop(job_id, None)

    task = asyncio.create_task(_runner())
    _TASKS[job_id] = task
