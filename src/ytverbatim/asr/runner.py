"""Parent side of the ASR worker: one worker at a time per machine, in its own process, with a typed result.

Nothing here imports MLX or numpy; the engine libraries only ever load inside the child.
"""

import contextlib
import fcntl
import logging
import os
import resource
import sys
import wave
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from ytverbatim import atomic, paths, proc
from ytverbatim.asr.models import (
    ERROR_FILE,
    HYPOTHESES_FILE,
    METRICS_FILE,
    REQUEST_FILE,
    RunMetrics,
    WorkerError,
    WorkerRequest,
)
from ytverbatim.errors import EngineError, MediaValidationError, stderr_preview

log = logging.getLogger(__name__)

LOCK_NAME = "mlx-worker.lock"
WORKER_MODULE = "ytverbatim.asr.worker"
TIMEOUT_FACTOR = 3.0
TIMEOUT_FLOOR_S = 30 * 60.0


@dataclass(frozen=True)
class WorkerResult:
    hypotheses: Path
    metrics: RunMetrics
    # ru_maxrss of the largest child this process has waited for, in bytes on macOS; an upper bound for this worker.
    children_max_rss_bytes: int
    elapsed_s: float


def worker_argv(request_path: Path, *, python: str | None = None) -> list[str]:
    return [python or sys.executable, "-P", "-m", WORKER_MODULE, str(request_path)]


def timeout_s(audio_seconds: float) -> float:
    return TIMEOUT_FACTOR * audio_seconds + TIMEOUT_FLOOR_S


def run_worker(request: WorkerRequest, *, python: str | None = None) -> WorkerResult:
    """Run the request in a worker subprocess and return where its output is, or raise `EngineError`."""
    out = Path(request.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    request_path = out / REQUEST_FILE
    atomic.write_text(request_path, request.model_dump_json(indent=2) + "\n")
    limit = timeout_s(sum(_wav_seconds(Path(job.wav)) for job in request.files))
    hypothesis = request.engine.spec.id
    with worker_lock():
        result = proc.run(worker_argv(request_path, python=python), timeout_s=limit)
    children_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if result.returncode != 0:
        raise EngineError(_failure(hypothesis, out, result.returncode, result.stderr))
    try:
        metrics = RunMetrics.model_validate_json((out / METRICS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EngineError(f"{hypothesis} worker exited cleanly but left no readable {METRICS_FILE}: {exc}") from exc
    return WorkerResult(out / HYPOTHESES_FILE, metrics, children_rss, result.elapsed_s)


@contextlib.contextmanager
def worker_lock() -> Generator[None, None, None]:
    """Hold the machine-wide MLX lock; two engines at once would fight over unified memory."""
    fd = os.open(paths.locks_dir() / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("waiting for another ASR worker to finish (%s)", LOCK_NAME)
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _failure(hypothesis: str, out: Path, returncode: int, stderr: str) -> str:
    try:
        error = WorkerError.model_validate_json((out / ERROR_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"{hypothesis} worker exited with {returncode} and left no error report\n{stderr_preview(stderr)}"
    return f"{hypothesis} worker failed ({error.type}): {error.message}"


def _wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as reader:
            return reader.getnframes() / reader.getframerate()
    except (wave.Error, EOFError, OSError) as exc:
        raise MediaValidationError(f"not a readable WAV file: {exc}") from exc
