"""Parent side of the ASR worker: one worker at a time per machine, in its own process, with a typed result.

Nothing here imports MLX or numpy; the engine libraries only ever load inside the child.
"""

import contextlib
import fcntl
import logging
import math
import os
import resource
import sys
import wave
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from ytverbatim import atomic, paths, proc, sysmem
from ytverbatim.asr.models import (
    ERROR_FILE,
    HYPOTHESES_FILE,
    METRICS_FILE,
    REQUEST_FILE,
    RunMetrics,
    WorkerError,
    WorkerRequest,
)
from ytverbatim.errors import (
    ConfigError,
    EngineError,
    EngineMemoryError,
    MediaValidationError,
    ProcMemoryExceeded,
    stderr_preview,
)

log = logging.getLogger(__name__)

LOCK_NAME = "mlx-worker.lock"
WORKER_MODULE = "ytverbatim.asr.worker"
TIMEOUT_FACTOR = 3.0
TIMEOUT_FLOOR_S = 30 * 60.0
DEFAULT_MEMORY_CAP_FRACTION = 0.5
# With the 10% available-memory floor, a cap above 90% of RAM can never fire before the floor does, so it would be a
# cap in name only.
MAX_MEMORY_CAP_FRACTION = 0.9
CAP_SETTING = "memory_cap_gib"
_GIB = 1 << 30


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


def memory_cap_bytes(gib: float | None) -> int | None:
    """Convert the `asr.memory_cap_gib` setting; None keeps the default cap."""
    if gib is None:
        return None
    if not math.isfinite(gib) or gib <= 0:
        raise ConfigError(f"asr.{CAP_SETTING} must be a positive number of GiB, got {gib}")
    cap = max(1, round(gib * _GIB))
    ceiling = sysmem.physical_memory_bytes() * MAX_MEMORY_CAP_FRACTION
    if cap > ceiling:
        raise ConfigError(
            f"asr.{CAP_SETTING} = {gib:g} is above 90% of this Mac's memory ({ceiling / _GIB:.1f} GiB), where the "
            "watch would only fire once the whole machine is short of memory; set it lower"
        )
    return cap


def default_memory_cap_bytes() -> int:
    return int(sysmem.physical_memory_bytes() * DEFAULT_MEMORY_CAP_FRACTION)


def run_worker(request: WorkerRequest, *, python: str | None = None) -> WorkerResult:
    """Run the request in a worker subprocess and return where its output is, or raise `EngineError`.

    The worker is killed if its footprint passes `request.memory_cap_bytes` (default: half of physical memory) or the
    machine runs short of memory; the chunks it finished stay on disk for the next run to resume from.
    """
    cap = request.memory_cap_bytes or default_memory_cap_bytes()
    request = request.model_copy(update={"memory_cap_bytes": cap})
    out = Path(request.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    request_path = out / REQUEST_FILE
    limit = timeout_s(sum(_wav_seconds(Path(job.wav)) for job in request.files))
    hypothesis = request.engine.spec.id
    with worker_lock():
        # Under the lock, so a run waiting its turn cannot overwrite the request a running worker was given, and a
        # report left by an earlier run cannot be read as this worker's failure.
        (out / ERROR_FILE).unlink(missing_ok=True)
        atomic.write_text(request_path, request.model_dump_json(indent=2) + "\n")
        try:
            result = proc.run(
                worker_argv(request_path, python=python), timeout_s=limit, memory_watch=proc.MemoryWatch(cap)
            )
        except ProcMemoryExceeded as exc:
            detail = f"{hypothesis} worker stopped to protect the machine: {exc}"
            raise EngineMemoryError(detail, memory_fix(exc), exc) from exc
    children_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if result.returncode != 0:
        raise EngineError(_failure(hypothesis, out, result.returncode, result.stderr))
    try:
        metrics = RunMetrics.model_validate_json((out / METRICS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EngineError(f"{hypothesis} worker exited cleanly but left no readable {METRICS_FILE}: {exc}") from exc
    return WorkerResult(out / HYPOTHESES_FILE, metrics, children_rss, result.elapsed_s)


@contextlib.contextmanager
def worker_lock() -> Generator[int, None, None]:
    """Hold the machine-wide MLX lock; two engines at once would fight over unified memory.

    Yields the locked descriptor. A child given a copy (`pass_fds`) keeps the lock held until it exits, even if this
    process dies first, because a flock belongs to the open file and not to the process that took it.
    """
    fd = os.open(paths.locks_dir() / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("waiting for another ASR worker to finish (%s)", LOCK_NAME)
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        os.close(fd)


def memory_fix(exc: ProcMemoryExceeded, *, smaller: str = "use a smaller engine", kept: str = "finished chunks") -> str:
    """Say how to avoid another kill by the memory watch; `smaller` names the lighter choice, `kept` what survives."""
    resume = f"{kept} are kept and the next run resumes from them"
    if exc.trigger == "available":
        return f"close memory-heavy apps, then run again ({resume})"
    return (
        f"{smaller}, or raise asr.{CAP_SETTING} in the config file only if this Mac has that much memory to spare "
        f"({resume})"
    )


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
