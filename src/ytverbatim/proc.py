import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ytverbatim import sysmem
from ytverbatim.errors import DependencyError, MemoryTrigger, ProcMemoryExceeded, ProcTimeout

DEFAULT_FLOOR_PERCENT = 10


@dataclass(frozen=True)
class ProcResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float


@dataclass(frozen=True)
class MemoryWatch:
    """Kill the child's process group when its footprint passes `cap_bytes` or available memory drops under the floor.

    The floor covers memory the child does not own: a machine already short of memory must not be pushed into the
    kernel's watchdog by a child that is still under its own cap.
    """

    cap_bytes: int
    floor_percent: int = DEFAULT_FLOOR_PERCENT
    poll_s: float = 1.0
    footprint: Callable[[int], sysmem.Footprint | None] = sysmem.footprint
    available_percent: Callable[[], int] = sysmem.available_percent


def run(
    argv: Sequence[str],
    *,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    stdin: str | None = None,
    memory_watch: MemoryWatch | None = None,
) -> ProcResult:
    command = Path(argv[0]).name
    if memory_watch is not None and stdin is not None:
        # The watch retries communicate(); on Python 3.12 a retry stops writing the input, so the child never sees EOF.
        raise ValueError("a memory watch cannot be combined with stdin")
    started = time.monotonic()
    try:
        # Own session so a timeout can kill grandchildren (node, ffmpeg) along with the child.
        child = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=None if env is None else dict(env),
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        if cwd is not None and not cwd.is_dir():
            raise
        raise DependencyError(f"required executable not found: {command}") from exc
    except PermissionError as exc:
        raise DependencyError(f"required executable is not runnable: {command}") from exc

    try:
        if memory_watch is None:
            stdout, stderr = child.communicate(stdin, timeout=timeout_s)
        else:
            stdout, stderr = _watched(child, command, started + timeout_s, memory_watch)
    except subprocess.TimeoutExpired:
        _kill_group(child)
        raise ProcTimeout(command, timeout_s) from None
    except BaseException:
        _kill_group(child)
        raise
    return ProcResult(child.returncode, stdout, stderr, time.monotonic() - started)


def _watched(child: subprocess.Popen[str], command: str, deadline: float, watch: MemoryWatch) -> tuple[str, str]:
    while True:
        remaining = deadline - time.monotonic()
        try:
            stdout, stderr = child.communicate(timeout=max(0.0, min(watch.poll_s, remaining)))
            return stdout, stderr
        except subprocess.TimeoutExpired:
            if remaining <= watch.poll_s:
                raise
        sample = watch.footprint(child.pid)
        available = watch.available_percent()
        current = None if sample is None else sample.current
        trigger: MemoryTrigger | None = None
        if current is not None and current > watch.cap_bytes:
            trigger = "footprint"
        elif available < watch.floor_percent:
            trigger = "available"
        # A child that exited since the last wait finished its work; collect it instead of reporting a kill.
        if trigger is not None and child.poll() is None:
            raise ProcMemoryExceeded(
                command,
                trigger,
                footprint_bytes=current,
                cap_bytes=watch.cap_bytes,
                available_percent=available,
                floor_percent=watch.floor_percent,
            )


def _kill_group(child: subprocess.Popen[str]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.communicate()
