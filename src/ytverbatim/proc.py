import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ytverbatim import sysmem
from ytverbatim.errors import DependencyError, MemoryTrigger, ProcMemoryExceeded, ProcTimeout

DEFAULT_FLOOR_PERCENT = 10
DEFAULT_TERM_GRACE_S = 10.0


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
    with _spawning(command, cwd):
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
        _, breach = _check(child, command, watch)
        if breach is not None:
            raise breach


class Background:
    """A running child started by `background`; `alive()` turns false once it has exited or been killed."""

    def __init__(self, child: subprocess.Popen[bytes], command: str, watch: MemoryWatch) -> None:
        self.pid = child.pid
        self._child = child
        self._command = command
        self._watch = watch
        # Held around every reap and every read by pid, so a pid freed by a reap is never read again.
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._peak: int | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=f"memory-watch-{child.pid}", daemon=True)
        self._thread.start()

    @property
    def peak_footprint_bytes(self) -> int | None:
        """The highest physical footprint the child reached while it was watched."""
        return self._peak

    @property
    def returncode(self) -> int | None:
        with self._lock:
            return self._child.poll()

    def alive(self) -> bool:
        return self.returncode is None

    def _sample(self) -> ProcMemoryExceeded | None:
        with self._lock:
            if self._child.poll() is not None:
                return None
            sample, breach = _check(self._child, self._command, self._watch)
        if sample is not None:
            self._peak = max(sample.peak, self._peak or 0)
        return breach

    def _run(self) -> None:
        try:
            while True:
                breach = self._sample()
                if breach is not None:
                    self._error = breach
                    _signal_group(self.pid, signal.SIGKILL)
                    return
                if self._child.returncode is not None or self._stop.wait(self._watch.poll_s):
                    return
        except BaseException as exc:
            # A watch that cannot read memory must not leave the child running unwatched.
            self._error = exc
            _signal_group(self.pid, signal.SIGKILL)

    def _close(self, grace_s: float) -> BaseException | None:
        """Stop watching, end the process group, and return what the watch caught, if anything."""
        self._stop.set()
        self._thread.join()
        _signal_group(self.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._child.wait(timeout=grace_s)
        # Also after a clean exit of the leader: a grandchild that ignored SIGTERM would otherwise outlive it.
        _signal_group(self.pid, signal.SIGKILL)
        self._child.wait()
        return self._error


@contextlib.contextmanager
def background(
    argv: Sequence[str],
    *,
    log: Path,
    memory_watch: MemoryWatch,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    term_grace_s: float = DEFAULT_TERM_GRACE_S,
) -> Generator[Background, None, None]:
    """Run a long-lived child (a server) under `memory_watch` for the duration of the block.

    The child gets its own process group, `/dev/null` as stdin, and stdout and stderr appended to `log`. A watcher
    thread applies the same check as `run`'s watch and kills the group the moment it fires. Leaving the block, however
    it is left, sends the group SIGTERM, then SIGKILL after `term_grace_s`. A breach is raised as `ProcMemoryExceeded`
    at exit, chained to the block's own exception if it raised one: the block usually fails only because its server
    was killed under it.
    """
    command = Path(argv[0]).name
    with log.open("ab") as sink, _spawning(command, cwd):
        child = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=subprocess.STDOUT,
            env=None if env is None else dict(env),
            cwd=cwd,
            start_new_session=True,
        )
    try:
        handle = Background(child, command, memory_watch)
    except BaseException:
        _signal_group(child.pid, signal.SIGKILL)
        child.wait()
        raise
    try:
        yield handle
    except BaseException as exc:
        error = handle._close(term_grace_s)
        if error is not None:
            raise error from exc
        raise
    error = handle._close(term_grace_s)
    if error is not None:
        raise error


def _check(
    child: subprocess.Popen[str] | subprocess.Popen[bytes], command: str, watch: MemoryWatch
) -> tuple[sysmem.Footprint | None, ProcMemoryExceeded | None]:
    """Sample the child's footprint and return the breach to raise, if the watch fires on a child still running."""
    sample = watch.footprint(child.pid)
    available = watch.available_percent()
    current = None if sample is None else sample.current
    trigger: MemoryTrigger | None = None
    if current is not None and current > watch.cap_bytes:
        trigger = "footprint"
    elif available < watch.floor_percent:
        trigger = "available"
    # A child that exited since the last wait finished its work; collect it instead of reporting a kill.
    if trigger is None or child.poll() is not None:
        return sample, None
    return sample, ProcMemoryExceeded(
        command,
        trigger,
        footprint_bytes=current,
        cap_bytes=watch.cap_bytes,
        available_percent=available,
        floor_percent=watch.floor_percent,
    )


@contextlib.contextmanager
def _spawning(command: str, cwd: Path | None) -> Generator[None, None, None]:
    try:
        yield
    except FileNotFoundError as exc:
        if cwd is not None and not cwd.is_dir():
            raise
        raise DependencyError(f"required executable not found: {command}") from exc
    except PermissionError as exc:
        raise DependencyError(f"required executable is not runnable: {command}") from exc


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    # EPERM: only zombies are left in the group, which no signal can reach and nothing needs to.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def _kill_group(child: subprocess.Popen[str]) -> None:
    _signal_group(child.pid, signal.SIGKILL)
    child.communicate()
