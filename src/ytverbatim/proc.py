import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ytverbatim.errors import DependencyError, ProcTimeout


@dataclass(frozen=True)
class ProcResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float


def run(
    argv: Sequence[str],
    *,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    stdin: str | None = None,
) -> ProcResult:
    command = Path(argv[0]).name
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
        stdout, stderr = child.communicate(stdin, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(child)
        raise ProcTimeout(command, timeout_s) from None
    except BaseException:
        _kill_group(child)
        raise
    return ProcResult(child.returncode, stdout, stderr, time.monotonic() - started)


def _kill_group(child: subprocess.Popen[str]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.communicate()
