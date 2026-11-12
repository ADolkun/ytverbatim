import contextlib
import fcntl
import logging
import os
import sys
import time
from collections.abc import Callable, Generator, Sequence
from pathlib import Path

from ytverbatim import paths, proc
from ytverbatim.errors import YtDlpError, stderr_preview
from ytverbatim.ytdlp import jsruntime
from ytverbatim.ytdlp.signals import RETRIES, classify
from ytverbatim.ytdlp.types import YtDlpResult

log = logging.getLogger(__name__)

# yt-dlp's own `-t sleep` preset; batches pass it to keep the request rate polite.
SLEEP_PRESET: tuple[str, ...] = ("-t", "sleep")
BASE_DELAY_S = 5.0


class SubprocessRunner:
    """The one place yt-dlp runs: serialized across processes, classified, and retried only when transient."""

    def __init__(
        self,
        *,
        js_args: Sequence[str] | None = None,
        sleep_args: Sequence[str] = (),
        sleep: Callable[[float], None] = time.sleep,
        base_delay_s: float = BASE_DELAY_S,
    ) -> None:
        self._js_args = tuple(js_args) if js_args is not None else None
        self._sleep_args = tuple(sleep_args)
        self._sleep = sleep
        self._base_delay_s = base_delay_s

    def argv(self, args: Sequence[str], *, cookies: Path | None) -> list[str]:
        if self._js_args is None:
            self._js_args = jsruntime.detect().args
        cookie_args = ("--cookies", str(cookies)) if cookies is not None else ()
        return [
            sys.executable,
            "-m",
            "yt_dlp",
            "--ignore-config",
            "--no-progress",
            *self._js_args,
            *self._sleep_args,
            *cookie_args,
            *args,
        ]

    def run(self, args: Sequence[str], *, cookies: Path | None, timeout_s: float, label: str) -> YtDlpResult:
        argv = self.argv(args, cookies=cookies)
        attempt = 0
        while True:
            # Held per attempt, not across the backoff, so a sleeping retry does not stall other processes.
            with _ytdlp_lock():
                result = proc.run(argv, timeout_s=timeout_s)
            found = classify(result.stderr)
            warnings = tuple(redact(w) for w in found.warnings)
            if result.returncode == 0:
                for warning in warnings:
                    log.warning("%s: %s", label, warning)
                return YtDlpResult(0, result.stdout, result.stderr, found.primary, found.secondary, warnings)
            budget = RETRIES.get(found.primary, 0) if found.primary is not None else 0
            if attempt >= budget:
                preview = stderr_preview(redact(result.stderr))
                raise YtDlpError(found.primary, found.secondary, preview, result.returncode)
            delay = self._base_delay_s * 2**attempt
            attempt += 1
            log.warning("%s: %s, retrying in %gs (%d/%d)", label, found.primary, delay, attempt, budget)
            self._sleep(delay)


def redact(text: str) -> str:
    """Hide the home directory; cookie snapshots and output paths live under it and name the user."""
    home = str(Path.home())
    return text.replace(home, "~") if home not in ("", "/") else text


@contextlib.contextmanager
def _ytdlp_lock() -> Generator[None, None, None]:
    fd = os.open(paths.locks_dir() / "ytdlp.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)
