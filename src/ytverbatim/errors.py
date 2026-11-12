from datetime import datetime

from ytverbatim.auth.state import AuthState
from ytverbatim.ytdlp.types import Signal


class YtverbatimError(Exception):
    """Base class for every error raised on purpose by this package."""


class ConfigError(YtverbatimError):
    pass


class DependencyError(YtverbatimError):
    pass


class ProcTimeout(YtverbatimError):  # noqa: N818
    def __init__(self, command: str, timeout_s: float) -> None:
        super().__init__(f"{command} did not finish within {timeout_s:g}s and was killed")
        self.command = command
        self.timeout_s = timeout_s


class YtDlpError(YtverbatimError):
    def __init__(
        self,
        primary: Signal | None,
        secondary: tuple[Signal, ...],
        preview: str,
        returncode: int,
    ) -> None:
        cause = primary.value if primary is not None else "unclassified"
        extra = f" (also: {', '.join(s.value for s in secondary)})" if secondary else ""
        super().__init__(f"yt-dlp failed with exit {returncode}: {cause}{extra}\n{preview}")
        self.primary = primary
        self.secondary = secondary
        self.preview = preview
        self.returncode = returncode


class AuthError(YtverbatimError):
    def __init__(self, state: AuthState, detail: str = "") -> None:
        super().__init__(f"{state.value}: {detail}" if detail else state.value)
        self.state = state
        self.detail = detail


class BatchStopError(YtverbatimError):
    def __init__(self, signal: Signal, until: datetime | None) -> None:
        suffix = f"; wait until {until.isoformat(timespec='seconds')}" if until is not None else ""
        super().__init__(f"batch stopped by {signal.value}{suffix}")
        self.signal = signal
        self.until = until


class UnavailableError(YtverbatimError):
    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"video unavailable ({kind}): {detail}" if detail else f"video unavailable ({kind})")
        self.kind = kind


class MediaValidationError(YtverbatimError):
    pass


class EngineError(YtverbatimError):
    pass


class LLMUnavailableError(YtverbatimError):
    pass


class LLMOutputError(YtverbatimError):
    pass


class BusyError(YtverbatimError):
    pass


def stderr_preview(text: str, *, head: int = 800, tail: int = 800) -> str:
    """Head and tail of long output; yt-dlp prints the cause at the top and the exit detail at the bottom."""
    text = text.strip()
    if len(text) <= head + tail:
        return text
    dropped = len(text) - head - tail
    return f"{text[:head]}\n... [truncated {dropped} chars] ...\n{text[-tail:]}"
