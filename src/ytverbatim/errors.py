from datetime import datetime
from typing import TYPE_CHECKING, Literal

from ytverbatim.auth.state import AuthState
from ytverbatim.ytdlp.types import Signal

if TYPE_CHECKING:
    from ytverbatim.llm.protocol import Usage


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


MemoryTrigger = Literal["footprint", "available"]
_GIB = 1 << 30


class ProcMemoryExceeded(YtverbatimError):  # noqa: N818
    """A child killed by the memory watch: its own footprint passed the cap, or the machine ran short of memory."""

    def __init__(
        self,
        command: str,
        trigger: MemoryTrigger,
        *,
        footprint_bytes: int | None,
        cap_bytes: int,
        available_percent: int,
        floor_percent: int,
    ) -> None:
        used = "unknown" if footprint_bytes is None else f"{footprint_bytes / _GIB:.2f} GiB"
        if trigger == "footprint":
            detail = (
                f"its physical footprint reached {used}, over the {cap_bytes / _GIB:.2f} GiB cap "
                f"(system memory available: {available_percent}%, floor {floor_percent}%)"
            )
        else:
            detail = (
                f"system memory available fell to {available_percent}%, under the {floor_percent}% floor "
                f"(its physical footprint: {used}, cap {cap_bytes / _GIB:.2f} GiB)"
            )
        super().__init__(f"{command} was killed: {detail}")
        self.command = command
        self.trigger: MemoryTrigger = trigger
        self.footprint_bytes = footprint_bytes
        self.cap_bytes = cap_bytes
        self.available_percent = available_percent
        self.floor_percent = floor_percent


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


class EngineMemoryError(EngineError):
    """The memory watch killed the ASR worker; the batch stops, since the next video would hit the same limit."""

    def __init__(self, detail: str, fix: str, cause: ProcMemoryExceeded) -> None:
        super().__init__(f"{detail}; {fix}")
        self.detail = detail
        self.fix = fix
        self.trigger: MemoryTrigger = cause.trigger


class LLMError(YtverbatimError):
    """A failed LLM call.

    `responded` is True when a reply arrived, and was billed, but could not be used, for example when it was cut off
    at the token limit; `usage` then holds its tokens, or None when the transport did not report them.
    """

    def __init__(self, message: str, *, responded: bool = False, usage: "Usage | None" = None) -> None:
        super().__init__(message)
        self.responded = responded
        self.usage = usage


class LLMUnavailableError(LLMError):
    pass


class LLMOutputError(LLMError):
    pass


class BusyError(YtverbatimError):
    pass


class ContractError(YtverbatimError):
    """A `context.json` that cannot be read: bad JSON, an unsupported version, or a contract violation."""


def stderr_preview(text: str, *, head: int = 800, tail: int = 800) -> str:
    """Head and tail of long output; yt-dlp prints the cause at the top and the exit detail at the bottom."""
    text = text.strip()
    if len(text) <= head + tail:
        return text
    dropped = len(text) - head - tail
    return f"{text[:head]}\n... [truncated {dropped} chars] ...\n{text[-tail:]}"
