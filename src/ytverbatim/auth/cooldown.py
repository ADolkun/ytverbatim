"""The persisted cooldown after a bot or rate signal, and the runner guard that starts it.

Every yt-dlp call that can reach YouTube (fetch stages and the ``:ytfav`` probe alike) runs through
`GuardedRunner`, so no caller can turn a stop signal into a retry or a cookie escalation, and every stop
writes or extends the one cooldown file that later runs and probes respect.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from ytverbatim import atomic, paths
from ytverbatim.auth.fixes import fix_for
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import BatchStopError, YtDlpError, YtverbatimError
from ytverbatim.ytdlp.types import Signal, YtDlpResult, YtDlpRunner

STOP_STATES: dict[Signal, AuthState] = {
    Signal.BOT_CHECK: AuthState.BOT_CHECK,
    Signal.IP_BLOCKED: AuthState.IP_BLOCKED,
    Signal.CAPTCHA: AuthState.CAPTCHA,
    Signal.RATE_LIMITED: AuthState.RATE_LIMITED,
    Signal.HTTP_429: AuthState.RATE_LIMITED,
}
# A bare 429 on a subtitle download only costs that video its captions (fetch_captions handles it).
CAPTIONS_LABEL = "captions"

COOLDOWN_BASE = timedelta(hours=1)
COOLDOWN_MAX = timedelta(hours=24)
COOLDOWN_REPEAT_WINDOW = timedelta(hours=24)


@dataclass(frozen=True)
class Cooldown:
    until: datetime
    signal: Signal
    hours: float

    def active(self, now: datetime) -> bool:
        return self.until > now


def read_cooldown() -> Cooldown | None:
    """Return the persisted cooldown, active or not; raise ValueError unless the file is what `start_cooldown` wrote."""
    try:
        raw: object = json.loads(paths.cooldown_file().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(raw, dict):
        raise ValueError("cooldown.json is not an object")
    fields = cast("dict[str, object]", raw)
    until_raw, signal_raw, hours = fields.get("until"), fields.get("signal"), fields.get("hours")
    if not isinstance(until_raw, str) or not isinstance(signal_raw, str):
        raise ValueError("cooldown.json needs 'until' and 'signal' strings")
    if not isinstance(hours, int | float) or isinstance(hours, bool) or hours <= 0:
        raise ValueError("cooldown.json needs a positive 'hours' number")
    until = datetime.fromisoformat(until_raw)
    if until.tzinfo is None:
        raise ValueError("cooldown.json 'until' has no time zone")
    signal = Signal(signal_raw)
    if signal not in STOP_STATES:
        raise ValueError(f"cooldown.json names {signal_raw!r}, which is not a stop signal")
    return Cooldown(until, signal, float(hours))


def active_cooldown(now: datetime) -> Cooldown | None:
    """Return the cooldown if it is still running; an unreadable cooldown file refuses (fail loud)."""
    try:
        cooldown = read_cooldown()
    except (OSError, ValueError) as exc:
        raise YtverbatimError(
            f"cooldown.json is unreadable ({exc}); inspect it with: ytverbatim auth doctor, or pass --ignore-cooldown"
        ) from exc
    return cooldown if cooldown is not None and cooldown.active(now) else None


def check_cooldown(now: datetime) -> None:
    """Refuse to start while a cooldown is active."""
    if (cooldown := active_cooldown(now)) is not None:
        raise BatchStopError(cooldown.signal, cooldown.until)


def start_cooldown(signal: Signal, now: datetime) -> Cooldown:
    """Persist a cooldown of 1 h, doubled (up to 24 h) when the previous one ended less than 24 h ago."""
    try:
        previous = read_cooldown()
    except (OSError, ValueError):
        previous = None
    hours = COOLDOWN_BASE / timedelta(hours=1)
    until = now + COOLDOWN_BASE
    if previous is not None and now < previous.until + COOLDOWN_REPEAT_WINDOW:
        hours = min(previous.hours * 2, COOLDOWN_MAX / timedelta(hours=1))
        until = max(now + timedelta(hours=hours), previous.until)
    cooldown = Cooldown(until, signal, hours)
    atomic.write_json(
        paths.cooldown_file(),
        {"until": until.isoformat(timespec="seconds"), "signal": signal.value, "hours": hours},
        mode=0o600,
    )
    return cooldown


def shown_time(until: datetime) -> str:
    return until.astimezone().strftime("%Y-%m-%d %H:%M")


def stop_fix(signal: Signal, until: datetime | None) -> str:
    return fix_for(STOP_STATES[signal], until=shown_time(until) if until is not None else None)


def stop_signal(label: str, primary: Signal | None, secondary: Sequence[Signal]) -> Signal | None:
    """Pick the stop signal among a call's signals, primary first; a 429 on the subtitle call is not one."""
    for signal in (primary, *secondary):
        if signal in STOP_STATES and not (signal is Signal.HTTP_429 and label == CAPTIONS_LABEL):
            return signal
    return None


@dataclass(frozen=True)
class Seen:
    label: str
    ok: bool
    primary: Signal | None
    secondary: tuple[Signal, ...]
    warnings: tuple[str, ...]


class GuardedRunner:
    """Wraps a runner so that no caller can ignore a stop signal or leak `-v` stderr.

    A failed call with a stop signal becomes a cooldown plus `BatchStopError` right here, before any caller's
    except clause could turn it into an escalation. A successful call with a stop signal among its secondary
    signals starts the cooldown too and leaves it in `stop`: the current video may finish, nothing after it may
    start. Only classified signals and warnings are kept, never stderr.
    """

    def __init__(self, inner: YtDlpRunner, clock: Callable[[], datetime]) -> None:
        self._inner = inner
        self._clock = clock
        self.seen: list[Seen] = []
        self.stop: Cooldown | None = None

    def run(self, args: Sequence[str], *, cookies: Path | None, timeout_s: float, label: str) -> YtDlpResult:
        try:
            result = self._inner.run(args, cookies=cookies, timeout_s=timeout_s, label=label)
        except YtDlpError as exc:
            self.seen.append(Seen(label, False, exc.primary, exc.secondary, ()))
            if (signal := stop_signal(label, exc.primary, exc.secondary)) is not None:
                cooldown = self._start(signal)
                raise BatchStopError(cooldown.signal, cooldown.until) from None
            if "-v" in args:
                # `from None`: the original's message holds the debug lines, which echo argv.
                raise _without_debug(exc) from None
            raise
        self.seen.append(Seen(label, True, result.primary, result.secondary, result.warnings))
        if (signal := stop_signal(label, result.primary, result.secondary)) is not None:
            self._start(signal)
        return result

    def last(self, label: str) -> Seen | None:
        return next((seen for seen in reversed(self.seen) if seen.label == label), None)

    def _start(self, signal: Signal) -> Cooldown:
        # One cooldown per stop: a second signal before the batch ends must not double it again.
        if self.stop is None:
            self.stop = start_cooldown(signal, self._clock())
        return self.stop


def guarded(runner: YtDlpRunner, clock: Callable[[], datetime]) -> GuardedRunner:
    """Wrap ``runner`` unless it is already guarded, so a shared guard keeps its one `stop`."""
    return runner if isinstance(runner, GuardedRunner) else GuardedRunner(runner, clock)


def utcnow() -> datetime:
    return datetime.now(UTC)


def _without_debug(exc: YtDlpError) -> YtDlpError:
    # Only ERROR and WARNING lines survive: the preview is cut mid-line, so a debug fragment need not start
    # with "[debug]".
    kept = "\n".join(line for line in exc.preview.splitlines() if line.startswith(("ERROR:", "WARNING:")))
    return YtDlpError(exc.primary, exc.secondary, kept, exc.returncode)
