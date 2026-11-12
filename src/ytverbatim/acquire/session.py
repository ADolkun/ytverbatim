"""Fetch session: anonymous-first extraction, cookie escalation, rotation recovery, and batch stops.

Account footprint is the design constraint. Each video gets one `-J`, anonymous unless the target is known
to need a login; cookies are used only after YouTube asks for them, and a bot or rate signal stops the
whole batch and starts a cooldown instead of being retried, least of all with cookies.
"""

import contextlib
import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Literal, cast

from ytverbatim import atomic, ids, paths
from ytverbatim.acquire.audio import INFO_MAX_AGE_S, WAV_NAME, fetch_audio, find_audio, to_wav16k, validate_audio
from ytverbatim.acquire.captions import DEFAULT_LANGUAGES, CaptionsResult, fetch_captions
from ytverbatim.acquire.captions import FILE_STEM as CAPTIONS_STEM
from ytverbatim.acquire.expand import expand_target
from ytverbatim.acquire.metadata import VideoMetadata, fetch_info, info_path, load_metadata
from ytverbatim.auth import preflight, snapshot
from ytverbatim.auth.fixes import fix_for
from ytverbatim.auth.snapshot import Snapshot
from ytverbatim.auth.source import BrowserSource, CookieFileSource, CookieSource
from ytverbatim.auth.state import AuthState
from ytverbatim.contract.models import Access
from ytverbatim.errors import (
    AuthError,
    BatchStopError,
    DependencyError,
    MediaValidationError,
    ProcTimeout,
    UnavailableError,
    YtDlpError,
    YtverbatimError,
)
from ytverbatim.ytdlp import version
from ytverbatim.ytdlp.runner import SLEEP_PRESET, SubprocessRunner, redact
from ytverbatim.ytdlp.signals import js_challenge_failed
from ytverbatim.ytdlp.types import Signal, YtDlpResult, YtDlpRunner

log = logging.getLogger(__name__)

type Policy = Literal["auto", "always", "never"]
type BrowserKind = Literal["chrome", "brave", "chromium", "edge", "vivaldi"]
type AuthSourceKind = Literal["none", "cookie-file", "chrome", "brave", "chromium", "edge", "vivaldi"]

_BROWSER_KINDS: dict[str, BrowserKind] = {
    "chrome": "chrome",
    "brave": "brave",
    "chromium": "chromium",
    "edge": "edge",
    "vivaldi": "vivaldi",
}

STOP_STATES: dict[Signal, AuthState] = {
    Signal.BOT_CHECK: AuthState.BOT_CHECK,
    Signal.IP_BLOCKED: AuthState.IP_BLOCKED,
    Signal.CAPTCHA: AuthState.CAPTCHA,
    Signal.RATE_LIMITED: AuthState.RATE_LIMITED,
    Signal.HTTP_429: AuthState.RATE_LIMITED,
}
ESCALATE_SIGNALS = frozenset({Signal.LOGIN_REQUIRED, Signal.MEMBERS_ONLY, Signal.AGE_GATE, Signal.PRIVATE})
# A bare 429 on a subtitle download only costs that video its captions (fetch_captions handles it).
_CAPTIONS_LABEL = "captions"

# Chromium commits its cookie store about every 30 s, so a re-snapshot sooner reads the same stale values.
ROTATION_WAIT_S = 30.0
COOLDOWN_BASE = timedelta(hours=1)
COOLDOWN_MAX = timedelta(hours=24)
COOLDOWN_REPEAT_WINDOW = timedelta(hours=24)

SINGLE_VIDEO_SLEEP_ARGS: tuple[str, ...] = ("--sleep-requests", "0.75", "--sleep-subtitles", "5")
MEMBERS_PLAYLIST_PREFIX = "UUMO"

ACCESS_FILE = "access.json"
METADATA_FILE = "metadata.json"
CAPTIONS_DIR = "captions"
TRACKS_FILE = "tracks.json"
REPORT_FILE = "batch_report.json"
_REPORT_VERSION = 1

# Errors about the auth source itself: once one happens, every later cookied request in the batch would
# fail the same way, so they are replayed without touching YouTube again.
_PER_VIDEO_AUTH_STATES = frozenset({AuthState.NOT_MEMBER, AuthState.AUTH_REQUIRED})

_PERMANENT_FIXES: dict[Signal, str] = {
    Signal.UNAVAILABLE: "The video is unavailable or was removed; there is nothing to retry.",
    Signal.GEO: "The uploader blocks this video in your country.",
    Signal.DRM: "DRM-protected videos cannot be fetched.",
    Signal.PRIVATE: "Private video: only accounts the uploader invited can open it.",
    Signal.AGE_GATE: "Age-restricted: the auth profile needs a signed-in, age-verified YouTube account.",
    Signal.PREMIUM_ONLY: "YouTube Premium only: the auth profile's account needs Premium.",
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class FetchOptions:
    policy: Policy = "auto"
    captions: bool = True
    audio: bool = True
    languages: tuple[str, ...] = DEFAULT_LANGUAGES
    # Channel ids (UC…), handles (@name) or channel URLs whose videos use cookies from the first request.
    cookie_channels: tuple[str, ...] = ()
    max_videos: int = 50
    ignore_cooldown: bool = False


@dataclass(frozen=True)
class Cooldown:
    until: datetime
    signal: str
    hours: float


def read_cooldown() -> Cooldown | None:
    """Return the persisted cooldown, active or not; raise ValueError when the file exists but is unreadable."""
    path = paths.cooldown_file()
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(raw, dict):
        raise ValueError("cooldown.json is not an object")
    fields = cast("dict[str, object]", raw)
    until_raw, signal, hours = fields.get("until"), fields.get("signal"), fields.get("hours", 1)
    if not isinstance(until_raw, str) or not isinstance(signal, str):
        raise ValueError("cooldown.json needs 'until' and 'signal' strings")
    until = datetime.fromisoformat(until_raw)
    if until.tzinfo is None:
        until = until.replace(tzinfo=UTC)
    duration = float(hours) if isinstance(hours, int | float) and not isinstance(hours, bool) and hours > 0 else 1.0
    return Cooldown(until, signal, duration)


def check_cooldown(now: datetime) -> None:
    """Refuse to start while a cooldown is active; an unreadable cooldown file also refuses (fail loud)."""
    try:
        cooldown = read_cooldown()
    except (OSError, ValueError) as exc:
        raise YtverbatimError(
            f"cooldown.json is unreadable ({exc}); inspect it with: ytverbatim auth doctor, or pass --ignore-cooldown"
        ) from exc
    if cooldown is not None and cooldown.until > now:
        raise BatchStopError(_signal(cooldown.signal), cooldown.until)


def start_cooldown(signal: Signal, now: datetime) -> datetime:
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
    # `hours` rides along so the next repeat can double it; the doctor reads only `until` and `signal`.
    atomic.write_json(
        paths.cooldown_file(),
        {"until": until.isoformat(timespec="seconds"), "signal": signal.value, "hours": hours},
        mode=0o600,
    )
    return until


def _signal(value: str) -> Signal:
    try:
        return Signal(value)
    except ValueError:
        return Signal.BOT_CHECK


def stop_fix(signal: Signal, until: datetime | None) -> str:
    shown = until.astimezone().strftime("%Y-%m-%d %H:%M") if until is not None else None
    return fix_for(STOP_STATES.get(signal, AuthState.BOT_CHECK), until=shown)


@dataclass(frozen=True)
class _Seen:
    label: str
    ok: bool
    primary: Signal | None
    secondary: tuple[Signal, ...]
    warnings: tuple[str, ...]


class _GuardedRunner:
    """Wraps the runner so that no caller can ignore a stop signal or leak `-v` stderr.

    A stop signal becomes a cooldown plus `BatchStopError` right here, before any caller's except clause
    could turn it into an escalation. Only classified signals and warnings are kept, never stderr.
    """

    def __init__(self, inner: YtDlpRunner, clock: Callable[[], datetime]) -> None:
        self._inner = inner
        self._clock = clock
        self.seen: list[_Seen] = []

    def run(self, args: Sequence[str], *, cookies: Path | None, timeout_s: float, label: str) -> YtDlpResult:
        try:
            result = self._inner.run(args, cookies=cookies, timeout_s=timeout_s, label=label)
        except YtDlpError as exc:
            self.seen.append(_Seen(label, False, exc.primary, exc.secondary, ()))
            stops = exc.primary in STOP_STATES and not (exc.primary is Signal.HTTP_429 and label == _CAPTIONS_LABEL)
            if exc.primary is not None and stops:
                until = start_cooldown(exc.primary, self._clock())
                raise BatchStopError(exc.primary, until) from None
            if "-v" in args:
                # `from None`: the original's message holds the debug lines, which echo argv.
                raise _without_debug(exc) from None
            raise
        self.seen.append(_Seen(label, True, result.primary, result.secondary, result.warnings))
        return result

    def last(self, label: str) -> _Seen | None:
        return next((seen for seen in reversed(self.seen) if seen.label == label), None)


def _without_debug(exc: YtDlpError) -> YtDlpError:
    # Only ERROR and WARNING lines survive: the preview is cut mid-line, so a debug fragment need not start
    # with "[debug]".
    kept = "\n".join(line for line in exc.preview.splitlines() if line.startswith(("ERROR:", "WARNING:")))
    return YtDlpError(exc.primary, exc.secondary, kept, exc.returncode)


class VideoStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True)
class VideoOutcome:
    video_id: str
    status: VideoStatus
    access: Access | None = None
    chapters: int = 0
    captions: tuple[str, ...] = ()
    audio_s: float | None = None
    reused: tuple[str, ...] = ()
    error: str | None = None
    detail: str = ""
    fix: str = ""
    auth_error: bool = False

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.video_id,
            "status": self.status.value,
            "error": self.error,
            "detail": self.detail,
            "fix": self.fix,
            "cookies_used": self.access.cookies_used if self.access is not None else None,
            "chapters": self.chapters,
            "captions": list(self.captions),
            "audio_s": self.audio_s,
            "reused": list(self.reused),
        }


@dataclass(frozen=True)
class BatchReport:
    videos: tuple[VideoOutcome, ...]
    stopped: BatchStopError | None = None
    capped: bool = False
    started_at: datetime = field(default_factory=_utcnow)
    finished_at: datetime = field(default_factory=_utcnow)

    @property
    def exit_code(self) -> int:
        """0 ok, 1 some videos failed, 2 an auth error, 3 the batch stopped."""
        if self.stopped is not None:
            return 3
        failed = [v for v in self.videos if v.status is not VideoStatus.OK]
        if any(v.auth_error for v in failed):
            return 2
        return 1 if failed else 0

    def to_json(self) -> dict[str, object]:
        stopped = None
        if self.stopped is not None:
            until = self.stopped.until
            stopped = {
                "signal": self.stopped.signal.value,
                "until": until.isoformat(timespec="seconds") if until is not None else None,
                "fix": stop_fix(self.stopped.signal, until),
            }
        return {
            "version": _REPORT_VERSION,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": self.finished_at.isoformat(timespec="seconds"),
            "stopped": stopped,
            "capped": self.capped,
            "videos": [video.to_json() for video in self.videos],
        }


class _Snapshots(contextlib.AbstractContextManager["_Snapshots"]):
    """One cookie snapshot per video, shared by its stages; `fresh()` replaces it after a rotation."""

    def __init__(self, source: CookieSource) -> None:
        self._source = source
        self._stack: contextlib.ExitStack | None = None
        self.current: Snapshot | None = None

    def get(self) -> Snapshot:
        return self.current if self.current is not None else self.fresh()

    def fresh(self) -> Snapshot:
        self.close()
        stack = contextlib.ExitStack()
        try:
            # Looked up at call time so tests (and preflight) share one patch point.
            self.current = stack.enter_context(snapshot.cookie_snapshot(self._source))
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        return self.current

    def close(self) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self.current = None

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


@dataclass(frozen=True)
class _Extracted:
    metadata: VideoMetadata
    cookies_used: bool
    logged_in: bool | None = None
    premium: bool | None = None


class FetchSession:
    """Fetches videos one at a time under the auth policy.

    `fetch_video` records per-video failures in its outcome. It raises `BatchStopError`, and
    `DependencyError` for a missing tool, because both would fail every later video too.
    """

    def __init__(
        self,
        runner: YtDlpRunner,
        *,
        source: CookieSource | None,
        out: Path,
        options: FetchOptions,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._runner = _GuardedRunner(runner, clock)
        self._source = source
        self._out = out
        self._options = options
        self._sleep = sleep
        self._source_failure: AuthError | None = None
        self._ytdlp_version: str | None = None

    def fetch_video(self, video_id: str, *, cookies_first: bool = False) -> VideoOutcome:
        vdir = paths.video_dir(video_id, self._out)
        self._runner.seen.clear()
        try:
            with contextlib.ExitStack() as stack:
                snaps = stack.enter_context(_Snapshots(self._source)) if self._source is not None else None
                return self._fetch(video_id, vdir, snaps, cookies_first=cookies_first)
        except AuthError as exc:
            if exc.state not in _PER_VIDEO_AUTH_STATES:
                self._source_failure = exc
            return self._failure(video_id, exc)
        except (BatchStopError, DependencyError):
            raise
        except YtverbatimError as exc:
            return self._failure(video_id, exc)

    def _fetch(self, video_id: str, vdir: Path, snaps: _Snapshots | None, *, cookies_first: bool) -> VideoOutcome:
        captions_done = not self._options.captions or _load_tracks(vdir) is not None
        audio_done = not self._options.audio or _audio_ready(vdir)
        prior = _load_prior(vdir, need_fresh=not (captions_done and audio_done))
        reused: list[str] = []
        if prior is not None:
            metadata, access = prior
            reused.append("info")
        else:
            extracted = self._extract(video_id, vdir, snaps, cookies_first=cookies_first)
            metadata = extracted.metadata
            atomic.write_json(vdir / METADATA_FILE, metadata.to_dict())
            access = self._access(metadata, extracted, ())
            atomic.write_text(vdir / ACCESS_FILE, access.model_dump_json(indent=2) + "\n")

        warnings: list[str] = list(access.warnings)
        captions: tuple[str, ...] = ()
        if self._options.captions:
            entries = _load_tracks(vdir)
            if entries is None:
                result = self._stage(snaps, access.cookies_used, lambda c: self._captions(vdir, metadata, c))
                warnings += result.warnings
                entries = _track_entries(result)
                if self._captions_completed():
                    (vdir / CAPTIONS_DIR).mkdir(exist_ok=True)
                    atomic.write_json(vdir / CAPTIONS_DIR / TRACKS_FILE, {"tracks": entries})
            else:
                reused.append("captions")
            captions = tuple(_track_label(entry) for entry in entries)

        audio_s: float | None = None
        if self._options.audio:
            audio_s = self._audio(video_id, vdir, metadata, snaps, access.cookies_used, reused)

        warnings += [w for seen in self._runner.seen for w in seen.warnings]
        access = access.model_copy(update={"warnings": _clean_warnings(warnings)})
        atomic.write_text(vdir / ACCESS_FILE, access.model_dump_json(indent=2) + "\n")
        return VideoOutcome(
            video_id=video_id,
            status=VideoStatus.OK,
            access=access,
            chapters=len(metadata.chapters),
            captions=captions,
            audio_s=audio_s,
            reused=tuple(reused),
        )

    def _extract(self, video_id: str, vdir: Path, snaps: _Snapshots | None, *, cookies_first: bool) -> _Extracted:
        policy = self._options.policy
        if policy == "always" or (cookies_first and policy == "auto"):
            return self._extract_with_cookies(video_id, vdir, self._cookie_snapshots(snaps, None))
        try:
            return _Extracted(fetch_info(self._runner, video_id, vdir), cookies_used=False)
        except YtDlpError as exc:
            if exc.primary not in ESCALATE_SIGNALS:
                raise
            return self._extract_with_cookies(video_id, vdir, self._cookie_snapshots(snaps, exc.primary))

    def _cookie_snapshots(self, snaps: _Snapshots | None, cause: Signal | None) -> _Snapshots:
        reason = f"YouTube requires sign-in ({cause.value})" if cause is not None else "this video uses cookies"
        if self._options.policy == "never":
            raise AuthError(AuthState.AUTH_REQUIRED, f"{reason}, but --policy is never")
        if snaps is None:
            raise AuthError(AuthState.AUTH_REQUIRED, f"{reason}, but no auth source is set")
        if self._source_failure is not None:
            failure = self._source_failure
            raise AuthError(failure.state, f"{failure.detail} (from an earlier video in this batch)".strip())
        return snaps

    def _extract_with_cookies(self, video_id: str, vdir: Path, snaps: _Snapshots) -> _Extracted:
        try:
            metadata = self._with_cookies(
                snaps, lambda cookies: fetch_info(self._runner, video_id, vdir, cookies=cookies, verbose=True)
            )
        except YtDlpError as exc:
            if exc.primary in (Signal.MEMBERS_ONLY, Signal.LOGIN_REQUIRED):
                raise self._login_verdict(exc.primary) from None
            raise
        seen = self._runner.last("metadata")
        secondary = seen.secondary if seen is not None else ()
        logged_in = Signal.AUTH_COOKIES_FOUND in secondary
        premium = True if Signal.PREMIUM in secondary else False if logged_in else None
        if premium is not None and self._source is not None:
            preflight.record_premium(self._source, premium)
        return _Extracted(metadata, cookies_used=True, logged_in=logged_in, premium=premium)

    def _login_verdict(self, cause: Signal) -> AuthError:
        """Tell "not a member" from "the login did not work" with one fresh `:ytfav` probe (A1)."""
        report = preflight.preflight(self._source, self._runner, refresh=True)
        if report.state is AuthState.OK:
            if cause is Signal.MEMBERS_ONLY:
                return AuthError(AuthState.NOT_MEMBER, "YouTube accepted the login but refused this members video")
            return AuthError(AuthState.AUTH_UNCERTAIN, "the login probe passed, yet the video still asks to sign in")
        if report.state in (AuthState.LOGGED_OUT, AuthState.COOKIE_FILE_LOGGED_OUT):
            return AuthError(report.state, report.detail)
        detail = f"the video refused the cookies ({cause.value}) and the login probe gave {report.state.value}"
        return AuthError(AuthState.AUTH_UNCERTAIN, detail)

    def _with_cookies[T](self, snaps: _Snapshots, call: Callable[[Path], T]) -> T:
        try:
            result = call(snaps.get().path)
        except YtDlpError as exc:
            if Signal.ROTATED not in exc.secondary:
                raise
            log.warning("YouTube rotated the cookies; re-reading them in %gs", ROTATION_WAIT_S)
            self._sleep(ROTATION_WAIT_S)
            try:
                result = call(snaps.fresh().path)
            except YtDlpError as again:
                if Signal.ROTATED in again.secondary:
                    raise AuthError(AuthState.ROTATED, "cookies were rotated again after a fresh snapshot") from None
                raise
        latest = self._runner.seen[-1] if self._runner.seen else None
        if latest is not None and Signal.ROTATED in latest.secondary and self._source is not None:
            # Served anyway, but the cached login is stale now; the next preflight must probe again.
            preflight.invalidate(self._source)
        return result

    def _stage[T](self, snaps: _Snapshots | None, cookies_used: bool, call: Callable[[Path | None], T]) -> T:
        if not cookies_used:
            return call(None)
        return self._with_cookies(self._cookie_snapshots(snaps, None), call)

    def _captions(self, vdir: Path, metadata: VideoMetadata, cookies: Path | None) -> CaptionsResult:
        return fetch_captions(
            self._runner,
            info_path(vdir),
            vdir / CAPTIONS_DIR,
            metadata,
            languages=self._options.languages,
            cookies=cookies,
        )

    def _captions_completed(self) -> bool:
        seen = self._runner.last(_CAPTIONS_LABEL)
        # No call at all means no wanted track exists, which is complete; a failed call (429) is not.
        return seen is None or seen.ok

    def _audio(
        self,
        video_id: str,
        vdir: Path,
        metadata: VideoMetadata,
        snaps: _Snapshots | None,
        cookies_used: bool,
        reused: list[str],
    ) -> float:
        expected = metadata.duration
        native = find_audio(vdir)
        if native is not None and _duration_if_valid(native, expected) is not None:
            reused.append("audio")
        else:
            native = self._stage(
                snaps,
                cookies_used,
                lambda c: fetch_audio(self._runner, video_id, vdir, vdir, expected_duration=expected, cookies=c),
            )
        wav = vdir / WAV_NAME
        wav_s = _duration_if_valid(wav, expected)
        if wav_s is not None:
            reused.append("wav")
            return wav_s
        to_wav16k(native, wav, expected_duration=expected)
        return validate_audio(wav, expected_duration=expected)

    def _access(self, metadata: VideoMetadata, extracted: _Extracted, warnings: Iterable[str]) -> Access:
        if self._ytdlp_version is None:
            self._ytdlp_version = version.installed().version
        member = True if extracted.cookies_used and metadata.availability == "subscriber_only" else None
        return Access(
            auth_source=_auth_source_kind(self._source),
            cookies_used=extracted.cookies_used,
            logged_in=extracted.logged_in,
            premium=extracted.premium,
            member=member,
            availability=metadata.availability,
            ytdlp_version=self._ytdlp_version,
            warnings=_clean_warnings(warnings),
        )

    def _failure(self, video_id: str, exc: YtverbatimError) -> VideoOutcome:
        match exc:
            case AuthError(state=state, detail=detail):
                fix = fix_for(state, **preflight.fix_context(self._source))
                return VideoOutcome(
                    video_id, VideoStatus.ERROR, error=state.value, detail=redact(detail), fix=fix, auth_error=True
                )
            case YtDlpError(primary=primary, preview=preview):
                cause = primary.value if primary is not None else "unclassified"
                return VideoOutcome(
                    video_id, VideoStatus.ERROR, error=cause, detail=_cause_line(preview), fix=_ytdlp_fix(exc)
                )
            case UnavailableError(kind=kind):
                return VideoOutcome(video_id, VideoStatus.ERROR, error=kind, detail=redact(str(exc)))
            case MediaValidationError():
                return VideoOutcome(video_id, VideoStatus.ERROR, error="media_invalid", detail=redact(str(exc)))
            case ProcTimeout():
                return VideoOutcome(video_id, VideoStatus.ERROR, error="timeout", detail=redact(str(exc)))
            case _:
                return VideoOutcome(video_id, VideoStatus.ERROR, error="error", detail=redact(str(exc)))


def _auth_source_kind(source: CookieSource | None) -> AuthSourceKind:
    match source:
        case None:
            return "none"
        case CookieFileSource():
            return "cookie-file"
        case BrowserSource(browser=browser):
            return _BROWSER_KINDS[browser]


def _clean_warnings(warnings: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(redact(w) for w in warnings if w))


def _cause_line(preview: str) -> str:
    lines = [line.strip() for line in redact(preview).splitlines() if line.strip()]
    cause = next((line for line in lines if line.startswith("ERROR:")), lines[0] if lines else "")
    return cause[:300]


def _ytdlp_fix(exc: YtDlpError) -> str:
    if js_challenge_failed(exc.primary, exc.secondary):
        return fix_for(AuthState.JS_RUNTIME_MISSING)
    signals = {exc.primary, *exc.secondary}
    if exc.primary is Signal.NO_FORMATS and not signals.isdisjoint({Signal.PO_TOKEN_GVS, Signal.PO_TOKEN_SUBS}):
        return fix_for(AuthState.PO_TOKEN_REQUIRED)
    if exc.primary is not None and exc.primary in _PERMANENT_FIXES:
        return _PERMANENT_FIXES[exc.primary]
    return "Upgrade yt-dlp in this environment and retry; if it persists, report the error above."


def _track_entries(result: CaptionsResult) -> list[dict[str, object]]:
    return [
        {
            "language": t.language,
            "kind": t.kind.value,
            "format": t.source_format,
            "word_timings": t.has_word_timings,
            "segments": len(t.segments),
        }
        for t in result.tracks
    ]


def _track_label(entry: dict[str, object]) -> str:
    timings = ", word timings" if entry.get("word_timings") is True else ""
    return f"{entry.get('language')} ({entry.get('kind')}, {entry.get('format')}{timings})"


def _load_tracks(vdir: Path) -> list[dict[str, object]] | None:
    """Return the recorded caption tracks, or None when the stage has to run (again)."""
    try:
        raw: object = json.loads((vdir / CAPTIONS_DIR / TRACKS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    tracks = cast("dict[str, object]", raw).get("tracks") if isinstance(raw, dict) else None
    if not isinstance(tracks, list):
        return None
    entries = [cast("dict[str, object]", t) for t in cast("list[object]", tracks) if isinstance(t, dict)]
    for entry in entries:
        track_file = vdir / CAPTIONS_DIR / f"{CAPTIONS_STEM}.{entry.get('language')}.{entry.get('format')}"
        if not track_file.is_file():
            return None
    return entries


def _audio_ready(vdir: Path) -> bool:
    try:
        expected = load_metadata(vdir).duration
    except (OSError, YtDlpError):
        return False
    native = find_audio(vdir)
    return (
        native is not None
        and _duration_if_valid(native, expected) is not None
        and _duration_if_valid(vdir / WAV_NAME, expected) is not None
    )


def _duration_if_valid(path: Path, expected: float | None) -> float | None:
    if not path.is_file():
        return None
    try:
        return validate_audio(path, expected_duration=expected)
    except MediaValidationError:
        return None


def _load_prior(vdir: Path, *, need_fresh: bool) -> tuple[VideoMetadata, Access] | None:
    """Reuse the saved extraction; with stages still to run it must be young enough for its signed URLs."""
    info = info_path(vdir)
    try:
        if need_fresh and time.time() - info.stat().st_mtime > INFO_MAX_AGE_S:
            return None
        metadata = load_metadata(vdir)
        access = Access.model_validate_json((vdir / ACCESS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, YtDlpError):
        return None
    if not (vdir / METADATA_FILE).is_file():
        return None
    return metadata, access


def sleep_args(video_count: int) -> tuple[str, ...]:
    return SLEEP_PRESET if video_count >= 2 else SINGLE_VIDEO_SLEEP_ARGS  # noqa: PLR2004


def _default_runner(sleep: Sequence[str]) -> YtDlpRunner:
    return SubprocessRunner(sleep_args=sleep)


@dataclass(frozen=True)
class PlannedVideo:
    video_id: str
    cookies_first: bool


def _channel_keys(entries: Iterable[str]) -> set[str]:
    keys: set[str] = set()
    for entry in entries:
        ref = ids.classify(entry)
        if ref is not None and ref.kind is ids.RefKind.CHANNEL:
            keys.add(ref.id.lower())
        elif entry.startswith("UC"):
            keys.add(f"channel/{entry}".lower())
        elif entry.startswith("@"):
            keys.add(entry.lower())
    return keys


def _needs_cookies(ref: ids.YouTubeRef, channel_keys: set[str]) -> bool:
    if ref.kind is ids.RefKind.PLAYLIST and ref.id.startswith(MEMBERS_PLAYLIST_PREFIX):
        return True
    return ref.kind is ids.RefKind.CHANNEL and ref.id.lower() in channel_keys


def plan_videos(
    runner: YtDlpRunner,
    targets: Sequence[str],
    *,
    source: CookieSource | None,
    options: FetchOptions,
) -> tuple[list[PlannedVideo], bool]:
    """Expand targets into at most `max_videos` videos; the flag says the cap cut the list short."""
    channel_keys = _channel_keys(options.cookie_channels)
    planned: dict[str, PlannedVideo] = {}
    capped = False
    for target in targets:
        ref = ids.classify(target)
        if ref is None:
            raise ValueError(f"not a YouTube video, playlist or channel: {target!r}")
        cookies_first = _needs_cookies(ref, channel_keys) and options.policy != "never"
        # Members playlists are listed with cookies too: anonymous listing of them is unverified.
        with contextlib.ExitStack() as stack:
            cookies: Path | None = None
            if cookies_first and ref.kind is ids.RefKind.PLAYLIST and source is not None:
                cookies = stack.enter_context(snapshot.cookie_snapshot(source)).path
            found = expand_target(runner, target, limit=options.max_videos, cookies=cookies)
        if ref.kind is not ids.RefKind.VIDEO and len(found) >= options.max_videos:
            capped = True
        for video_id in found:
            planned.setdefault(video_id, PlannedVideo(video_id, cookies_first))
    videos = list(planned.values())
    if len(videos) > options.max_videos:
        capped = True
        videos = videos[: options.max_videos]
    return videos, capped


def fetch(
    targets: Sequence[str],
    *,
    source: CookieSource | None,
    out: Path | None = None,
    options: FetchOptions | None = None,
    runner_factory: Callable[[Sequence[str]], YtDlpRunner] = _default_runner,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = _utcnow,
    on_video: Callable[[VideoOutcome], None] | None = None,
) -> BatchReport:
    """Fetch every video behind `targets` into `<out>/<id>/` and write `<out>/batch_report.json`.

    Raises `BatchStopError` before any request while a cooldown is active, and `AuthError(NO_SOURCE)` for
    policy `always` without a source. Per-video failures, and a batch stop mid-run, land in the report.
    """
    options = options or FetchOptions()
    out = out if out is not None else paths.default_output_root()
    started = clock()
    if not options.ignore_cooldown:
        check_cooldown(started)
    if options.policy == "always" and source is None:
        raise AuthError(AuthState.NO_SOURCE, "--policy always needs an auth source")

    videos, capped = plan_videos(_GuardedRunner(runner_factory(()), clock), targets, source=source, options=options)
    session = FetchSession(
        runner_factory(sleep_args(len(videos))), source=source, out=out, options=options, sleep=sleep, clock=clock
    )
    outcomes: list[VideoOutcome] = []
    stopped: BatchStopError | None = None
    try:
        for planned in videos:
            if stopped is not None:
                outcomes.append(VideoOutcome(planned.video_id, VideoStatus.NOT_ATTEMPTED))
                continue
            try:
                outcome = session.fetch_video(planned.video_id, cookies_first=planned.cookies_first)
            except BatchStopError as exc:
                stopped = exc
                outcome = VideoOutcome(
                    planned.video_id, VideoStatus.ERROR, error=exc.signal.value, fix=stop_fix(exc.signal, exc.until)
                )
            outcomes.append(outcome)
            if on_video is not None:
                on_video(outcome)
    finally:
        report = BatchReport(tuple(outcomes), stopped, capped, started, clock())
        out.mkdir(parents=True, exist_ok=True)
        atomic.write_json(out / REPORT_FILE, report.to_json())
    return report


__all__ = [
    "ESCALATE_SIGNALS",
    "STOP_STATES",
    "BatchReport",
    "Cooldown",
    "FetchOptions",
    "FetchSession",
    "PlannedVideo",
    "Policy",
    "VideoOutcome",
    "VideoStatus",
    "check_cooldown",
    "fetch",
    "plan_videos",
    "read_cooldown",
    "sleep_args",
    "start_cooldown",
    "stop_fix",
]
