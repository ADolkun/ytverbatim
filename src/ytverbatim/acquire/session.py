"""Fetch session: anonymous-first extraction, cookie escalation, rotation recovery, and batch stops.

Account footprint is the design constraint. Each video gets one `-J`, anonymous unless the target is known
to need a login; cookies are used only after YouTube asks for them, and a bot or rate signal stops the
whole batch and starts a cooldown instead of being retried, least of all with cookies.
"""

import contextlib
import fcntl
import json
import logging
import os
import time
from collections.abc import Callable, Generator, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, cast

from ytverbatim import atomic, ids, paths
from ytverbatim.acquire.audio import INFO_MAX_AGE_S, WAV_NAME, fetch_audio, find_audio, to_wav16k, validate_audio
from ytverbatim.acquire.captions import DEFAULT_LANGUAGES, CaptionsResult, fetch_captions
from ytverbatim.acquire.captions import FILE_STEM as CAPTIONS_STEM
from ytverbatim.acquire.expand import expand_target
from ytverbatim.acquire.metadata import VideoMetadata, fetch_info, info_path, load_metadata
from ytverbatim.auth import preflight, snapshot
from ytverbatim.auth.cooldown import (
    CAPTIONS_LABEL,
    STOP_STATES,
    Cooldown,
    GuardedRunner,
    active_cooldown,
    check_cooldown,
    read_cooldown,
    start_cooldown,
    stop_fix,
    utcnow,
)
from ytverbatim.auth.fixes import fix_for
from ytverbatim.auth.snapshot import Snapshot
from ytverbatim.auth.source import BrowserSource, CookieFileSource, CookieSource
from ytverbatim.auth.state import AuthState
from ytverbatim.contract.models import Access
from ytverbatim.errors import (
    AuthError,
    BatchStopError,
    BusyError,
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
from ytverbatim.ytdlp.types import Signal, YtDlpRunner

log = logging.getLogger(__name__)

type Policy = Literal["auto", "always", "never"]
type AuthSourceKind = Literal["none", "cookie-file", "chrome", "brave", "chromium", "edge", "vivaldi"]

ESCALATE_SIGNALS = frozenset({Signal.LOGIN_REQUIRED, Signal.MEMBERS_ONLY, Signal.AGE_GATE, Signal.PRIVATE})

# Chromium commits its cookie store about every 30 s, so a re-snapshot sooner reads the same stale values.
ROTATION_WAIT_S = 30.0

SINGLE_VIDEO_SLEEP_ARGS: tuple[str, ...] = ("--sleep-requests", "0.75", "--sleep-subtitles", "5")
MEMBERS_PLAYLIST_PREFIX = "UUMO"

ACCESS_FILE = "access.json"
METADATA_FILE = "metadata.json"
CAPTIONS_DIR = "captions"
TRACKS_FILE = "tracks.json"
REPORT_FILE = "batch_report.json"
LOCK_NAME = ".lock"
_REPORT_VERSION = 1

# Any other AuthError is about the auth source itself: once one happens, every later cookied request in the
# batch would fail the same way, so it is replayed without touching YouTube again. AUTH_UNCERTAIN is one
# video's refusal that the probe could not explain, so it says nothing about the next video.
_PER_VIDEO_AUTH_STATES = frozenset({AuthState.NOT_MEMBER, AuthState.AUTH_REQUIRED, AuthState.AUTH_UNCERTAIN})

_PERMANENT_FIXES: dict[Signal, str] = {
    Signal.UNAVAILABLE: "The video is unavailable or was removed; there is nothing to retry.",
    Signal.GEO: "The uploader blocks this video in your country.",
    Signal.DRM: "DRM-protected videos cannot be fetched.",
    Signal.PRIVATE: "Private video: only accounts the uploader invited can open it.",
    Signal.AGE_GATE: "Age-restricted: the auth profile needs a signed-in, age-verified YouTube account.",
    Signal.PREMIUM_ONLY: "YouTube Premium only: the auth profile's account needs Premium.",
}


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
    started_at: datetime = field(default_factory=utcnow)
    finished_at: datetime = field(default_factory=utcnow)

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
    `DependencyError` for a missing tool, because both would fail every later video too. A stop signal on a
    successful call lets that video finish; check `pending_stop` after each video, since the next
    `fetch_video` raises it.
    """

    def __init__(
        self,
        runner: YtDlpRunner,
        *,
        source: CookieSource | None,
        out: Path,
        options: FetchOptions,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._runner = GuardedRunner(runner, clock)
        self._source = source
        self._out = out
        self._options = options
        self._sleep = sleep
        self._source_failure: AuthError | None = None
        self._ytdlp_version: str | None = None

    def pending_stop(self) -> BatchStopError | None:
        """Return the stop that arrived on a successful call: that video finished, but no later one may start."""
        stop = self._runner.stop
        return BatchStopError(stop.signal, stop.until) if stop is not None else None

    def fetch_video(self, video_id: str, *, cookies_first: bool = False) -> VideoOutcome:
        if (stop := self.pending_stop()) is not None:
            raise stop
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
        report = preflight.preflight(
            self._source, self._runner, refresh=True, ignore_cooldown=self._options.ignore_cooldown
        )
        if report.cooldown is not None:
            raise BatchStopError(report.cooldown.signal, report.cooldown.until)
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
        seen = self._runner.last(CAPTIONS_LABEL)
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
            # Access validates the literal, so an unsupported browser still fails loud.
            return cast("AuthSourceKind", browser)


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


def default_runner(sleep: Sequence[str]) -> YtDlpRunner:
    return SubprocessRunner(sleep_args=sleep)


@contextlib.contextmanager
def video_lock(vdir: Path) -> Generator[None]:
    """Hold the video directory exclusively, or raise `BusyError` at once; waiting could mean hours of ASR."""
    vdir.mkdir(parents=True, exist_ok=True)
    fd = os.open(vdir / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BusyError(f"another ytverbatim run is processing {vdir.name}") from None
        yield
    finally:
        os.close(fd)


@dataclass(frozen=True)
class PlannedVideo:
    video_id: str
    cookies_first: bool


class VideoHook(Protocol):
    """What `fetch` does with each planned video; `ytverbatim run` plugs its whole per-video pipeline in here.

    `done` must answer without any request: during a cooldown a batch whose videos are all done still runs.
    `__call__` records the video's own failures in the outcome and raises only what ends the batch.
    """

    def done(self, video_id: str) -> bool: ...

    def __call__(self, session: "FetchSession", planned: PlannedVideo) -> VideoOutcome: ...


@dataclass(frozen=True)
class _FetchOnly:
    out: Path

    def done(self, video_id: str) -> bool:
        return False

    def __call__(self, session: "FetchSession", planned: PlannedVideo) -> VideoOutcome:
        try:
            with video_lock(paths.video_dir(planned.video_id, self.out)):
                return session.fetch_video(planned.video_id, cookies_first=planned.cookies_first)
        except BusyError as exc:
            return VideoOutcome(planned.video_id, VideoStatus.ERROR, error="busy", detail=str(exc))


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
    runner_factory: Callable[[Sequence[str]], YtDlpRunner] = default_runner,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = utcnow,
    on_video: Callable[[VideoOutcome], None] | None = None,
    hook: VideoHook | None = None,
) -> BatchReport:
    """Fetch every video behind `targets` into `<out>/<id>/` and write `<out>/batch_report.json`.

    Each video runs under its directory lock, through `hook` when one is given. Raises `BatchStopError` before
    any request while a cooldown is active (unless every target is a video the hook reports done), or when
    planning itself met a stop; and `AuthError(NO_SOURCE)` for policy `always` without a source. Per-video
    failures, and a batch stop mid-run, land in the report.
    """
    options = options or FetchOptions()
    out = out if out is not None else paths.default_output_root()
    run_video = hook if hook is not None else _FetchOnly(out)
    started = clock()
    waived = False
    if not options.ignore_cooldown and (cooldown := active_cooldown(started)) is not None:
        if not _all_done(targets, run_video):
            raise BatchStopError(cooldown.signal, cooldown.until)
        waived = True
    if options.policy == "always" and source is None:
        raise AuthError(AuthState.NO_SOURCE, "--policy always needs an auth source")

    planner = GuardedRunner(runner_factory(()), clock)
    videos, capped = plan_videos(planner, targets, source=source, options=options)
    if planner.stop is not None:
        raise BatchStopError(planner.stop.signal, planner.stop.until)
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
                # The cooldown was waived only because nothing needed a request; a video undone since must not fetch.
                if waived and not run_video.done(planned.video_id):
                    check_cooldown(clock())
                outcome = run_video(session, planned)
            except BatchStopError as exc:
                stopped = exc
                outcome = VideoOutcome(
                    planned.video_id, VideoStatus.ERROR, error=exc.signal.value, fix=stop_fix(exc.signal, exc.until)
                )
            else:
                stopped = session.pending_stop()
            outcomes.append(outcome)
            if on_video is not None:
                on_video(outcome)
    finally:
        report = BatchReport(tuple(outcomes), stopped, capped, started, clock())
        out.mkdir(parents=True, exist_ok=True)
        atomic.write_json(out / REPORT_FILE, report.to_json())
    return report


def _all_done(targets: Sequence[str], hook: VideoHook) -> bool:
    refs = [ids.classify(target) for target in targets]
    return all(ref is not None and ref.kind is ids.RefKind.VIDEO and hook.done(ref.id) for ref in refs)


__all__ = [
    "ESCALATE_SIGNALS",
    "STOP_STATES",
    "BatchReport",
    "Cooldown",
    "FetchOptions",
    "FetchSession",
    "PlannedVideo",
    "Policy",
    "VideoHook",
    "VideoOutcome",
    "VideoStatus",
    "check_cooldown",
    "default_runner",
    "fetch",
    "plan_videos",
    "read_cooldown",
    "sleep_args",
    "start_cooldown",
    "stop_fix",
    "video_lock",
]
