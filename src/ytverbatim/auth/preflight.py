"""Auth status: offline snapshot checks plus one ``:ytfav`` probe, with OK results cached for an hour.

The probe runs through the cooldown guard like every fetch call: a stop signal starts or extends the cooldown,
and while one is active the probe is skipped unless the caller passes ``ignore_cooldown``.

The cache (``auth/preflight.json``) holds only states, cookie counts, profile directory names and times. It
is keyed by a hash of the source spec and resolved profile, so a cookie-file path never lands in it.
"""

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from ytverbatim import atomic, paths
from ytverbatim.auth import chromium, snapshot
from ytverbatim.auth.cooldown import STOP_STATES, Cooldown, active_cooldown, guarded, shown_time
from ytverbatim.auth.fixes import fix_for
from ytverbatim.auth.source import BrowserSource, CookieFileSource, CookieSource, format_spec
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError, BatchStopError, YtDlpError
from ytverbatim.ytdlp import jsruntime, version
from ytverbatim.ytdlp.runner import redact
from ytverbatim.ytdlp.types import Signal, YtDlpRunner

TTL_S = 3600.0
YTDLP_STALE_DAYS = 45
PROBE_ARGS: tuple[str, ...] = ("--flat-playlist", "-I", "1", "--print", "id", ":ytfav")
PROBE_TIMEOUT_S = 120.0
_CACHE_VERSION = 1

_PO_TOKEN = frozenset({Signal.PO_TOKEN_GVS, Signal.PO_TOKEN_SUBS})


@dataclass(frozen=True)
class AuthReport:
    state: AuthState
    source: str
    profile: str | None
    # The offline cookie rule; None when no snapshot was taken.
    logged_in: bool | None
    premium: bool | None
    checked_at: datetime
    ytdlp_version: str
    ytdlp_age_days: int
    js_runtime: str | None
    fix: str
    counts: Mapping[str, int] = field(default_factory=dict[str, int])
    warnings: tuple[str, ...] = ()
    detail: str = ""
    cached: bool = False
    # Set when a cooldown is active: either the probe was skipped for it or the probe started it.
    cooldown: Cooldown | None = None
    probe_skipped: bool = False

    @property
    def ok(self) -> bool:
        """YouTube accepted the cookies on the ``:ytfav`` probe."""
        return self.state is AuthState.OK

    def to_json(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "ok": self.ok,
            "fix": self.fix,
            "detail": self.detail,
            "source": self.source,
            "profile": self.profile,
            "logged_in": self.logged_in,
            "premium": self.premium,
            "counts": dict(self.counts),
            "checked_at": self.checked_at.isoformat(timespec="seconds"),
            "cached": self.cached,
            "ytdlp_version": self.ytdlp_version,
            "ytdlp_age_days": self.ytdlp_age_days,
            "js_runtime": self.js_runtime,
            "warnings": list(self.warnings),
            "cooldown_until": self.cooldown.until.isoformat(timespec="seconds") if self.cooldown else None,
            "probe_skipped": self.probe_skipped,
        }


def premium_text(premium: bool | None) -> str:
    return "unknown" if premium is None else "yes" if premium else "no"


def fix_context(source: CookieSource | None, profile: str | None = None) -> dict[str, str | None]:
    if isinstance(source, BrowserSource):
        return {
            "browser": source.browser,
            "profile": profile or source.profile,
            "keychain_item": chromium.keychain_item(source.browser),
        }
    return {"profile": profile}


@dataclass(frozen=True)
class _Outcome:
    state: AuthState
    profile: str | None = None
    logged_in: bool | None = None
    counts: Mapping[str, int] = field(default_factory=dict[str, int])
    warnings: tuple[str, ...] = ()
    detail: str = ""
    checked_at: float | None = None
    cached: bool = False
    cooldown: Cooldown | None = None
    probe_skipped: bool = False


def preflight(
    source: CookieSource | None,
    runner: YtDlpRunner,
    refresh: bool = False,
    *,
    now: float | None = None,
    ignore_cooldown: bool = False,
) -> AuthReport:
    """Check ``source`` offline (snapshot) and online (``:ytfav``); a fresh OK result is served from cache.

    Only OK results are cached, so a problem the user just fixed never lingers for an hour. Raises
    `YtverbatimError` when the cooldown file is unreadable, unless ``ignore_cooldown`` is set.
    """
    now = time.time() if now is None else now
    outcome, premium = _check(source, runner, refresh=refresh, now=now, ignore_cooldown=ignore_cooldown)
    return _report(outcome, source, premium, now)


def _check(
    source: CookieSource | None, runner: YtDlpRunner, *, refresh: bool, now: float, ignore_cooldown: bool
) -> tuple[_Outcome, bool | None]:
    if source is None:
        return _Outcome(AuthState.NO_SOURCE), None
    try:
        profile = resolve_profile(source)
    except AuthError as exc:
        return _Outcome(exc.state, _named_profile(source), detail=exc.detail), None
    key = cache_key(source, profile)
    entry = _load_entries().get(key, {})
    premium = _premium(entry)
    if not refresh and (cached := _cached_outcome(entry, now)) is not None:
        return cached, premium
    try:
        with snapshot.cookie_snapshot(source) as snap:
            summary = snap.summary
            probed = _probe(runner, snap.path, source, now=now, ignore_cooldown=ignore_cooldown)
    except AuthError as exc:
        return _Outcome(exc.state, profile, detail=exc.detail), premium
    outcome = replace(
        probed, profile=summary.profile or profile, logged_in=summary.logged_in, counts=dict(summary.counts)
    )
    if outcome.state is AuthState.OK:
        _store_result(key, outcome, now)
    return outcome, premium


def resolve_profile(source: CookieSource) -> str | None:
    """Return the browser profile directory a snapshot would read, raising the errors the snapshot would."""
    match source:
        case BrowserSource(browser=browser, profile=profile):
            return chromium.cookie_db(browser, profile).profile
        case CookieFileSource():
            return None


def cache_key(source: CookieSource, profile: str | None) -> str:
    return hashlib.sha256(f"{format_spec(source)}\0{profile or ''}".encode()).hexdigest()[:24]


def record_premium(source: CookieSource, premium: bool, *, now: float | None = None) -> None:
    """Remember the Premium status a cookied fetch saw in ``-v`` output; it outlives the login TTL."""
    stamp = time.time() if now is None else now
    _update(cache_key(source, resolve_profile(source)), "premium", {"value": premium, "recorded_at": stamp}, now=stamp)


def invalidate(source: CookieSource) -> None:
    """Drop the cached login result, for example after yt-dlp reports rotated cookies."""
    _update(cache_key(source, resolve_profile(source)), "result", None)


def _named_profile(source: CookieSource) -> str | None:
    return source.profile if isinstance(source, BrowserSource) else None


def _probe(runner: YtDlpRunner, cookies: Path, source: CookieSource, *, now: float, ignore_cooldown: bool) -> _Outcome:
    when = datetime.fromtimestamp(now, UTC)
    if not ignore_cooldown and (cooldown := active_cooldown(when)) is not None:
        detail = f"cooldown active until {shown_time(cooldown.until)}; the :ytfav probe was skipped"
        return _Outcome(STOP_STATES[cooldown.signal], detail=detail, cooldown=cooldown, probe_skipped=True)
    guard = guarded(runner, lambda: when)
    try:
        result = guard.run(PROBE_ARGS, cookies=cookies, timeout_s=PROBE_TIMEOUT_S, label="auth-probe")
    except BatchStopError as exc:
        return _stopped(exc.signal, guard.stop)
    except YtDlpError as exc:
        return _failed_probe(exc, source)
    if guard.stop is not None:
        return _stopped(guard.stop.signal, guard.stop)
    # stdout is the first liked video's id (or empty for an empty list); it is never read or kept.
    warnings = (AuthState.PO_TOKEN_REQUIRED.value,) if not _PO_TOKEN.isdisjoint(result.secondary) else ()
    if Signal.ROTATED in result.secondary:
        return _Outcome(AuthState.ROTATED, detail="yt-dlp reported the account cookies are no longer valid")
    return _Outcome(AuthState.OK, warnings=warnings)


def _stopped(signal: Signal, cooldown: Cooldown | None) -> _Outcome:
    until = f"; cooldown until {shown_time(cooldown.until)}" if cooldown is not None else ""
    return _Outcome(STOP_STATES[signal], detail=f"the :ytfav probe hit {signal.value}{until}", cooldown=cooldown)


def _failed_probe(exc: YtDlpError, source: CookieSource) -> _Outcome:
    # The stderr preview is not kept: the probe's output can name a liked video.
    if Signal.ROTATED in exc.secondary:
        return _Outcome(AuthState.ROTATED, detail="yt-dlp reported the account cookies are no longer valid")
    if exc.primary is Signal.LOGIN_REQUIRED:
        state = AuthState.COOKIE_FILE_LOGGED_OUT if isinstance(source, CookieFileSource) else AuthState.LOGGED_OUT
        return _Outcome(state, detail="YouTube did not accept the cookies as a signed-in session")
    cause = exc.primary.value if exc.primary is not None else "an unclassified error"
    return _Outcome(AuthState.AUTH_UNCERTAIN, detail=f"the :ytfav probe failed with {cause} (exit {exc.returncode})")


def _report(outcome: _Outcome, source: CookieSource | None, premium: bool | None, now: float) -> AuthReport:
    installed = version.installed()
    js_runtime = jsruntime.detect().name
    warnings = list(outcome.warnings)
    if js_runtime is None:
        warnings.append(AuthState.JS_RUNTIME_MISSING.value)
    if installed.age_days() > YTDLP_STALE_DAYS:
        warnings.append(AuthState.YTDLP_STALE.value)
    checked_at = datetime.fromtimestamp(outcome.checked_at if outcome.checked_at is not None else now, UTC)
    fix = fix_for(
        outcome.state,
        **fix_context(source, outcome.profile),
        premium=premium_text(premium),
        verified=checked_at.astimezone().strftime("%Y-%m-%d %H:%M"),
        until=shown_time(outcome.cooldown.until) if outcome.cooldown is not None else None,
    )
    return AuthReport(
        state=outcome.state,
        source=redact(format_spec(source)),
        profile=outcome.profile,
        logged_in=outcome.logged_in,
        premium=premium,
        checked_at=checked_at,
        ytdlp_version=installed.version,
        ytdlp_age_days=installed.age_days(),
        js_runtime=js_runtime,
        fix=fix,
        counts=outcome.counts,
        warnings=tuple(dict.fromkeys(warnings)),
        detail=outcome.detail,
        cached=outcome.cached,
        cooldown=outcome.cooldown,
        probe_skipped=outcome.probe_skipped,
    )


type _Entry = dict[str, object]


def _load_entries() -> dict[str, _Entry]:
    try:
        raw: object = json.loads(paths.preflight_cache_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A cache: missing or corrupt only costs one probe.
        return {}
    if not isinstance(raw, dict) or raw.get("version") != _CACHE_VERSION:
        return {}
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {str(k): v for k, v in entries.items() if isinstance(v, dict)}


def _update(key: str, slot: str, value: object, *, now: float | None = None) -> None:
    now = time.time() if now is None else now
    entries = _load_entries()
    entry = entries.setdefault(key, {})
    if value is None:
        entry.pop(slot, None)
    else:
        entry[slot] = value
    for name in list(entries):
        if _cached_outcome(entries[name], now) is None:
            entries[name].pop("result", None)
        if not entries[name]:
            del entries[name]
    atomic.write_json(paths.preflight_cache_file(), {"version": _CACHE_VERSION, "entries": entries}, mode=0o600)


def _store_result(key: str, outcome: _Outcome, now: float) -> None:
    result = {
        "checked_at": now,
        "profile": outcome.profile,
        "logged_in": outcome.logged_in,
        "counts": dict(outcome.counts),
        "warnings": list(outcome.warnings),
    }
    _update(key, "result", result, now=now)


def _cached_outcome(entry: _Entry, now: float) -> _Outcome | None:
    result = entry.get("result")
    if not isinstance(result, dict):
        return None
    checked_at = result.get("checked_at")
    if not isinstance(checked_at, int | float) or not 0 <= now - checked_at < TTL_S:
        return None
    profile = result.get("profile")
    counts = result.get("counts")
    warnings = result.get("warnings")
    return _Outcome(
        AuthState.OK,
        profile=profile if isinstance(profile, str) else None,
        logged_in=result.get("logged_in") is True,
        counts={str(k): v for k, v in counts.items() if isinstance(v, int)} if isinstance(counts, dict) else {},
        warnings=tuple(str(w) for w in warnings) if isinstance(warnings, list) else (),
        checked_at=float(checked_at),
        cached=True,
    )


def _premium(entry: _Entry) -> bool | None:
    recorded = entry.get("premium")
    if isinstance(recorded, dict):
        value = recorded.get("value")
        if isinstance(value, bool):
            return value
    return None
