import json
import logging
import os
import sys
import time
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ytverbatim import atomic, paths, proc
from ytverbatim.auth import chromium, cookiefile
from ytverbatim.auth.cookiefile import CookieEntry
from ytverbatim.auth.source import BrowserSource, CookieFileSource, CookieSource
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError, ProcTimeout, YtverbatimError
from ytverbatim.ytdlp.runner import redact

log = logging.getLogger(__name__)

ALLOWED_DOMAINS = ("youtube.com", "google.com")
# The hosts whose cookies yt-dlp sends to https://www.youtube.com, where it checks the login.
_YOUTUBE_HOSTS = frozenset({"youtube.com", "www.youtube.com"})
_LOGIN_COOKIE = "LOGIN_INFO"
_SAPISID_COOKIES = frozenset({"SAPISID", "__Secure-1PAPISID", "__Secure-3PAPISID"})

SNAPSHOT_MODE = 0o600
STALE_AFTER_S = 24 * 3600
# Generous: a Keychain prompt waits for a click; a normal extraction of ~1000 cookies takes seconds.
DEFAULT_TIMEOUT_S = 120.0
CHILD_MODULE = "ytverbatim.auth._snapshot_child"


@dataclass(frozen=True)
class SnapshotSummary:
    counts: dict[str, int]
    decrypt_failures: int
    logged_in: bool
    warnings: tuple[str, ...]
    # The browser profile directory actually read; None for a cookie file.
    profile: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "counts": self.counts,
            "decrypt_failures": self.decrypt_failures,
            "logged_in": self.logged_in,
            "warnings": list(self.warnings),
            "profile": self.profile,
        }


@dataclass(frozen=True)
class Snapshot:
    path: Path
    summary: SnapshotSummary


def allowed_domain(domain: str) -> str | None:
    host = domain.lstrip(".").lower()
    for allowed in ALLOWED_DOMAINS:
        if host == allowed or host.endswith(f".{allowed}"):
            return allowed
    return None


def filter_cookies(entries: Iterable[CookieEntry], now: float) -> list[CookieEntry]:
    return [e for e in entries if allowed_domain(e.domain) is not None and not e.is_expired(now)]


def is_logged_in(entries: Iterable[CookieEntry], now: float) -> bool:
    """Offline mirror of yt-dlp's ``_has_auth_cookies`` for the cookies sent to www.youtube.com."""
    names = {e.name for e in entries if e.domain.lstrip(".").lower() in _YOUTUBE_HOSTS and not e.is_expired(now)}
    return _LOGIN_COOKIE in names and not names.isdisjoint(_SAPISID_COOKIES)


def write_snapshot(
    entries: Iterable[CookieEntry],
    out: Path,
    *,
    source: CookieSource,
    decrypt_failures: int = 0,
    warnings: tuple[str, ...] = (),
    now: float | None = None,
) -> SnapshotSummary:
    """Filter, check the login offline and write ``out`` with mode 0600; logged-out jars are never written."""
    now = time.time() if now is None else now
    kept = filter_cookies(entries, now)
    counts = dict.fromkeys(ALLOWED_DOMAINS, 0)
    for entry in kept:
        domain = allowed_domain(entry.domain)
        if domain is not None:
            counts[domain] += 1
    profile = source.profile if isinstance(source, BrowserSource) else None
    summary = SnapshotSummary(counts, decrypt_failures, is_logged_in(kept, now), warnings, profile)
    if not summary.logged_in:
        raise _logged_out_error(source, summary)
    atomic.write_text(out, cookiefile.render(kept), mode=SNAPSHOT_MODE)
    return summary


def _logged_out_error(source: CookieSource, summary: SnapshotSummary) -> AuthError:
    if isinstance(source, CookieFileSource):
        return AuthError(
            AuthState.COOKIE_FILE_LOGGED_OUT,
            "the exported cookie file is signed out or expired; re-export it from a fresh private window",
        )
    where = f"{source.browser} profile {source.profile!r}" if source.profile else source.browser
    # Undecryptable cookies or a refused/wrong key explain a missing login better than "signed out" does,
    # and telling the user to sign in again would not fix them.
    reasons: list[str] = []
    if summary.decrypt_failures:
        reasons.append(f"{summary.decrypt_failures} cookies could not be decrypted")
    if "keychain" in summary.warnings:
        reasons.append("macOS Keychain refused the cookie key")
    if "wrong_key" in summary.warnings:
        reasons.append("the cookie key looks wrong")
    if reasons:
        return AuthError(
            AuthState.KEYCHAIN_DENIED,
            f"no signed-in YouTube session could be read from {where} ({'; '.join(reasons)}); allow "
            f"/usr/bin/security for {chromium.keychain_item(source.browser)!r} in Keychain Access, then retry",
        )
    return AuthError(
        AuthState.LOGGED_OUT,
        f"no signed-in YouTube session in {where} ({summary.counts['youtube.com']} youtube.com cookies); "
        "sign in to YouTube in that browser profile, then retry",
    )


def new_snapshot_path() -> Path:
    return paths.snapshots_dir() / f"{os.getpid()}-{time.time_ns()}.txt"


def cleanup_stale_snapshots(*, max_age_s: float = STALE_AFTER_S, now: float | None = None) -> int:
    """Delete snapshots (and atomic-write leftovers) a crashed run left behind."""
    cutoff = (time.time() if now is None else now) - max_age_s
    removed = 0
    for entry in paths.snapshots_dir().iterdir():
        try:
            if entry.is_file() and entry.lstat().st_mtime < cutoff:
                entry.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


def _summary_from_json(raw: dict[str, object]) -> SnapshotSummary:
    counts = raw.get("counts")
    warnings = raw.get("warnings")
    failures = raw.get("decrypt_failures")
    profile = raw.get("profile")
    return SnapshotSummary(
        counts={str(k): v for k, v in counts.items() if isinstance(v, int)} if isinstance(counts, dict) else {},
        decrypt_failures=failures if isinstance(failures, int) else 0,
        logged_in=raw.get("logged_in") is True,
        warnings=tuple(str(w) for w in warnings) if isinstance(warnings, list) else (),
        profile=profile if isinstance(profile, str) else None,
    )


def _parse_child_output(stdout: str, stderr: str, returncode: int) -> SnapshotSummary:
    lines = stdout.strip().splitlines()
    try:
        raw = json.loads(lines[-1]) if lines else None
    except ValueError:
        raw = None
    if not isinstance(raw, dict):
        tail = stderr.strip().splitlines()[-1:] or ["no output"]
        raise YtverbatimError(redact(f"cookie snapshot child failed with exit {returncode}: {tail[0]}"))
    if raw.get("ok") is True:
        return _summary_from_json(raw)
    state = raw.get("state")
    detail = str(raw.get("detail", ""))
    if isinstance(state, str) and state in set(AuthState):
        raise AuthError(AuthState(state), detail)
    raise YtverbatimError(redact(f"cookie snapshot failed: {detail}"))


def snapshot_browser(source: BrowserSource, out: Path, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> SnapshotSummary:
    """Snapshot browser cookies in a child process, because macOS may block on a Keychain prompt."""
    # -P: never let a json.py or similar in the caller's working directory shadow a module in the child.
    argv = [sys.executable, "-P", "-m", CHILD_MODULE, "--browser", source.browser, "--out", str(out)]
    if source.profile is not None:
        argv += ["--profile", source.profile]
    try:
        result = proc.run(argv, timeout_s=timeout_s)
    except ProcTimeout:
        raise AuthError(
            AuthState.KEYCHAIN_DENIED,
            f"reading {source.browser} cookies did not finish within {timeout_s:g}s; macOS is probably showing "
            f"a Keychain prompt for {chromium.keychain_item(source.browser)!r}. Enter your login password, "
            "click Always Allow, then retry",
        ) from None
    summary = _parse_child_output(result.stdout, result.stderr, result.returncode)
    log.info(
        "cookie snapshot from %s profile %s: %s, %d undecryptable, %.1fs",
        source.browser,
        summary.profile,
        ", ".join(f"{n} {d}" for d, n in summary.counts.items()),
        summary.decrypt_failures,
        result.elapsed_s,
    )
    return summary


def take_snapshot(source: CookieSource, out: Path, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> SnapshotSummary:
    match source:
        case BrowserSource():
            return snapshot_browser(source, out, timeout_s=timeout_s)
        case CookieFileSource(path=path):
            # A private filtered copy: yt-dlp writes rotated values back to --cookies, and must not touch
            # the user's own file.
            return write_snapshot(cookiefile.read_cookie_file(path), out, source=source)


@contextmanager
def cookie_snapshot(source: CookieSource, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> Generator[Snapshot]:
    """Yield a per-job filtered snapshot of ``source`` and delete it on exit."""
    cleanup_stale_snapshots()
    out = new_snapshot_path()
    try:
        summary = take_snapshot(source, out, timeout_s=timeout_s)
        yield Snapshot(out, summary)
    finally:
        out.unlink(missing_ok=True)
