"""Child-process entry point for browser cookie snapshots; the only module that imports yt-dlp's cookie code.

Prints one JSON line on stdout: ``{"ok": true, counts, decrypt_failures, logged_in, warnings, profile}`` or
``{"ok": false, state, detail}``. It never prints cookie values or raw log messages.
"""

import argparse
import http.cookiejar
import json
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from ytverbatim.auth import chromium
from ytverbatim.auth.cookiefile import CookieEntry
from ytverbatim.auth.snapshot import write_snapshot
from ytverbatim.auth.source import BrowserSource
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError
from ytverbatim.ytdlp.runner import redact

_KEYCHAIN = re.compile(
    r"find-generic-password failed|exception running find-generic-password|cannot decrypt v1[01] cookies: no key found",
    re.I,
)
_EXTRACTED = re.compile(r"extracted (\d+) cookies from \w+ \((\d+) could not be decrypted\)", re.I)
_WARNING_KINDS = (
    ("keychain", _KEYCHAIN),
    # yt-dlp's per-cookie AES failures; with a wrong key every v10 cookie fails this way.
    ("wrong_key", re.compile(r"possibly the key is wrong", re.I)),
    ("unknown_cookie_version", re.compile(r"unknown cookie version", re.I)),
    ("no_sqlite", re.compile(r"without sqlite3 support", re.I)),
)


class CaptureLogger:
    """yt-dlp cookie logger that keeps only message kinds and counts, so no value or path can leak."""

    def __init__(self) -> None:
        self.kinds: list[str] = []
        self.decrypt_failures = 0

    def _record(self, message: str) -> None:
        kind = next((k for k, pattern in _WARNING_KINDS if pattern.search(message)), "other")
        if kind not in self.kinds:
            self.kinds.append(kind)

    def debug(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        if found := _EXTRACTED.search(message):
            self.decrypt_failures = int(found.group(2))

    def warning(self, message: str, *, once: bool = False, only_once: bool = False) -> None:
        self._record(message)

    def error(self, message: str) -> None:
        self._record(message)

    def stdout(self, message: str) -> None:
        pass

    def stderr(self, message: str) -> None:
        self._record(message)


type Extractor = Callable[[str, str, CaptureLogger], Iterable[http.cookiejar.Cookie]]


def _entry(cookie: http.cookiejar.Cookie) -> CookieEntry:
    name, value = cookie.name, cookie.value
    if value is None:
        # Matches yt-dlp's cookies.txt convention for a nameless cookie.
        name, value = "", name
    return CookieEntry(
        domain=cookie.domain,
        path=cookie.path,
        secure=cookie.secure,
        expires=cookie.expires or None,
        name=name,
        value=value,
        http_only=cookie.has_nonstandard_attr("HttpOnly"),
    )


def run_snapshot(browser: str, profile: str | None, out: Path, extract: Extractor) -> dict[str, object]:
    store = chromium.cookie_db(browser, profile)
    pin_dir = chromium.pin_cookie_db(store.path)
    logger = CaptureLogger()
    try:
        jar = list(extract(browser, str(pin_dir), logger))
    except FileNotFoundError:
        raise AuthError(
            AuthState.PROFILE_NOT_FOUND, f"the {browser} cookie database disappeared while reading it"
        ) from None
    summary = write_snapshot(
        (_entry(c) for c in jar),
        out,
        source=BrowserSource(browser, store.profile),
        decrypt_failures=logger.decrypt_failures,
        warnings=tuple(logger.kinds),
    )
    return {"ok": True, **summary.to_json()}


def _ytdlp_extract(browser: str, profile: str, logger: CaptureLogger) -> Iterable[http.cookiejar.Cookie]:
    from yt_dlp.cookies import extract_cookies_from_browser

    jar: Iterable[http.cookiejar.Cookie] = extract_cookies_from_browser(browser, profile, logger=logger)
    return jar


def main(argv: Sequence[str] | None = None, extract: Extractor = _ytdlp_extract) -> int:
    parser = argparse.ArgumentParser(prog="ytverbatim.auth._snapshot_child")
    parser.add_argument("--browser", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    browser: str = args.browser
    profile: str | None = args.profile
    out: Path = args.out
    try:
        result = run_snapshot(browser, profile, out, extract)
    except AuthError as exc:
        result = {"ok": False, "state": exc.state.value, "detail": exc.detail}
    except Exception as exc:
        result = {"ok": False, "state": None, "detail": redact(f"{type(exc).__name__}: {exc}")}
    sys.stdout.write(json.dumps(result) + "\n")
    return 0 if result["ok"] is True else 1


if __name__ == "__main__":
    sys.exit(main())
