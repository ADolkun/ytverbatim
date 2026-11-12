"""Opt-in live checks against real YouTube; never part of ``make check`` or CI.

Run with ``make live-smoke``. Output goes to the terminal only and must never be committed.

Environment:
- ``YTVERBATIM_SMOKE_PUBLIC_ID``: a public video (default: yt-dlp's own test video).
- ``YTVERBATIM_SMOKE_MEMBERS_ID``: a members-only video; its check is skipped when unset.
- ``YTVERBATIM_AUTH`` (or the saved source): enables the cookie snapshot plus ``:ytfav`` check.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

from ytverbatim.auth.snapshot import cookie_snapshot
from ytverbatim.auth.source import resolve_source
from ytverbatim.errors import YtDlpError, YtverbatimError
from ytverbatim.ids import video_url
from ytverbatim.ytdlp.runner import SubprocessRunner, redact
from ytverbatim.ytdlp.types import Signal

PUBLIC_ENV = "YTVERBATIM_SMOKE_PUBLIC_ID"
MEMBERS_ENV = "YTVERBATIM_SMOKE_MEMBERS_ID"
# yt-dlp's own primary test video; it replaced BaW_jenozKc, which YouTube now reports as unavailable.
DEFAULT_PUBLIC_ID = "YE7VzlLtp-4"
TIMEOUT_S = 180.0


@dataclass(frozen=True)
class Outcome:
    status: str
    detail: str


def _pass(detail: str) -> Outcome:
    return Outcome("PASS", detail)


def _fail(detail: str) -> Outcome:
    return Outcome("FAIL", detail)


def _skip(detail: str) -> Outcome:
    return Outcome("SKIP", detail)


class _CountingSleep:
    """Records retries instead of sleeping; any retry fails the members check."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _seconds: float) -> None:
        self.calls += 1


def check_public() -> Outcome:
    video_id = os.environ.get(PUBLIC_ENV) or DEFAULT_PUBLIC_ID
    result = SubprocessRunner().run(
        ["-J", "--no-playlist", video_url(video_id)], cookies=None, timeout_s=TIMEOUT_S, label="smoke-public"
    )
    info: object = json.loads(result.stdout)
    if not isinstance(info, dict):
        return _fail("yt-dlp output is not a JSON object")
    title, duration = info.get("title"), info.get("duration")
    if result.returncode != 0 or not isinstance(title, str) or not title:
        return _fail(f"exit {result.returncode}, title present: {bool(title)}")
    if not isinstance(duration, int | float) or isinstance(duration, bool) or duration <= 0:
        return _fail(f"no usable duration ({duration!r})")
    return _pass(f"{video_id}: exit 0, title present, duration {duration:g}s")


def check_members() -> Outcome:
    video_id = os.environ.get(MEMBERS_ENV)
    if not video_id:
        return _skip(f"{MEMBERS_ENV} is not set")
    sleep = _CountingSleep()
    try:
        SubprocessRunner(sleep=sleep).run(
            ["-J", "--no-playlist", video_url(video_id)], cookies=None, timeout_s=TIMEOUT_S, label="smoke-members"
        )
    except YtDlpError as exc:
        wanted = {Signal.MEMBERS_ONLY, Signal.LOGIN_REQUIRED}
        if exc.primary not in wanted:
            return _fail(f"classified as {exc.primary}, expected members_only or login_required")
        if sleep.calls:
            return _fail(f"classified as {exc.primary} but retried {sleep.calls} time(s)")
        return _pass(f"anonymous -J classified as {exc.primary}, no retry")
    return _fail("anonymous -J succeeded; the video is not members-only")


def check_auth() -> Outcome:
    source = resolve_source()
    if source is None:
        return _skip("no auth source configured")
    with cookie_snapshot(source) as snap:
        result = SubprocessRunner().run(
            ["--flat-playlist", "-I", "1", "--print", "id", ":ytfav"],
            cookies=snap.path,
            timeout_s=TIMEOUT_S,
            label="smoke-ytfav",
        )
    returned = "yes" if result.stdout.strip() else "no"
    return _pass(f"snapshot taken, :ytfav exit 0, id returned: {returned}")


CHECKS: tuple[tuple[str, Callable[[], Outcome]], ...] = (
    ("public -J", check_public),
    ("members-only anonymous -J", check_members),
    ("cookie snapshot + :ytfav", check_auth),
)


def main() -> int:
    # Runner warnings would interleave with the one-line-per-check report.
    logging.basicConfig(level=logging.ERROR)
    failed = False
    for name, check in CHECKS:
        try:
            outcome = check()
        except (YtverbatimError, ValueError) as exc:
            first_line = str(exc).strip().splitlines()[:1] or [type(exc).__name__]
            outcome = _fail(redact(first_line[0]))
        failed = failed or outcome.status == "FAIL"
        print(f"{outcome.status}  {name}: {outcome.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
