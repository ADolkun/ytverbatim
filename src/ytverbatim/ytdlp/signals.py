import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from ytverbatim.ytdlp.types import Signal

# YouTube writes "you’re" with U+2019; yt-dlp's own strings use ASCII.
_Q = "[’']"

# Precedence order for ERROR lines. The login hint ("Use --cookies-from-browser or --cookies for the
# authentication") is appended to every "sign in" reason, so the sign-in rows must precede LOGIN_REQUIRED.
# N_CHALLENGE and JS_MISSING are absent on purpose: yt-dlp only ever emits them as warnings.
_PRIMARY: tuple[tuple[Signal, re.Pattern[str]], ...] = tuple(
    (signal, re.compile(pattern, re.IGNORECASE))
    for signal, pattern in (
        (Signal.BOT_CHECK, rf"sign in to confirm you{_Q}re not a bot"),
        (Signal.IP_BLOCKED, r"your ip is likely being blocked by youtube"),
        (Signal.CAPTCHA, r"youtube is requiring a captcha challenge before playback"),
        (Signal.RATE_LIMITED, rf"this content isn{_Q}t available, try again later|has been rate-limited by youtube"),
        (Signal.HTTP_429, r"http error 429"),
        (Signal.AGE_GATE, r"sign in to confirm your age"),
        (Signal.PRIVATE, r"\bprivate video\b"),
        (
            Signal.MEMBERS_ONLY,
            rf"join this channel to get access to members-only content|available to this channel{_Q}s members"
            r"|members-only content",
        ),
        (Signal.PREMIUM_ONLY, r"youtube premium\b.*\bmembers|only available (?:to|for) (?:youtube )?premium"),
        (
            Signal.LOGIN_REQUIRED,
            r"login details are needed to download this content|this video is only available for registered users"
            r"|use --cookies-from-browser or --cookies for the authentication",
        ),
        (Signal.GEO, r"not made this video available in your country|not available from your location"),
        (
            Signal.UNAVAILABLE,
            r"video unavailable|this video is unavailable|this video has been removed|no longer available"
            r"|account associated with this video has been terminated",
        ),
        (Signal.DRM, r"this video is drm protected"),
        (Signal.PO_TOKEN_GVS, r"formats require a gvs po token which was not provided"),
        (Signal.PO_TOKEN_SUBS, r"subtitles require a po token which was not provided"),
        (Signal.NO_FORMATS, r"requested format is not available|no video formats found"),
        (Signal.HTTP_403, r"http error 403"),
        (
            Signal.TRANSIENT,
            r"http error 50[0234]|timed out|temporary failure in name resolution|connection reset"
            r"|connection aborted|unable to download webpage|incompleteread",
        ),
    )
)

_WARNING_ONLY: tuple[tuple[Signal, re.Pattern[str]], ...] = tuple(
    (signal, re.compile(pattern, re.IGNORECASE))
    for signal, pattern in (
        (Signal.JS_MISSING, r"no supported javascript runtime could be found"),
        (Signal.N_CHALLENGE, r"n challenge solving failed"),
    )
)

_SECONDARY: tuple[tuple[Signal, re.Pattern[str]], ...] = tuple(
    (signal, re.compile(pattern, re.IGNORECASE))
    for signal, pattern in (
        (Signal.ROTATED, r"the provided youtube account cookies are no longer valid"),
        (Signal.PREMIUM, r"detected youtube premium subscription"),
        (Signal.AUTH_COOKIES_FOUND, r"found youtube account cookies"),
        (Signal.AGE_VERIFICATION, r"youtube is requiring account age-verification"),
        (Signal.COOKIE_DB_MISSING, r"could not find \w+ cookies database"),
        (
            Signal.KEYCHAIN,
            r"find-generic-password failed|exception running find-generic-password"
            r"|cannot decrypt v1[01] cookies: no key found",
        ),
        (Signal.DECRYPT_PARTIAL, r"extracted (\d+) cookies from (\w+) \((\d+) could not be decrypted\)"),
        (Signal.COOKIE_LOAD, r"failed to load cookies"),
        (Signal.SKIP_CLIENT_NO_COOKIES, r'skipping client "\w+" since it does not support cookies'),
    )
)

_DECRYPT_PARTIAL = dict(_SECONDARY)[Signal.DECRYPT_PARTIAL]
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# Never retried: retrying a bot, captcha, IP or rate signal digs the hole deeper, and an unknown error
# is more likely a new permanent failure than a fluke.
RETRIES: dict[Signal, int] = {Signal.TRANSIENT: 2, Signal.HTTP_403: 1}


class LineKind(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    DEBUG = "debug"
    OTHER = "other"


@dataclass(frozen=True)
class DecryptCounts:
    extracted: int
    browser: str
    failed: int


@dataclass(frozen=True)
class Classification:
    primary: Signal | None
    secondary: tuple[Signal, ...]
    warnings: tuple[str, ...]
    decrypt_partial: DecryptCounts | None

    @property
    def js_challenge_failed(self) -> bool:
        return js_challenge_failed(self.primary, self.secondary)


def js_challenge_failed(primary: Signal | None, secondary: Iterable[Signal]) -> bool:
    """No formats because the JS challenge went unsolved: the fix is a JS runtime or a newer yt-dlp, not a retry."""
    return primary is Signal.NO_FORMATS and not {Signal.N_CHALLENGE, Signal.JS_MISSING}.isdisjoint(secondary)


def line_kind(line: str) -> LineKind:
    if line.startswith("ERROR:"):
        return LineKind.ERROR
    if line.startswith("WARNING:"):
        return LineKind.WARNING
    if line.startswith("[debug]"):
        return LineKind.DEBUG
    return LineKind.OTHER


def classify(stderr: str) -> Classification:
    """Decide the cause from ERROR lines only; every other line contributes secondary signals.

    Signals that ERROR lines match below the primary (such as the login hint on a bot check) are dropped,
    because they are part of the same message rather than separate evidence.
    """
    primary_rank: int | None = None
    secondary: dict[Signal, None] = {}
    warnings: list[str] = []
    decrypt: DecryptCounts | None = None

    for raw in _ANSI.sub("", stderr).splitlines():
        line = raw.strip()
        if not line:
            continue
        kind = line_kind(line)
        if kind is LineKind.ERROR:
            rank = next((i for i, (_, pattern) in enumerate(_PRIMARY) if pattern.search(line)), None)
            if rank is not None and (primary_rank is None or rank < primary_rank):
                primary_rank = rank
        else:
            if kind is LineKind.WARNING:
                warnings.append(line.removeprefix("WARNING:").strip())
            for signal, pattern in (*_PRIMARY, *_WARNING_ONLY):
                if pattern.search(line):
                    secondary[signal] = None
        for signal, pattern in _SECONDARY:
            if pattern.search(line):
                secondary[signal] = None
        if decrypt is None and (match := _DECRYPT_PARTIAL.search(line)):
            decrypt = DecryptCounts(int(match.group(1)), match.group(2), int(match.group(3)))

    primary = _PRIMARY[primary_rank][0] if primary_rank is not None else None
    return Classification(
        primary=primary,
        secondary=tuple(s for s in secondary if s is not primary),
        warnings=tuple(warnings),
        decrypt_partial=decrypt,
    )
