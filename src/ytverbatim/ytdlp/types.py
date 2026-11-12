from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class Signal(StrEnum):
    BOT_CHECK = "bot_check"
    IP_BLOCKED = "ip_blocked"
    CAPTCHA = "captcha"
    RATE_LIMITED = "rate_limited"
    AGE_GATE = "age_gate"
    PRIVATE = "private"
    MEMBERS_ONLY = "members_only"
    PREMIUM_ONLY = "premium_only"
    LOGIN_REQUIRED = "login_required"
    GEO = "geo"
    UNAVAILABLE = "unavailable"
    DRM = "drm"
    PO_TOKEN_GVS = "po_token_gvs"
    PO_TOKEN_SUBS = "po_token_subs"
    JS_MISSING = "js_missing"
    N_CHALLENGE = "n_challenge"
    NO_FORMATS = "no_formats"
    HTTP_403 = "http_403"
    # Bare urllib 429, kept apart from RATE_LIMITED: on a subtitle download it only costs that video its captions.
    HTTP_429 = "http_429"
    TRANSIENT = "transient"
    ROTATED = "rotated"
    PREMIUM = "premium"
    AUTH_COOKIES_FOUND = "auth_cookies_found"
    AGE_VERIFICATION = "age_verification"
    COOKIE_DB_MISSING = "cookie_db_missing"
    KEYCHAIN = "keychain"
    DECRYPT_PARTIAL = "decrypt_partial"
    COOKIE_LOAD = "cookie_load"
    SKIP_CLIENT_NO_COOKIES = "skip_client_no_cookies"


@dataclass(frozen=True)
class YtDlpResult:
    returncode: int
    stdout: str
    stderr: str
    primary: Signal | None
    secondary: tuple[Signal, ...]
    warnings: tuple[str, ...]


class YtDlpRunner(Protocol):
    def run(self, args: Sequence[str], *, cookies: Path | None, timeout_s: float, label: str) -> YtDlpResult:
        """Run one yt-dlp call; return on exit 0 and raise YtDlpError on any failure."""
        ...
