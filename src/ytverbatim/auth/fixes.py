from collections.abc import Mapping
from types import MappingProxyType

from ytverbatim.auth.state import AuthState

DOCS = "docs/auth.md"

# One fix per state. `{name}` slots are filled from context where the caller knows it; `_DEFAULTS`
# keeps every message actionable when it does not (preflight has no cooldown end, for instance).
FIXES: Mapping[AuthState, str] = MappingProxyType(
    {
        AuthState.NO_SOURCE: (
            "No auth source set; members-only videos will fail. "
            "Run: ytverbatim auth profiles, then ytverbatim auth use chrome:<DIR>"
        ),
        AuthState.BROWSER_NOT_FOUND: (
            "{browser} data not found. Open {browser} once and sign in to YouTube, or use an exported file: "
            f"ytverbatim auth use cookie-file:PATH ({DOCS}#cookie-file)"
        ),
        AuthState.PROFILE_NOT_FOUND: (
            "{browser} profile '{profile}' has no cookie database. "
            "Run ytverbatim auth profiles to list them, then: ytverbatim auth use {browser}:<DIR>"
        ),
        AuthState.COOKIE_FILE_INVALID: (
            "The cookie file is missing or not a Netscape cookies.txt export. "
            f"Re-export it ({DOCS}#cookie-file) and run: ytverbatim auth use cookie-file:PATH"
        ),
        AuthState.KEYCHAIN_DENIED: (
            "macOS Keychain refused '{keychain_item}'. Run 'ytverbatim auth status --refresh' in Terminal "
            f"and click 'Always Allow' for 'security' ({DOCS}#keychain)."
        ),
        AuthState.LOGGED_OUT: (
            "{browser} profile '{profile}' is not signed in to YouTube. "
            "Open youtube.com in that profile, sign in, then re-run."
        ),
        AuthState.COOKIE_FILE_LOGGED_OUT: (
            f"The exported cookies are signed out or expired. Re-export from a fresh private window ({DOCS}"
            "#cookie-file) and run: ytverbatim auth use cookie-file:PATH"
        ),
        AuthState.ROTATED: (
            "YouTube rotated this session's cookies. With a browser profile, keep a youtube.com tab open in "
            "'{profile}' and re-run; if it repeats, sign out and back in there. With a cookie file, re-export "
            f"it from a fresh private window ({DOCS}#cookie-file)."
        ),
        AuthState.BOT_CHECK: (
            "YouTube flagged this connection as automated. Wait until {until}. Do not add cookies to get past "
            f"a bot check; disconnect any VPN or proxy ({DOCS}#account-safety)."
        ),
        AuthState.CAPTCHA: (
            "YouTube asked for a captcha, so it flagged this connection as automated. Wait until {until}. "
            f"Do not add cookies to get past it; disconnect any VPN or proxy ({DOCS}#account-safety)."
        ),
        AuthState.IP_BLOCKED: (
            "YouTube is likely blocking this IP address. Wait until {until}. Do not add cookies to get past "
            f"it; disconnect any VPN or proxy ({DOCS}#account-safety)."
        ),
        AuthState.RATE_LIMITED: (
            "YouTube rate-limited this session for up to an hour. Wait until {until}, then use a smaller --max-videos."
        ),
        AuthState.NOT_MEMBER: "Signed in, but this account is not a member of {channel} (or its tier is too low).",
        AuthState.AUTH_UNCERTAIN: (
            "Could not confirm that YouTube accepts this login. "
            "Run: ytverbatim auth status --refresh; if it fails again, run: ytverbatim auth doctor"
        ),
        AuthState.AUTH_REQUIRED: (
            "This video needs a signed-in account. Run: ytverbatim auth profiles, then ytverbatim auth use chrome:<DIR>"
        ),
        AuthState.JS_RUNTIME_MISSING: (
            "No JavaScript runtime for YouTube's challenges. Install Deno (brew install deno) or Node >= 22."
        ),
        AuthState.YTDLP_STALE: (
            "yt-dlp {version} is {days} days old and YouTube changes often. Upgrade yt-dlp in this environment."
        ),
        AuthState.PO_TOKEN_REQUIRED: (
            f"YouTube wants a PO token for some formats or subtitles. See {DOCS}#po-token (optional provider plugin)."
        ),
        AuthState.OK: "Signed in · Premium: {premium} · verified {verified}",
    }
)

_DEFAULTS: Mapping[str, str] = MappingProxyType(
    {
        "browser": "chrome",
        "profile": "<DIR>",
        "keychain_item": "Chrome Safe Storage",
        "until": "an hour has passed",
        "channel": "this channel",
        "premium": "unknown",
        "verified": "just now",
        "version": "?",
        "days": "?",
    }
)


class _WithDefaults(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return _DEFAULTS.get(key, f"<{key}>")


def fix_for(state: AuthState, **context: str | None) -> str:
    """Return the fix for ``state``; missing or ``None`` context falls back to a generic placeholder."""
    return FIXES[state].format_map(_WithDefaults({k: v for k, v in context.items() if v is not None}))
