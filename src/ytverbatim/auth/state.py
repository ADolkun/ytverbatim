from enum import StrEnum


class AuthState(StrEnum):
    NO_SOURCE = "no_source"
    BROWSER_NOT_FOUND = "browser_not_found"
    PROFILE_NOT_FOUND = "profile_not_found"
    COOKIE_FILE_INVALID = "cookie_file_invalid"
    KEYCHAIN_DENIED = "keychain_denied"
    LOGGED_OUT = "logged_out"
    ROTATED = "rotated"
    BOT_CHECK = "bot_check"
    CAPTCHA = "captcha"
    IP_BLOCKED = "ip_blocked"
    RATE_LIMITED = "rate_limited"
    NOT_MEMBER = "not_member"
    AUTH_UNCERTAIN = "auth_uncertain"
    AUTH_REQUIRED = "auth_required"
    JS_RUNTIME_MISSING = "js_runtime_missing"
    YTDLP_STALE = "ytdlp_stale"
    PO_TOKEN_REQUIRED = "po_token_required"
    OK = "ok"
