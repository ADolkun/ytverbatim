import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from ytverbatim import paths
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError


@dataclass(frozen=True)
class _Browser:
    root: str
    keyring: str


# Data roots under ~/Library/Application Support (macOS is the only supported host) and the
# Keychain item each browser encrypts its cookies with.
_BROWSERS = {
    "brave": _Browser("BraveSoftware/Brave-Browser", "Brave"),
    "chrome": _Browser("Google/Chrome", "Chrome"),
    "chromium": _Browser("Chromium", "Chromium"),
    "edge": _Browser("Microsoft Edge", "Microsoft Edge"),
    "vivaldi": _Browser("Vivaldi", "Vivaldi"),
}

SUPPORTED_BROWSERS: tuple[str, ...] = tuple(sorted(_BROWSERS))

type EmailMode = Literal["omit", "masked", "full"]


@dataclass(frozen=True)
class Profile:
    dir: str
    name: str
    active_time: datetime | None
    last_used: bool
    cookie_db: Path | None
    email: str | None = None


def _browser(browser: str) -> _Browser:
    try:
        return _BROWSERS[browser]
    except KeyError:
        raise AuthError(
            AuthState.BROWSER_NOT_FOUND,
            f"unsupported browser {browser!r}; use one of {', '.join(SUPPORTED_BROWSERS)}",
        ) from None


def keychain_item(browser: str) -> str:
    return f"{_browser(browser).keyring} Safe Storage"


def browser_root(browser: str) -> Path:
    return Path.home() / "Library" / "Application Support" / _browser(browser).root


def _existing_root(browser: str) -> Path:
    root = browser_root(browser)
    if not root.is_dir():
        raise AuthError(
            AuthState.BROWSER_NOT_FOUND,
            f"no {browser} data directory found; install {browser} and sign in to YouTube once",
        )
    return root


def profile_cookie_store(profile_dir: Path) -> Path | None:
    for candidate in (profile_dir / "Network" / "Cookies", profile_dir / "Cookies"):
        if candidate.is_file():
            return candidate
    return None


def _profile_stores(root: Path) -> dict[str, Path]:
    return {
        child.name: store
        for child in sorted(root.iterdir())
        if child.is_dir() and (store := profile_cookie_store(child)) is not None
    }


def cookie_db(browser: str, profile: str | None) -> Path:
    """Resolve a profile's own cookie database.

    Without a profile this follows yt-dlp's "most recently used profile" meaning, but compares
    profile stores only, never extension stores.
    """
    root = _existing_root(browser)
    stores = _profile_stores(root)
    if profile is None:
        store = max(stores.values(), key=lambda s: s.stat().st_mtime, default=None)
    else:
        store = stores.get(profile)
    if store is None:
        wanted = f"profile {profile!r}" if profile is not None else "any profile"
        available = ", ".join(repr(name) for name in stores) or "none"
        raise AuthError(
            AuthState.PROFILE_NOT_FOUND,
            f"no {browser} cookie database for {wanted}; profiles with cookies: {available}",
        )
    return store


def pin_cookie_db(store: Path) -> Path:
    """Return a private directory whose only entry is a ``Cookies`` symlink to ``store``.

    yt-dlp reads the newest file named ``Cookies`` anywhere under a profile, and Chrome extension
    storage (``Storage/ext/<id>/Cookies``) is often newer than the real store, which silently drops
    the YouTube login. Passing ``<profile>/Network`` as the path does not work either.
    """
    resolved = store.resolve()
    pin_dir = paths.pins_dir() / hashlib.sha256(str(resolved).encode()).hexdigest()[:16]
    pin_dir.mkdir(mode=0o700, exist_ok=True)
    link = pin_dir / "Cookies"
    if not (link.is_symlink() and link.readlink() == resolved):
        # Staged then renamed so a concurrent reader never sees a missing or half-made link.
        staged = pin_dir / f".Cookies.{os.getpid()}.{secrets.token_hex(4)}"
        staged.symlink_to(resolved)
        try:
            staged.replace(link)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
    return pin_dir


def mask_email(email: str) -> str:
    local, at, domain = email.partition("@")
    if not at:
        return "***"
    return f"{local[:1]}***@{domain}"


def _active_time(value: object) -> datetime | None:
    if isinstance(value, int | float) and value > 0:
        return datetime.fromtimestamp(value, UTC)
    return None


def list_profiles(browser: str, *, emails: EmailMode = "omit") -> list[Profile]:
    """List profiles from the browser's ``Local State``; emails only when asked, masked unless ``full``."""
    root = _existing_root(browser)
    try:
        state = json.loads((root / "Local State").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuthError(AuthState.BROWSER_NOT_FOUND, f"cannot read {browser} Local State: {exc}") from exc
    section = state.get("profile") if isinstance(state, dict) else None
    info_cache = section.get("info_cache") if isinstance(section, dict) else None
    if not isinstance(section, dict) or not isinstance(info_cache, dict):
        return []
    last_used = section.get("last_used")
    profiles: list[Profile] = []
    for directory, entry in sorted(info_cache.items()):
        if not isinstance(directory, str) or not isinstance(entry, dict):
            continue
        email = entry.get("user_name")
        shown: str | None = None
        if isinstance(email, str) and email and emails != "omit":
            shown = email if emails == "full" else mask_email(email)
        name = entry.get("name")
        profiles.append(
            Profile(
                dir=directory,
                name=name if isinstance(name, str) else directory,
                active_time=_active_time(entry.get("active_time")),
                last_used=directory == last_used,
                cookie_db=profile_cookie_store(root / directory),
                email=shown,
            )
        )
    return profiles
