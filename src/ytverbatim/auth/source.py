import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ytverbatim import atomic, paths
from ytverbatim.auth import chromium
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError

ENV_VAR = "YTVERBATIM_AUTH"
_COOKIE_FILE_PREFIX = "cookie-file:"
_NONE = "none"
_SOURCE_FILE_VERSION = 1


@dataclass(frozen=True)
class BrowserSource:
    browser: str
    profile: str | None = None


@dataclass(frozen=True)
class CookieFileSource:
    path: Path


type CookieSource = BrowserSource | CookieFileSource


def parse_spec(spec: str) -> CookieSource | None:
    """Parse ``chrome[:PROFILE]`` (or another supported Chromium browser), ``cookie-file:PATH`` or ``none``."""
    spec = spec.strip()
    if spec == _NONE:
        return None
    if spec.startswith(_COOKIE_FILE_PREFIX):
        raw = spec.removeprefix(_COOKIE_FILE_PREFIX)
        if not raw:
            raise AuthError(AuthState.COOKIE_FILE_INVALID, "cookie-file: needs a path, e.g. cookie-file:~/cookies.txt")
        return CookieFileSource(Path(raw).expanduser().absolute())
    browser, _, profile = spec.partition(":")
    if browser not in chromium.SUPPORTED_BROWSERS:
        choices = ", ".join(f"{name}[:PROFILE]" for name in chromium.SUPPORTED_BROWSERS)
        raise AuthError(
            AuthState.BROWSER_NOT_FOUND,
            f"unknown auth source {spec!r}; use {choices}, {_COOKIE_FILE_PREFIX}PATH or {_NONE}",
        )
    return BrowserSource(browser, profile or None)


def format_spec(source: CookieSource | None) -> str:
    match source:
        case None:
            return _NONE
        case BrowserSource(browser=browser, profile=None):
            return browser
        case BrowserSource(browser=browser, profile=profile):
            return f"{browser}:{profile}"
        case CookieFileSource(path=path):
            return f"{_COOKIE_FILE_PREFIX}{path}"


def _saved_spec(path: Path) -> str | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise AuthError(AuthState.NO_SOURCE, f"{path} is unreadable ({exc}); rerun `ytverbatim auth use`") from exc
    spec = raw.get("spec") if isinstance(raw, dict) else None
    if not isinstance(spec, str):
        raise AuthError(AuthState.NO_SOURCE, f"{path} has no 'spec' string; rerun `ytverbatim auth use`")
    return spec


def resolve_source(cli: str | None = None, *, env: Mapping[str, str] | None = None) -> CookieSource | None:
    """Pick the auth source: ``--auth`` > ``$YTVERBATIM_AUTH`` > saved ``auth/source.json``.

    ``None`` means anonymous, either because nothing is configured or because the winning spec is
    ``none``. A misconfigured spec raises instead of falling back to anonymous.
    """
    environ = os.environ if env is None else env
    for spec in (cli, environ.get(ENV_VAR)):
        if spec is not None and spec.strip():
            return parse_spec(spec)
    saved = _saved_spec(paths.auth_source_file())
    return parse_spec(saved) if saved is not None else None


def save_source(source: CookieSource | None) -> Path:
    path = paths.auth_source_file()
    atomic.write_json(path, {"version": _SOURCE_FILE_VERSION, "spec": format_spec(source)}, mode=0o600)
    return path
