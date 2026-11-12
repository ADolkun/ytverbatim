import os
from pathlib import Path

from ytverbatim.ids import is_video_id

_APP = "ytverbatim"
_PRIVATE_MODE = 0o700


def _xdg(variable: str, fallback: Path) -> Path:
    value = os.environ.get(variable)
    # The XDG spec says relative values are invalid and must be ignored.
    if value and Path(value).is_absolute():
        return Path(value)
    return fallback


def _private(path: Path) -> Path:
    path.mkdir(mode=_PRIVATE_MODE, parents=True, exist_ok=True)
    if path.stat().st_mode & 0o777 != _PRIVATE_MODE:
        path.chmod(_PRIVATE_MODE)
    return path


def config_file() -> Path:
    return _xdg("XDG_CONFIG_HOME", Path.home() / ".config") / _APP / "config.toml"


def state_dir() -> Path:
    return _private(_xdg("XDG_STATE_HOME", Path.home() / ".local" / "state") / _APP)


def auth_dir() -> Path:
    return _private(state_dir() / "auth")


def locks_dir() -> Path:
    return _private(state_dir() / "locks")


def cache_dir() -> Path:
    return _private(Path.home() / "Library" / "Caches" / _APP)


def pins_dir() -> Path:
    return _private(cache_dir() / "pins")


def llm_dir(provider: str) -> Path:
    return _private(_private(cache_dir() / "llm") / provider)


def home_dir() -> Path:
    value = os.environ.get("YTVERBATIM_HOME")
    return Path(value).expanduser() if value else Path.home() / ".local" / "share" / _APP


def default_output_root() -> Path:
    return home_dir() / "videos"


def video_dir(video_id: str, out: Path | None = None) -> Path:
    if not is_video_id(video_id):
        raise ValueError(f"not a YouTube video id: {video_id!r}")
    return (out if out is not None else default_output_root()) / video_id
