import tomllib
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from ytverbatim import paths
from ytverbatim.errors import ConfigError


@dataclass(frozen=True)
class AuthOptions:
    policy: Literal["auto", "always", "never"] = "auto"
    cookie_channels: tuple[str, ...] = ()


@dataclass(frozen=True)
class YtdlpOptions:
    max_videos_per_run: int = 50
    premium_probe_id: str = "BaW_jenozKc"
    creator_note: bool = False


@dataclass(frozen=True)
class AsrOptions:
    engine: str = "w-prod"
    source: Literal["asr", "captions"] = "asr"


@dataclass(frozen=True)
class TextOptions:
    correct: bool = True
    chapters: Literal["youtube", "none"] = "youtube"
    numeric_second_opinion: str = "off"


@dataclass(frozen=True)
class LLMOptions:
    provider: Literal["none", "claude-cli", "openai-compat"] = "none"
    model: str | None = None
    effort: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 900.0


@dataclass(frozen=True)
class Options:
    auth: AuthOptions = field(default_factory=AuthOptions)
    ytdlp: YtdlpOptions = field(default_factory=YtdlpOptions)
    asr: AsrOptions = field(default_factory=AsrOptions)
    text: TextOptions = field(default_factory=TextOptions)
    llm: LLMOptions = field(default_factory=LLMOptions)


Overrides = Mapping[str, Mapping[str, object]]


def load_options(path: Path | None = None, overrides: Overrides | None = None) -> Options:
    """Read the config file (absent means defaults); `overrides` win, and None values mean "not given"."""
    raw = _read(path if path is not None else paths.config_file())
    given = overrides or {}
    sections = {"auth": AuthOptions, "ytdlp": YtdlpOptions, "asr": AsrOptions, "text": TextOptions, "llm": LLMOptions}
    unknown = (set(raw) | set(given)) - set(sections)
    if unknown:
        raise ConfigError(f"unknown config section(s): {', '.join(sorted(unknown))}")

    def merged(name: str) -> dict[str, object]:
        section = raw.get(name, {})
        if not isinstance(section, dict):
            raise ConfigError(f"config section [{name}] must be a table")
        values: dict[str, object] = dict(section)
        values.update({k: v for k, v in given.get(name, {}).items() if v is not None})
        return values

    return Options(
        auth=AuthOptions(**_checked(AuthOptions, "auth", merged("auth"))),
        ytdlp=YtdlpOptions(**_checked(YtdlpOptions, "ytdlp", merged("ytdlp"))),
        asr=AsrOptions(**_checked(AsrOptions, "asr", merged("asr"))),
        text=TextOptions(**_checked(TextOptions, "text", merged("text"))),
        llm=LLMOptions(**_checked(LLMOptions, "llm", merged("llm"))),
    )


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc


def _checked(cls: type, section: str, values: Mapping[str, object]) -> dict[str, Any]:
    hints = get_type_hints(cls)
    unknown = set(values) - set(hints)
    if unknown:
        raise ConfigError(f"unknown option(s) in [{section}]: {', '.join(sorted(unknown))}")
    return {key: _coerce(f"{section}.{key}", value, hints[key]) for key, value in values.items()}


def _coerce(name: str, value: object, hint: object) -> Any:
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        (inner,) = (arg for arg in get_args(hint) if arg is not type(None))
        return _coerce(name, value, inner)
    if origin is Literal:
        if value in get_args(hint):
            return value
        raise ConfigError(f"{name} must be one of: {', '.join(map(str, get_args(hint)))}")
    if origin is tuple:
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return tuple(value)
        raise ConfigError(f"{name} must be a list of strings")
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(hint, type) and type(value) is hint:
        return value
    expected = hint.__name__ if isinstance(hint, type) else str(hint)
    raise ConfigError(f"{name} must be of type {expected}")
