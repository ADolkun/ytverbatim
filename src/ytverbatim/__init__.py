from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ytverbatim.acquire.session import expand
    from ytverbatim.acquire.video import video_file
    from ytverbatim.config import Options
    from ytverbatim.contract import ContextDocument
    from ytverbatim.llm import LLMClient
    from ytverbatim.pipeline import process

__version__ = version("ytverbatim")

# The library surface is imported on first use so `ytverbatim --version` and the lighter commands never pay for
# yt-dlp, pydantic and the pipeline.
_LAZY = {
    "process": "ytverbatim.pipeline",
    "expand": "ytverbatim.acquire.session",
    "video_file": "ytverbatim.acquire.video",
    "Options": "ytverbatim.config",
    "ContextDocument": "ytverbatim.contract",
    "LLMClient": "ytverbatim.llm",
}

__all__ = ["ContextDocument", "LLMClient", "Options", "__version__", "expand", "process", "video_file"]


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    # Cached so the next lookup is an ordinary attribute and never reaches __getattr__ again.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
