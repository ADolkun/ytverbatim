from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ytverbatim.config import Options
    from ytverbatim.contract import ContextDocument
    from ytverbatim.pipeline import process

__version__ = version("ytverbatim")

# The library surface is imported on first use so `ytverbatim --version` and the lighter commands never pay for
# yt-dlp, pydantic and the pipeline.
_LAZY = {"process": "ytverbatim.pipeline", "Options": "ytverbatim.config", "ContextDocument": "ytverbatim.contract"}

__all__ = ["ContextDocument", "Options", "__version__", "process"]


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)
