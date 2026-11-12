"""Local LLM server for the correction eval: ``python -m evals.mlx_server --memory-cap-bytes N -- SERVER_ARGS``.

Start it with the venv's Python itself, under `proc.background`. This process is the server, so the parent's memory
watch reads the footprint of the process that holds the model; `uv run` or `uvx` would put a parent of their own in
between, and the watch would read that instead.
"""

import argparse
import importlib
import os
import sys
import threading
import time
from collections.abc import Sequence
from typing import Protocol, cast

from ytverbatim.asr.worker import MlxAllocator, bound_allocator
from ytverbatim.errors import DependencyError

# Unlike the ASR worker's zero, a server reuses buffers of the same shapes on every request; 1 GiB keeps that reuse
# while adding at most its own size to the footprint.
CACHE_LIMIT_BYTES = 1 << 30
_INSTALL = "the local LLM server needs mlx-lm; run `uv sync --group local-llm`"
PARENT_POLL_S = 1.0
EXIT_ORPHANED = 3


class _Server(Protocol):
    def main(self) -> None: ...


def main(argv: Sequence[str] | None = None) -> None:
    _exit_with_parent()
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m evals.mlx_server", usage="%(prog)s --memory-cap-bytes N -- ...")
    parser.add_argument("--memory-cap-bytes", type=int, required=True)
    if "--" not in args:
        parser.error("put the mlx_lm.server arguments after --")
    split = args.index("--")
    cap = parser.parse_args(args[:split]).memory_cap_bytes
    if cap <= 0:
        parser.error("--memory-cap-bytes must be positive")
    bound_allocator(_import("mlx.core", MlxAllocator), cap, cache_limit_bytes=CACHE_LIMIT_BYTES)
    server = _import("mlx_lm.server", _Server)
    sys.argv = ["mlx_lm.server", *args[split + 1 :]]
    server.main()


def _exit_with_parent() -> None:
    """Exit as soon as the parent is gone.

    The parent's memory watch and the worker lock it shares with this process both end with it, so a server that
    outlived a parent killed with SIGKILL would hold gigabytes unwatched while another MLX worker started beside it.
    """

    def watch() -> None:
        # macOS has no subreapers: an orphan is reparented to launchd, pid 1.
        while os.getppid() != 1:
            time.sleep(PARENT_POLL_S)
        # The main thread is blocked serving, so only a hard exit ends the process from here.
        os._exit(EXIT_ORPHANED)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


def _import[T](name: str, kind: type[T]) -> T:
    try:
        return cast(T, importlib.import_module(name))
    except ModuleNotFoundError as exc:
        # Only the package itself missing means the group is not installed; a missing dependency of it is a broken
        # install, whose own message says what is wrong.
        if exc.name is None or not (name == exc.name or name.startswith(f"{exc.name}.")):
            raise
        raise DependencyError(_INSTALL) from exc


if __name__ == "__main__":
    main()
