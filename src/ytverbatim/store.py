"""Per-video stage store: one JSON checkpoint per pipeline stage, keyed by a hash of everything the stage read.

`.stages/<stage>.json` holds `{stage_version, input_hash, output}` and is replaced atomically, so a crash mid-stage
leaves the previous checkpoint or none, never a partial one. A stage's input hash folds in the digests of the upstream
outputs it consumed and its own options; an option change therefore misses only the stage that reads it, and a stage
downstream misses only if that stage's output actually changed. `.stages/<stage>/` is the stage's own working
directory: the ASR worker's resumable output and the LLM stages' per-batch checkpoints.
"""

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable, Generator, Mapping
from pathlib import Path

from pydantic import BaseModel, JsonValue, ValidationError

from ytverbatim import atomic
from ytverbatim.errors import BusyError

log = logging.getLogger(__name__)

STAGES_DIR = ".stages"
LOCK_NAME = ".lock"


def digest(value: JsonValue | BaseModel) -> str:
    """sha256 of the canonical JSON form; key order and whitespace never change it."""
    plain = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    canonical = json.dumps(plain, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def input_hash(stage: str, version: int, inputs: Mapping[str, JsonValue]) -> str:
    return digest({"stage": stage, "version": version, "inputs": dict(inputs)})


class StageStore:
    def __init__(self, video_dir: Path) -> None:
        self.root = video_dir / STAGES_DIR

    def path(self, stage: str) -> Path:
        return self.root / f"{stage}.json"

    def work_dir(self, stage: str) -> Path:
        return self.root / stage

    def load[M: BaseModel](self, stage: str, version: int, key: str, model: type[M]) -> M | None:
        """Return the stored output if this stage version made it from these inputs, else None."""
        path = self.path(stage)
        try:
            raw: object = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            log.warning("ignoring an unreadable stage checkpoint %s: %s", path.name, exc)
            return None
        if not isinstance(raw, dict):
            return None
        if raw.get("stage_version") != version or raw.get("input_hash") != key:
            return None
        try:
            return model.model_validate(raw.get("output"))
        except ValidationError as exc:
            # Written only by this module, atomically: an output that no longer validates predates a model change.
            log.warning("ignoring a stage checkpoint %s that no longer validates: %s", path.name, exc)
            return None

    def save(self, stage: str, version: int, key: str, output: BaseModel) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {"stage_version": version, "input_hash": key, "output": output.model_dump(mode="json")}
        atomic.write_json(self.path(stage), record)

    def run[M: BaseModel](
        self, stage: str, version: int, key: str, model: type[M], compute: Callable[[], M]
    ) -> tuple[M, bool]:
        """Return the stored output, or compute and store it; the flag says whether it was reused."""
        stored = self.load(stage, version, key, model)
        if stored is not None:
            return stored, True
        output = compute()
        self.save(stage, version, key, output)
        return output, False

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@contextlib.contextmanager
def lock(video_dir: Path) -> Generator[None]:
    """Hold the video directory exclusively, or raise `BusyError` at once; waiting could mean hours of ASR."""
    video_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(video_dir / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BusyError(f"another ytverbatim run is processing {video_dir.name}") from None
        yield
    finally:
        os.close(fd)
