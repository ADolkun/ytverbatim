import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evals.normalize import Language
from ytverbatim.errors import ConfigError

EVAL_DIR_ENV = "YTVERBATIM_EVAL_DIR"
MANIFEST_NAME = "manifest.json"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class ListedCaption(_Frozen):
    kind: Literal["manual", "auto"]
    lang: str
    formats: tuple[str, ...]


class ObtainedCaption(_Frozen):
    kind: Literal["manual", "auto"]
    lang: str
    format: str
    path: str
    bytes: int
    word_timing: bool | None = None


class Captions(_Frozen):
    listed: tuple[ListedCaption, ...]
    listed_note: str
    obtained: tuple[ObtainedCaption, ...]


class Video(_Frozen):
    id: str
    url: str
    channel: str
    title: str
    duration_s: int
    language: Literal["zh", "en", "zh+en"]
    access: Literal["public", "members"]
    audio: str
    audio_sha256: str
    audio_format: str
    audio_duration_s: float
    upload_date: str
    chapters: str
    chapter_count: int
    captions: Captions
    source: str
    notes: str
    meta: str

    @property
    def view_language(self) -> Language:
        """The scoring view: mixed Chinese-and-English speech is scored as Chinese."""
        return "en" if self.language == "en" else "zh"


class Manifest(_Frozen):
    created: str
    audio_format: str
    videos: tuple[Video, ...]

    def labels(self) -> dict[str, str]:
        """Return generic per-file labels (`zh-01`, `en-01`) that are safe to publish."""
        seen: dict[Language, int] = {"zh": 0, "en": 0}
        labels: dict[str, str] = {}
        for video in self.videos:
            seen[video.view_language] += 1
            labels[video.id] = f"{video.view_language}-{seen[video.view_language]:02d}"
        return labels


def eval_dir() -> Path:
    value = os.environ.get(EVAL_DIR_ENV)
    if not value:
        raise ConfigError(f"set {EVAL_DIR_ENV} to the directory that holds {MANIFEST_NAME} and the eval audio")
    return Path(value).expanduser()


def load_manifest(directory: Path | None = None) -> Manifest:
    path = (directory or eval_dir()) / MANIFEST_NAME
    manifest = Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    ids = [v.id for v in manifest.videos]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"{path} lists a video id more than once")
    return manifest
