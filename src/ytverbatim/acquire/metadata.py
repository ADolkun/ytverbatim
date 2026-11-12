import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from ytverbatim import atomic
from ytverbatim.errors import UnavailableError, YtDlpError, stderr_preview
from ytverbatim.ids import video_url
from ytverbatim.ytdlp.types import YtDlpRunner

INFO_FILENAME = "info.json"
DEFAULT_TIMEOUT_S = 120.0

_AUTO_ORIGINAL_SUFFIX = "-orig"
_LIVE_KINDS = {"is_live": "live", "is_upcoming": "upcoming"}


@dataclass(frozen=True)
class Chapter:
    title: str
    start: float
    end: float


@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    title: str
    channel: str
    channel_id: str | None
    upload_date: str | None
    duration: float | None
    description: str
    language: str | None
    availability: str | None
    live_status: str | None
    chapters: tuple[Chapter, ...]
    subtitle_langs: tuple[str, ...]
    # Only the original-language auto track: the others are machine translations of it.
    auto_caption_lang: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def info_path(stage_dir: Path) -> Path:
    return stage_dir / INFO_FILENAME


def fetch_info(
    runner: YtDlpRunner,
    video_id: str,
    stage_dir: Path,
    *,
    cookies: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> VideoMetadata:
    """Run the video's single extraction and keep the raw info JSON for `--load-info-json` reuse."""
    result = runner.run(
        ["-J", "--no-playlist", video_url(video_id)], cookies=cookies, timeout_s=timeout_s, label="metadata"
    )
    info = _parse_info(result.stdout, result.returncode)
    if info.get("id") != video_id:
        raise YtDlpError(None, (), f"yt-dlp returned info for a different video than {video_id}", result.returncode)
    metadata = trim_info(info)
    if metadata.live_status in _LIVE_KINDS:
        raise UnavailableError(_LIVE_KINDS[metadata.live_status], "live and upcoming streams have no final transcript")
    stage_dir.mkdir(parents=True, exist_ok=True)
    # 0600: the info JSON carries signed, session-bound format URLs.
    atomic.write_text(info_path(stage_dir), result.stdout, mode=0o600)
    return metadata


def load_metadata(stage_dir: Path) -> VideoMetadata:
    path = info_path(stage_dir)
    return trim_info(_parse_info(path.read_text(encoding="utf-8"), 0))


def trim_info(info: Mapping[str, object]) -> VideoMetadata:
    language = _text(info.get("language"))
    subtitles = _mapping(info.get("subtitles"))
    live_status = _text(info.get("live_status"))
    if live_status is None and info.get("is_live"):
        live_status = "is_live"
    return VideoMetadata(
        video_id=_text(info.get("id")) or "",
        title=_text(info.get("title")) or "",
        channel=_text(info.get("channel")) or _text(info.get("uploader")) or "",
        channel_id=_text(info.get("channel_id")),
        upload_date=_text(info.get("upload_date")),
        duration=_number(info.get("duration")),
        description=_text(info.get("description")) or "",
        language=language,
        availability=_text(info.get("availability")),
        live_status=live_status,
        chapters=_chapters(info.get("chapters")),
        subtitle_langs=tuple(sorted(subtitles)),
        auto_caption_lang=_auto_original_lang(_mapping(info.get("automatic_captions")), language),
    )


def _auto_original_lang(auto_captions: Mapping[str, object], language: str | None) -> str | None:
    originals = sorted(key for key in auto_captions if key.endswith(_AUTO_ORIGINAL_SUFFIX))
    if originals:
        if language is not None:
            base = language.split("-")[0].lower()
            for key in originals:
                if key.split("-")[0].lower() == base:
                    return key
        return originals[0]
    if language is not None and language in auto_captions:
        return language
    return None


def _chapters(raw: object) -> tuple[Chapter, ...]:
    if not isinstance(raw, list):
        return ()
    chapters: list[Chapter] = []
    for item in cast("list[object]", raw):
        entry = _mapping(item)
        start = _number(entry.get("start_time"))
        end = _number(entry.get("end_time"))
        if start is None or end is None:
            continue
        chapters.append(Chapter(title=_text(entry.get("title")) or "", start=start, end=end))
    return tuple(chapters)


def _parse_info(text: str, returncode: int) -> dict[str, object]:
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise YtDlpError(None, (), f"yt-dlp output is not JSON: {stderr_preview(text)}", returncode) from exc
    if not isinstance(parsed, dict):
        raise YtDlpError(None, (), "yt-dlp output is not a JSON object", returncode)
    return cast("dict[str, object]", parsed)


def _mapping(value: object) -> dict[str, object]:
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
