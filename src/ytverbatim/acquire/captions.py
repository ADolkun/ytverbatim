import html
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from ytverbatim.acquire.metadata import VideoMetadata
from ytverbatim.errors import YtDlpError
from ytverbatim.ytdlp.types import Signal, YtDlpRunner

log = logging.getLogger(__name__)

DEFAULT_LANGUAGES = ("en", "zh")
DEFAULT_TIMEOUT_S = 300.0
FILE_STEM = "captions"

_LANG_VARIANTS: dict[str, tuple[str, ...]] = {
    "en": ("en", "en-US", "en-GB"),
    "zh": ("zh", "zh-Hans", "zh-CN", "zh-Hant", "zh-TW"),
    "zh-Hans": ("zh-Hans", "zh-CN", "zh"),
    "zh-Hant": ("zh-Hant", "zh-TW", "zh"),
    "zh-CN": ("zh-CN", "zh-Hans", "zh"),
    "zh-TW": ("zh-TW", "zh-Hant", "zh"),
}

# Auto captions should be dense; manual ones can be sparser but must still carry real content.
_MIN_AUTO_SEGMENTS_PER_MIN = 4.0
_MIN_SEGMENTS = 1
_MIN_CHARS_PER_MIN = 20.0

# Rolling auto-caption cues touch each other; the slack absorbs millisecond rounding.
_ADJACENT_CUE_SLACK_S = 0.05

_TIMESTAMP = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")
_FORMAT_PREFERENCE = ("json3", "vtt")


class CaptionKind(StrEnum):
    MANUAL = "manual"
    AUTO = "auto"


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class CaptionSegment:
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class CaptionTrack:
    language: str
    kind: CaptionKind
    source_format: str
    segments: tuple[CaptionSegment, ...]

    @property
    def duration(self) -> float:
        return max((segment.end for segment in self.segments), default=0.0)

    @property
    def has_word_timings(self) -> bool:
        return any(segment.words for segment in self.segments)


@dataclass(frozen=True)
class CaptionQuality:
    segments_per_min: float
    chars_per_min: float
    avg_chars: float
    coverage: float | None
    is_auto: bool
    ok: bool


@dataclass(frozen=True)
class CaptionsResult:
    tracks: tuple[CaptionTrack, ...]
    warnings: tuple[str, ...]


def expand_language_priority(languages: Sequence[str]) -> tuple[str, ...]:
    expanded: dict[str, None] = {}
    for language in languages:
        for variant in _LANG_VARIANTS.get(language, (language,)):
            expanded.setdefault(variant)
    return tuple(expanded)


def select_caption_langs(metadata: VideoMetadata, languages: Sequence[str] = DEFAULT_LANGUAGES) -> tuple[str, ...]:
    """Pick the track keys worth a subtitle request: best manual variant per language, plus the original auto track."""
    wanted: dict[str, None] = {}
    available = set(metadata.subtitle_langs)
    for language in languages:
        manual = next((code for code in expand_language_priority((language,)) if code in available), None)
        if manual is not None:
            wanted.setdefault(manual)
    auto = metadata.auto_caption_lang
    if auto is not None and _base(auto) in {_base(language) for language in languages}:
        wanted.setdefault(auto)
    return tuple(wanted)


def fetch_captions(
    runner: YtDlpRunner,
    info_path: Path,
    captions_dir: Path,
    metadata: VideoMetadata,
    *,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
    cookies: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> CaptionsResult:
    """Download and parse the wanted caption tracks in one yt-dlp call, reusing the saved info JSON."""
    codes = select_caption_langs(metadata, languages)
    if not codes:
        return CaptionsResult((), ())
    captions_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "--load-info-json",
        str(info_path),
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        ",".join(codes),
        "--sub-format",
        "json3/vtt/best",
        "-o",
        f"subtitle:{captions_dir / FILE_STEM}.%(ext)s",
    ]
    try:
        runner.run(args, cookies=cookies, timeout_s=timeout_s, label="captions")
    except YtDlpError as exc:
        # A bare 429 on the subtitle download only costs this video its captions; the playability rate-limit
        # text (RATE_LIMITED) and everything else must reach the caller.
        if exc.primary is not Signal.HTTP_429:
            raise
        return _dropped(captions_dir)

    tracks: list[CaptionTrack] = []
    warnings: list[str] = []
    manual = set(metadata.subtitle_langs)
    for code in codes:
        track, warning = load_track(captions_dir, code, CaptionKind.MANUAL if code in manual else CaptionKind.AUTO)
        if track is not None:
            tracks.append(track)
        if warning is not None:
            log.warning(warning)
            warnings.append(warning)
    return CaptionsResult(tuple(tracks), tuple(warnings))


def parse_json3(content: str, *, kind: CaptionKind) -> tuple[CaptionSegment, ...]:
    """Parse YouTube json3; only auto tracks carry per-word timings, manual ones have segment timings."""
    try:
        parsed: object = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"caption file is not valid json3: {exc}") from exc
    events = cast("dict[str, object]", parsed).get("events") if isinstance(parsed, dict) else None
    if not isinstance(events, list):
        raise ValueError("json3 has no events list")
    # Newline-only aAppend events stay in the list: they carry no speech but mark where the previous line ends.
    entries = _json3_entries(cast("list[object]", events))
    segments: list[CaptionSegment] = []
    cursor = 0.0
    for index, (start_ms, duration_ms, parts) in enumerate(entries):
        start = start_ms / 1000
        end = (start_ms + duration_ms) / 1000
        text = _clean("".join(part for part, _ in parts))
        if not text:
            continue
        if kind is CaptionKind.MANUAL:
            segments.append(CaptionSegment(start=start, end=max(end, start), text=text))
            continue
        # Auto display windows outlive the words in them; the next line's start is where this one really ends.
        if index + 1 < len(entries):
            end = min(end, entries[index + 1][0] / 1000)
        words, cursor = _auto_words(start_ms, parts, end, cursor)
        if not words:
            continue
        segments.append(CaptionSegment(start=words[0].start, end=words[-1].end, text=text, words=words))
    return tuple(segments)


def parse_vtt(content: str, *, dedupe_rolling: bool = True) -> tuple[CaptionSegment, ...]:
    """Parse WebVTT; `dedupe_rolling` collapses the scrolling-window repeats of auto captions."""
    segments: list[CaptionSegment] = []
    last_line: str | None = None
    previous_end = float("-inf")
    for start, end, lines in _cue_blocks(content):
        fresh: list[str] = []
        # A line repeated by a cue that starts after the previous one ended is speech, not scrolling.
        adjacent = start <= previous_end + _ADJACENT_CUE_SLACK_S
        for line in lines:
            if dedupe_rolling and adjacent and line == last_line:
                continue
            fresh.append(line)
            last_line = line
        previous_end = end
        if fresh:
            segments.append(CaptionSegment(start=round(start, 3), end=round(end, 3), text=" ".join(fresh)))
    return tuple(segments)


def caption_quality(
    track: CaptionTrack,
    *,
    video_duration: float | None = None,
    min_coverage: float = 0.5,
) -> CaptionQuality:
    """Whether a track is good enough to stand in for ASR; coverage is only gated when the true duration is known."""
    non_empty = [segment for segment in track.segments if segment.text.strip()]
    minutes = track.duration / 60.0
    segments_per_min = len(non_empty) / minutes if minutes > 0 else 0.0
    total_chars = sum(len(segment.text.strip()) for segment in non_empty)
    chars_per_min = total_chars / minutes if minutes > 0 else 0.0
    avg_chars = total_chars / len(non_empty) if non_empty else 0.0
    is_auto = track.kind is CaptionKind.AUTO
    coverage = track.duration / video_duration if video_duration and video_duration > 0 else None

    degenerate = len(non_empty) < _MIN_SEGMENTS or minutes <= 0 or chars_per_min < _MIN_CHARS_PER_MIN
    ok = not (
        degenerate
        or (is_auto and segments_per_min < _MIN_AUTO_SEGMENTS_PER_MIN)
        or (coverage is not None and coverage < min_coverage)
    )
    return CaptionQuality(
        segments_per_min=round(segments_per_min, 2),
        chars_per_min=round(chars_per_min, 2),
        avg_chars=round(avg_chars, 2),
        coverage=round(coverage, 3) if coverage is not None else None,
        is_auto=is_auto,
        ok=ok,
    )


def _base(code: str) -> str:
    return code.split("-")[0].lower()


def _dropped(captions_dir: Path) -> CaptionsResult:
    # Keeping a partial set could silently swap a manual track for a worse auto one.
    for leftover in captions_dir.glob(f"{FILE_STEM}.*"):
        leftover.unlink(missing_ok=True)
    warning = f"captions skipped for this video: subtitle download hit {Signal.HTTP_429.value}"
    log.warning(warning)
    return CaptionsResult((), (warning,))


def load_track(captions_dir: Path, code: str, kind: CaptionKind) -> tuple[CaptionTrack | None, str | None]:
    files = {
        path.suffix.removeprefix(".").lower(): path
        for path in captions_dir.glob(f"{FILE_STEM}.{code}.*")
        if path.is_file()
    }
    source_format = next((fmt for fmt in _FORMAT_PREFERENCE if fmt in files), None)
    if source_format is None:
        found = ", ".join(sorted(files)) or "none"
        return None, f"no json3 or vtt caption file for '{code}' (found: {found})"
    content = files[source_format].read_text(encoding="utf-8", errors="replace")
    try:
        if source_format == "json3":
            segments = parse_json3(content, kind=kind)
        else:
            segments = parse_vtt(content, dedupe_rolling=kind is CaptionKind.AUTO)
    except ValueError as exc:
        return None, f"caption file for '{code}' is unreadable: {exc}"
    if not segments:
        return None, f"caption track '{code}' has no text"
    return CaptionTrack(language=code, kind=kind, source_format=source_format, segments=segments), None


type _Part = tuple[str, int]
type _Entry = tuple[int, int, list[_Part]]


def _json3_entries(events: list[object]) -> list[_Entry]:
    entries: list[_Entry] = []
    for raw in events:
        event = cast("dict[str, object]", raw) if isinstance(raw, dict) else {}
        segs = event.get("segs")
        start_ms = event.get("tStartMs")
        if not isinstance(segs, list) or not isinstance(start_ms, int):
            continue
        parts: list[_Part] = []
        for raw_seg in cast("list[object]", segs):
            seg = cast("dict[str, object]", raw_seg) if isinstance(raw_seg, dict) else {}
            text = seg.get("utf8")
            offset = seg.get("tOffsetMs")
            parts.append((text if isinstance(text, str) else "", offset if isinstance(offset, int) else 0))
        duration_ms = event.get("dDurationMs")
        entries.append((start_ms, duration_ms if isinstance(duration_ms, int) else 0, parts))
    return entries


def _auto_words(start_ms: int, parts: list[_Part], end: float, cursor: float) -> tuple[tuple[Word, ...], float]:
    timed = [(text.strip(), (start_ms + offset) / 1000) for text, offset in parts if text.strip()]
    words: list[Word] = []
    for index, (text, raw_start) in enumerate(timed):
        start = max(raw_start, cursor)
        cursor = start
        next_start = max(timed[index + 1][1], start) if index + 1 < len(timed) else end
        words.append(Word(text=text, start=start, end=max(next_start, start)))
    return tuple(words), cursor


def _clean(text: str) -> str:
    return _SPACES.sub(" ", text).strip()


def _cue_blocks(content: str) -> list[tuple[float, float, list[str]]]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    cues: list[tuple[float, float, list[str]]] = []
    for block in re.split(r"\n[ \t]*\n", normalized):
        lines = block.split("\n")
        timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing is None:
            continue
        start_part, _, end_part = lines[timing].partition("-->")
        start = _parse_timestamp(start_part)
        end = _parse_timestamp(end_part)
        if start is None or end is None:
            continue
        text_lines = [cleaned for raw in lines[timing + 1 :] if (cleaned := _clean(html.unescape(_TAG.sub("", raw))))]
        cues.append((start, max(end, start), text_lines))
    return cues


def _parse_timestamp(token: str) -> float | None:
    match = _TIMESTAMP.search(token)
    if not match:
        return None
    hours = int(match.group(1)) if match.group(1) else 0
    millis = int(match.group(4).ljust(3, "0")[:3])
    return hours * 3600 + int(match.group(2)) * 60 + int(match.group(3)) + millis / 1000.0
