import re
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import parse_qs, unquote, urlparse

_VIDEO_ID = re.compile(r"[0-9A-Za-z_-]{11}")
_LIST_ID = re.compile(r"[0-9A-Za-z_-]+")

_YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
    }
)
_SHORT_HOSTS = frozenset({"youtu.be", "www.youtu.be"})
_VIDEO_PATH_PREFIXES = frozenset({"shorts", "embed", "live", "v"})
_CHANNEL_PATH_PREFIXES = frozenset({"channel", "c", "user"})


class RefKind(StrEnum):
    VIDEO = "video"
    PLAYLIST = "playlist"
    CHANNEL = "channel"


@dataclass(frozen=True)
class YouTubeRef:
    """A classified YouTube target.

    `id` is the video id, the list id, or the channel path key (`@name`, `channel/UC...`).
    """

    kind: RefKind
    id: str
    url: str


def is_video_id(text: str) -> bool:
    return _VIDEO_ID.fullmatch(text) is not None


def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def extract_video_id(text: str) -> str | None:
    ref = classify(text)
    return ref.id if ref is not None and ref.kind is RefKind.VIDEO else None


def classify(text: str) -> YouTubeRef | None:
    text = text.strip()
    if is_video_id(text):
        return YouTubeRef(RefKind.VIDEO, text, video_url(text))
    parts = _split(text)
    if parts is None:
        return None
    host, segments, query = parts

    if host in _SHORT_HOSTS:
        if len(segments) == 1 and is_video_id(segments[0]):
            return YouTubeRef(RefKind.VIDEO, segments[0], video_url(segments[0]))
        return None

    first = segments[0] if segments else ""
    video_ids = query.get("v", [])
    if first == "watch" and len(segments) == 1:
        if len(video_ids) == 1 and is_video_id(video_ids[0]):
            return YouTubeRef(RefKind.VIDEO, video_ids[0], video_url(video_ids[0]))
        if not video_ids:
            return _playlist(query)
        return None
    if first in _VIDEO_PATH_PREFIXES and len(segments) == 2 and is_video_id(segments[1]):  # noqa: PLR2004
        return YouTubeRef(RefKind.VIDEO, segments[1], video_url(segments[1]))
    if first == "playlist" and len(segments) == 1:
        return _playlist(query)
    return _channel(segments)


def _split(text: str) -> tuple[str, list[str], dict[str, list[str]]] | None:
    candidate = text if "://" in text else f"https://{text}"
    parsed = urlparse(candidate)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in _YOUTUBE_HOSTS | _SHORT_HOSTS:
        return None
    segments = [unquote(s) for s in parsed.path.split("/") if s]
    return host, segments, parse_qs(parsed.query)


def _playlist(query: dict[str, list[str]]) -> YouTubeRef | None:
    lists = query.get("list", [])
    if len(lists) == 1 and _LIST_ID.fullmatch(lists[0]):
        return YouTubeRef(RefKind.PLAYLIST, lists[0], f"https://www.youtube.com/playlist?list={lists[0]}")
    return None


def _channel(segments: list[str]) -> YouTubeRef | None:
    if not segments:
        return None
    first = segments[0]
    if first.startswith("@") and len(first) > 1:
        key = first
    elif first in _CHANNEL_PATH_PREFIXES and len(segments) >= 2:  # noqa: PLR2004
        key = f"{first}/{segments[1]}"
    else:
        return None
    return YouTubeRef(RefKind.CHANNEL, key, f"https://www.youtube.com/{key}")
