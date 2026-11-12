import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ytverbatim.errors import YtDlpError
from ytverbatim.ids import video_url
from ytverbatim.ytdlp.types import Signal, YtDlpRunner

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 300.0
DEFAULT_MAX_COMMENTS = 30

# Failures that mean YouTube is pushing back on this connection must reach the batch policy, not be swallowed.
_STOP_SIGNALS = frozenset({Signal.BOT_CHECK, Signal.IP_BLOCKED, Signal.CAPTCHA, Signal.RATE_LIMITED})


@dataclass(frozen=True)
class CreatorNote:
    text: str
    author: str
    like_count: int | None
    published: str


def fetch_creator_note(
    runner: YtDlpRunner,
    video_id: str,
    *,
    cookies: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_comments: int = DEFAULT_MAX_COMMENTS,
) -> CreatorNote | None:
    """Return the uploader's pinned comment, or None. Opt-in: comment extraction is an extra, rate-limit-prone request.

    The note is an enrichment (errata the uploader pinned), so an ordinary failure returns None.
    """
    args = [
        "-J",
        "--no-playlist",
        "--skip-download",
        "--write-comments",
        "--extractor-args",
        # Replies disabled; the pinned comment sorts first under comment_sort=top.
        f"youtube:comment_sort=top;max_comments={max_comments},{max_comments},0,0",
        video_url(video_id),
    ]
    try:
        result = runner.run(args, cookies=cookies, timeout_s=timeout_s, label="creator-note")
        info: object = json.loads(result.stdout)
    except YtDlpError as exc:
        if any(signal in _STOP_SIGNALS for signal in (exc.primary, *exc.secondary)):
            raise
        log.warning("creator note skipped: %s", exc.primary.value if exc.primary else "unclassified yt-dlp failure")
        return None
    except json.JSONDecodeError:
        log.warning("creator note skipped: yt-dlp output was not JSON")
        return None
    comments = cast("dict[str, object]", info).get("comments") if isinstance(info, dict) else None
    return select_pinned_creator_comment(cast("list[object]", comments) if isinstance(comments, list) else [])


def select_pinned_creator_comment(comments: list[object]) -> CreatorNote | None:
    """Both flags are required: a pinned viewer comment or an unpinned creator comment is not an erratum."""
    for raw in comments:
        if not isinstance(raw, dict):
            continue
        comment = cast("dict[str, object]", raw)
        if not (comment.get("is_pinned") and comment.get("author_is_uploader")):
            continue
        text = comment.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        author = comment.get("author")
        likes = comment.get("like_count")
        published = comment.get("_time_text") or comment.get("time_text")
        return CreatorNote(
            text=text.strip(),
            author=author if isinstance(author, str) else "",
            like_count=likes if isinstance(likes, int) and not isinstance(likes, bool) else None,
            published=published if isinstance(published, str) else "",
        )
    return None
