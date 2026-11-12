from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

DEFAULT_MAX_COMMENTS = 30


@dataclass(frozen=True)
class CreatorNote:
    text: str
    author: str
    like_count: int | None
    published: str


def comment_args(max_comments: int = DEFAULT_MAX_COMMENTS) -> tuple[str, ...]:
    """Extra `-J` arguments for a bounded first page of top-level comments, where a pinned comment sorts first."""
    return (
        "--get-comments",
        "--extractor-args",
        # max_comments=<total>,<parents>,<replies>,<replies per thread>; any non-integer such as "all" means
        # unlimited. Replies are off: a pinned comment is always top-level.
        f"youtube:comment_sort=top;max_comments={max_comments},all,0,0",
    )


def creator_note_from_info(info: Mapping[str, object]) -> CreatorNote | None:
    comments = info.get("comments")
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
