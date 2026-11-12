"""Biasing text for context-aware engines, built from what the uploader wrote about the video.

Pure and deterministic: the same metadata always yields the same `AsrContext`, so a recorded run can be reproduced.
"""

import re
from collections.abc import Iterable, Sequence

from ytverbatim.acquire.metadata import VideoMetadata
from ytverbatim.asr.protocol import AsrContext

MAX_CONTEXT_CHARS = 600
# Hotwords get a fixed share of the budget so a glossary-heavy description cannot crowd out the prompt.
_HOTWORD_BUDGET = 200
_MAX_HOTWORD_CHARS = 40

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_TIMESTAMP = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)")
_BLANK_RUN = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{2,}")

# The lookarounds only reject ASCII alphanumerics, so a ticker glued to Chinese text (`XLP重要位置`) still matches.
_DOLLAR_TICKER = re.compile(r"(?<![A-Za-z0-9])\$([A-Za-z]{1,5})(?![A-Za-z0-9])")
_UPPERCASE_TOKEN = re.compile(r"(?<![A-Za-z0-9$])[A-Z]{2,5}(?![A-Za-z0-9])")
_MIXED_CASE_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?:[a-z]+[A-Z][A-Za-z]*|[A-Z][a-z]+[A-Z][A-Za-z]*)(?![A-Za-z0-9])")
_CAPITALIZED_RUN = re.compile(r"(?<![A-Za-z0-9])[A-Z][a-z]+(?: [A-Z][a-z]+)+(?![A-Za-z0-9])")


def context_from_metadata(metadata: VideoMetadata) -> AsrContext:
    return build_context(metadata.title, metadata.description, [chapter.title for chapter in metadata.chapters])


def build_context(title: str, description: str = "", chapter_titles: Sequence[str] = ()) -> AsrContext:
    """Prompt text plus hotwords; the prompt and the hotwords the engine appends to it fit in `MAX_CONTEXT_CHARS`.

    The prompt reads title, chapter titles, then description, so truncation drops the least specific text first.
    """
    sources = [_clean(title), *(_clean(chapter) for chapter in chapter_titles), _clean(description)]
    hotwords = _hotwords(sources)
    # The adapter hands both to the engine, which joins them as `prompt + "\n" + ", ".join(hotwords)`.
    reserved = len(", ".join(hotwords)) + 1 if hotwords else 0
    prompt = _truncate("\n".join(part for part in sources if part), MAX_CONTEXT_CHARS - reserved)
    return AsrContext(system_prompt=prompt, hotwords=hotwords)


def _clean(text: str) -> str:
    text = _TIMESTAMP.sub(" ", _URL.sub(" ", text))
    text = _BLANK_RUN.sub(" ", text)
    lines = (line.strip() for line in text.splitlines())
    return _BLANK_LINES.sub("\n", "\n".join(lines)).strip()


def _hotwords(sources: Iterable[str]) -> tuple[str, ...]:
    found: dict[str, str] = {}
    for source in sources:
        for word in _candidates(source):
            if len(word) <= _MAX_HOTWORD_CHARS:
                found.setdefault(word.casefold(), word)
    kept: list[str] = []
    used = 0
    for word in found.values():
        joined = used + len(word) + (len(", ") if kept else 0)
        if joined > _HOTWORD_BUDGET:
            break
        kept.append(word)
        used = joined
    return tuple(kept)


def _candidates(text: str) -> list[str]:
    """Tickers and Latin names in the order they appear; `$nvda` becomes `NVDA`."""
    matches: list[tuple[int, str]] = [(m.start(), m.group(1).upper()) for m in _DOLLAR_TICKER.finditer(text)]
    for pattern in (_UPPERCASE_TOKEN, _MIXED_CASE_TOKEN, _CAPITALIZED_RUN):
        matches.extend((m.start(), m.group()) for m in pattern.finditer(text))
    return [word for _, word in sorted(matches)]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # Never end on half a Latin word: a clipped "Bitco" would bias the decoder toward something that is not there.
    if _is_ascii_word_char(text[limit]) and _is_ascii_word_char(cut[-1]) and " " in cut:
        cut = cut[: cut.rindex(" ")]
    return cut.rstrip()


def _is_ascii_word_char(char: str) -> bool:
    return char.isascii() and char.isalnum()
