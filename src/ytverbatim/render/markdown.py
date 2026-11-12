import math
import re
from itertools import groupby

from ytverbatim.contract import ContextDocument, Flag, Segment

_HOUR = 3600
_WATCH_URL = "https://youtu.be/"

# `_` only emphasises at a word edge; escaping it inside identifiers would just add noise. `&` is escaped only
# where it would start an HTML entity, so "AT&T" and "R&D" stay untouched. Every `~` is escaped, because GFM
# strikes through text between single tildes as well as double ones.
_ESCAPE = re.compile(
    r"[\\`*\[\]<>~]"
    r"|(?<![^\W_])_"
    r"|_(?![^\W_])"
    r"|&(?=#\d+;|#[xX][0-9a-fA-F]+;|[A-Za-z][A-Za-z0-9]*;)"
)

# The edit stage that records the replaced text for flags whose footnote shows a "was" value.
_BEFORE_STAGE = {"entity_corrected": "correction", "numeric_fixed": "numeric"}


def render_markdown(doc: ContextDocument) -> str:
    return _Renderer(doc).render()


class _Renderer:
    def __init__(self, doc: ContextDocument) -> None:
        self._doc = doc
        self._hours = doc.video.duration_s >= _HOUR
        self._joiner = "" if doc.transcript.language.lower().startswith("zh") else " "
        self._footnotes: list[str] = []

    def render(self) -> str:
        # The body is rendered first: the provenance line reports how many flags it actually marked.
        body = self._body()
        blocks = [*self._header(), *body]
        if self._footnotes:
            blocks.append("\n".join(f"[^{number}]: {text}" for number, text in enumerate(self._footnotes, start=1)))
        return "\n\n".join(blocks) + "\n"

    def _header(self) -> list[str]:
        video = self._doc.video
        facts = [_escape(video.channel)]
        if video.upload_date is not None:
            facts.append(video.upload_date.isoformat())
        facts += [self._clock(video.duration_s), f"{_WATCH_URL}{video.id}"]
        return [f"# {_escape(video.title)}", " · ".join(facts), self._provenance()]

    def _provenance(self) -> str:
        transcript = self._doc.transcript
        flags = len(self._footnotes)
        parts = [
            f"Transcript: {_escape(self._source_label())}",
            f"punctuation: {transcript.punctuation}",
            f"corrected by: {_escape(transcript.correction_llm or 'none')}",
            f"{flags} flag" if flags == 1 else f"{flags} flags",
        ]
        return " · ".join(parts)

    def _source_label(self) -> str:
        runs = {run.id: run for run in self._doc.transcript.runs}
        used = {segment.source.run for segment in self._doc.segments}
        labels: list[str] = []
        for run in runs.values():
            if used and run.id not in used:
                continue
            label = {"caption_manual": "manual captions", "caption_auto": "auto captions"}.get(
                run.kind, run.engine or run.id
            )
            if label not in labels:
                labels.append(label)
        return " + ".join(labels)

    def _body(self) -> list[str]:
        doc = self._doc
        blocks: list[str] = []
        if not doc.chapters:
            blocks.append("## Transcript")
        current_chapter: int | None = None
        for (_, chapter_index), group in groupby(doc.segments, key=lambda s: (s.paragraph, s.chapter)):
            segments = list(group)
            if chapter_index is not None and chapter_index != current_chapter:
                chapter = doc.chapters[chapter_index]
                blocks.append(f"## {self._link(chapter.start)} {_escape(chapter.title)}")
                current_chapter = chapter_index
            blocks.append(self._paragraph(segments))
        return blocks

    def _paragraph(self, segments: list[Segment]) -> str:
        first = segments[0]
        paragraph = self._doc.paragraphs[first.paragraph]
        start = paragraph.start if first.id == paragraph.first_segment else first.start
        texts = (self._segment(segment) for segment in segments)
        return f"{self._link(start)} {self._joiner.join(text for text in texts if text)}"

    def _segment(self, segment: Segment) -> str:
        text = segment.text.replace("\r", " ").replace("\n", " ")
        escaped = _escape_positions(text)
        markers: dict[int, list[str]] = {}
        for flag in sorted(_visible_flags(segment), key=lambda f: (f.char_end, f.char_start, f.kind, f.note)):
            self._footnotes.append(self._footnote(segment, flag))
            markers.setdefault(flag.char_end, []).append(f"[^{len(self._footnotes)}]")
        out: list[str] = []
        for offset, char in enumerate(text):
            out += markers.get(offset, [])
            out.append(f"\\{char}" if offset in escaped else char)
        out += markers.get(len(text), [])
        return "".join(out).strip()

    def _footnote(self, segment: Segment, flag: Flag) -> str:
        head = f"{self._clock(segment.start)} — {flag.kind}"
        parts = [f"{head}: {_escape(flag.note)}" if flag.note else head]
        before = _before_text(segment, flag)
        if before is not None:
            parts.append(f"was {_escape(before)}")
        if flag.readings:
            heard = " / ".join(f"{_escape(r.text)} ({_escape(r.source)})" for r in flag.readings)
            parts.append(f"heard {heard}")
        return "; ".join(parts)

    def _link(self, seconds: float) -> str:
        return f"[{self._clock(seconds)}]({_WATCH_URL}{self._doc.video.id}?t={_whole(seconds)})"

    def _clock(self, seconds: float) -> str:
        total = _whole(seconds)
        minutes, secs = divmod(total, 60)
        if self._hours:
            hours, minutes = divmod(minutes, 60)
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


def _whole(seconds: float) -> int:
    return math.floor(seconds)


def _escape_positions(text: str) -> set[int]:
    return {match.start() for match in _ESCAPE.finditer(text)}


def _escape(text: str) -> str:
    flat = text.replace("\r", " ").replace("\n", " ")
    escaped = _escape_positions(flat)
    return "".join(f"\\{char}" if offset in escaped else char for offset, char in enumerate(flat))


def _visible_flags(segment: Segment) -> list[Flag]:
    low_confidence = [flag for flag in segment.flags if flag.kind == "low_confidence"]
    return [
        flag
        for flag in segment.flags
        if flag.kind != "numeric_single_source"
        or any(flag.char_start < low.char_end and low.char_start < flag.char_end for low in low_confidence)
    ]


def _before_text(segment: Segment, flag: Flag) -> str | None:
    stage = _BEFORE_STAGE.get(flag.kind)
    if stage is None:
        return None
    span = segment.text[flag.char_start : flag.char_end]
    matches = [edit.before for edit in segment.edits if edit.stage == stage and edit.after == span and edit.before]
    return matches[-1] if matches else None
