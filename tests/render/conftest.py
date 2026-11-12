from collections.abc import Callable, Mapping, Sequence
from datetime import date

import pytest

from ytverbatim.contract import (
    Access,
    Chapter,
    ContextDocument,
    Edit,
    Flag,
    Generator,
    Paragraph,
    Quality,
    Segment,
    SourceRef,
    Transcript,
    TranscriptRun,
    Video,
)

RUN = TranscriptRun(
    id="asr-1",
    kind="asr",
    engine="example-engine",
    model_repo=None,
    model_revision=None,
    params={},
    vad=None,
    chunking=None,
    aligner=None,
)


def _seg(
    text: str,
    start: float,
    end: float,
    *,
    paragraph: int = 0,
    chapter: int | None = None,
    flags: Sequence[Flag] = (),
    edits: Sequence[Edit] = (),
) -> Segment:
    return Segment(
        id=0,
        start=start,
        end=end,
        text=text,
        source=SourceRef(kind="asr", run=RUN.id),
        edits=tuple(edits),
        flags=tuple(flags),
        paragraph=paragraph,
        chapter=chapter,
    )


def _flag(
    kind: str,
    text: str,
    span: str,
    *,
    note: str = "",
    occurrence: int = 0,
    readings: Sequence[tuple[str, str]] = (),
) -> Flag:
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(span, start + 1)
    return Flag.model_validate(
        {
            "kind": kind,
            "char_start": start,
            "char_end": start + len(span),
            "note": note,
            "readings": [{"source": source, "text": reading} for source, reading in readings],
        }
    )


@pytest.fixture
def seg() -> Callable[..., Segment]:
    return _seg


@pytest.fixture
def flag() -> Callable[..., Flag]:
    return _flag


@pytest.fixture
def make_doc() -> Callable[..., ContextDocument]:
    def build(
        segments: Sequence[Segment],
        *,
        language: str = "en",
        duration_s: float = 600.0,
        chapters: Mapping[int, str] | None = None,
        run: TranscriptRun = RUN,
        punctuation: str = "engine",
        correction_llm: str | None = None,
        upload_date: date | None = date(2026, 1, 1),
    ) -> ContextDocument:
        numbered = [s.model_copy(update={"id": index}) for index, s in enumerate(segments)]
        paragraphs = _ranges(numbered, lambda s: s.paragraph)
        chapter_ranges = _ranges(numbered, lambda s: s.chapter) if chapters else {}
        return ContextDocument.model_validate(
            {
                "schema_version": "1.0",
                "generator": Generator(name="ytverbatim", version="0.0.1"),
                "created_at": "2026-01-02T03:04:05+00:00",
                "video": Video(
                    id="AAAAAAAAAAA",
                    url="https://www.youtube.com/watch?v=AAAAAAAAAAA",
                    title="Example Title",
                    channel="Example Channel",
                    channel_id="UCexample",
                    upload_date=upload_date,
                    duration_s=duration_s,
                    description="",
                    language=language,
                    tags=(),
                ),
                "access": Access(
                    auth_source="none",
                    cookies_used=False,
                    logged_in=None,
                    premium=None,
                    member=None,
                    availability="public",
                    ytdlp_version="2026.10.26",
                    warnings=(),
                ),
                "transcript": Transcript.model_validate(
                    {
                        "language": language,
                        "script": "zh-Hans" if language.startswith("zh") else "n/a",
                        "runs": (run,),
                        "punctuation": punctuation,
                        "correction_llm": correction_llm,
                    }
                ),
                "chapters": tuple(
                    Chapter(
                        index=index,
                        title=chapters[index] if chapters else "",
                        start=numbered[first].start,
                        end=numbered[last].end,
                        source="youtube",
                        first_segment=first,
                        last_segment=last,
                    )
                    for index, (first, last) in chapter_ranges.items()
                ),
                "paragraphs": tuple(
                    Paragraph(
                        index=index,
                        start=numbered[first].start,
                        end=numbered[last].end,
                        first_segment=first,
                        last_segment=last,
                    )
                    for index, (first, last) in paragraphs.items()
                ),
                "segments": tuple(
                    s.model_copy(update={"source": SourceRef(kind=run.kind, run=run.id)}) for s in numbered
                ),
                "quality": Quality(
                    speech_seconds=0, transcribed_seconds=0, decoder_anomalies=0, flags_by_kind={}, warnings=()
                ),
                "llm_usage": (),
            }
        )

    return build


def _ranges(segments: Sequence[Segment], key: Callable[[Segment], int | None]) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    for segment in segments:
        index = key(segment)
        if index is None:
            continue
        first, _ = ranges.get(index, (segment.id, segment.id))
        ranges[index] = (first, segment.id)
    return ranges
