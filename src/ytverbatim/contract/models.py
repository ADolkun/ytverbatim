import re
from collections.abc import Sequence
from datetime import date
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA_VERSION = "1.0"

# An absolute /Users/<name>/ or /home/<name>/ path, but not a URL path such as example.com/home/user/.
_HOME_PATH = re.compile(r"(?<![\w.~-])/(?:Users|home)/[^/\s]+(?:/|\s|$)")


def _round_ms(value: float) -> float:
    return round(value, 3)


def _sorted_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _sorted_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_json(item) for item in value]
    return value


def _sort_object(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: _sorted_json(value[key]) for key in sorted(value)}


def _sort_keys[T](mapping: dict[str, T]) -> dict[str, T]:
    return {key: mapping[key] for key in sorted(mapping)}


Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False), AfterValidator(_round_ms)]
# Free-form blobs are key-sorted on validation so a dump is byte-stable whatever order the producer inserted keys in.
JsonObject = Annotated[dict[str, JsonValue], AfterValidator(_sort_object)]
Counts = Annotated[dict[str, int], AfterValidator(_sort_keys)]
Kind = Literal["asr", "caption_manual", "caption_auto"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Word(_Model):
    text: str
    start: Seconds
    end: Seconds
    conf: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError(f"word {self.text!r} ends ({self.end}) before it starts ({self.start})")
        return self


class Reading(_Model):
    # "asr:<engine-id>" | "caption_manual" | "caption_auto" | "asr2:<engine-id>"
    source: str
    text: str


class Edit(_Model):
    stage: Literal["punctuation", "correction", "numeric"]
    before: str
    after: str
    llm: str | None


class Flag(_Model):
    kind: Literal[
        "numeric_disputed",
        "numeric_fixed",
        "numeric_single_source",
        "entity_corrected",
        "low_confidence",
        "decoder_anomaly",
        "timing_interpolated",
        "punctuation_failed",
    ]
    # Half-open offsets into Segment.text; the render marks this occurrence, never a global mask.
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    note: str
    readings: tuple[Reading, ...] = ()

    @model_validator(mode="after")
    def _non_empty_span(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError(f"flag span [{self.char_start}, {self.char_end}) is empty or reversed")
        return self


class SourceRef(_Model):
    kind: Kind
    run: str


class Segment(_Model):
    id: int = Field(ge=0)
    start: Seconds
    end: Seconds
    text: str
    # None when the engine produced no word timings; timing_interpolated marks only timings that were interpolated.
    words: tuple[Word, ...] | None = None
    source: SourceRef
    edits: tuple[Edit, ...] = ()
    flags: tuple[Flag, ...] = ()
    paragraph: int = Field(ge=0)
    # None only when the document has no chapters.
    chapter: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.end < self.start:
            raise ValueError(f"segment {self.id} ends ({self.end}) before it starts ({self.start})")
        for flag in self.flags:
            if flag.char_end > len(self.text):
                raise ValueError(
                    f"segment {self.id}: {flag.kind} flag ends at {flag.char_end}, past the {len(self.text)}-char text"
                )
        if self.words is not None:
            for previous, word in zip(self.words, self.words[1:], strict=False):
                if word.start < previous.start:
                    raise ValueError(f"segment {self.id}: word {word.text!r} starts before the word preceding it")
        return self


class Paragraph(_Model):
    index: int = Field(ge=0)
    start: Seconds
    end: Seconds
    first_segment: int = Field(ge=0)
    last_segment: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        _check_span(f"paragraph {self.index}", self.start, self.end, self.first_segment, self.last_segment)
        return self


class Chapter(_Model):
    index: int = Field(ge=0)
    title: str
    start: Seconds
    end: Seconds
    # Reserved: v1 takes chapters from YouTube only.
    source: Literal["youtube", "llm"]
    first_segment: int = Field(ge=0)
    last_segment: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        _check_span(f"chapter {self.index}", self.start, self.end, self.first_segment, self.last_segment)
        return self


class TranscriptRun(_Model):
    id: str
    kind: Kind
    engine: str | None
    model_repo: str | None
    model_revision: str | None
    params: JsonObject
    vad: JsonObject | None
    chunking: JsonObject | None
    aligner: str | None


class Transcript(_Model):
    language: str
    script: Literal["zh-Hans", "zh-Hant", "n/a"]
    runs: tuple[TranscriptRun, ...]
    punctuation: Literal["engine", "llm", "caption", "none"]
    correction_llm: str | None


class Access(_Model):
    auth_source: Literal["none", "chrome", "cookie-file"]
    cookies_used: bool
    logged_in: bool | None
    premium: bool | None
    member: bool | None
    availability: str | None
    ytdlp_version: str
    # Never cookie values, emails or paths under the home directory.
    warnings: tuple[str, ...]

    @model_validator(mode="after")
    def _no_home_paths(self) -> Self:
        for warning in self.warnings:
            if _HOME_PATH.search(warning):
                raise ValueError("access warning contains an absolute home path; redact it before recording")
        return self


class Video(_Model):
    id: str
    url: str
    title: str
    channel: str
    channel_id: str
    upload_date: date | None
    duration_s: Seconds
    description: str
    language: str | None
    tags: tuple[str, ...]


class Quality(_Model):
    speech_seconds: Seconds
    transcribed_seconds: Seconds
    decoder_anomalies: int = Field(ge=0)
    flags_by_kind: Counts
    warnings: tuple[str, ...]


class LLMUsage(_Model):
    stage: str
    client: str
    calls: int = Field(ge=0)
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)


class Generator(_Model):
    name: str
    version: str


def _check_span(label: str, start: float, end: float, first: int, last: int) -> None:
    if end < start:
        raise ValueError(f"{label} ends ({end}) before it starts ({start})")
    if last < first:
        raise ValueError(f"{label} last_segment ({last}) precedes first_segment ({first})")


def _check_partition(label: str, ranges: Sequence[tuple[int, int, int]], segment_count: int) -> None:
    """`ranges` holds (index, first_segment, last_segment), in document order."""
    for position, (index, _, _) in enumerate(ranges):
        if index != position:
            raise ValueError(f"{label} indexes must run 0..{len(ranges) - 1} in order; found {index} at {position}")
    if segment_count == 0:
        if ranges:
            raise ValueError(f"{label}s present but there are no segments")
        return
    if not ranges:
        raise ValueError(f"{label} ranges must cover all {segment_count} segments; none given")
    expected = 0
    for index, first, last in ranges:
        if first != expected:
            gap = "overlaps the previous range" if first < expected else f"leaves segment {expected} uncovered"
            raise ValueError(f"{label} {index} starts at segment {first} and {gap}")
        expected = last + 1
    if expected != segment_count:
        raise ValueError(f"{label} ranges end at segment {expected - 1}; the document has {segment_count} segments")


class ContextDocument(_Model):
    schema_version: Literal["1.0"]
    generator: Generator
    created_at: AwareDatetime
    video: Video
    access: Access
    transcript: Transcript
    chapters: tuple[Chapter, ...]
    paragraphs: tuple[Paragraph, ...]
    segments: tuple[Segment, ...]
    quality: Quality
    llm_usage: tuple[LLMUsage, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        self._check_segments()
        self._check_runs()
        _check_partition(
            "paragraph", [(p.index, p.first_segment, p.last_segment) for p in self.paragraphs], len(self.segments)
        )
        if self.chapters:
            _check_partition(
                "chapter", [(c.index, c.first_segment, c.last_segment) for c in self.chapters], len(self.segments)
            )
        self._check_membership()
        return self

    def _check_segments(self) -> None:
        for position, segment in enumerate(self.segments):
            if segment.id != position:
                raise ValueError(f"segment ids must be dense from 0; found id {segment.id} at position {position}")
            if position == 0:
                continue
            previous = self.segments[position - 1]
            if segment.start < previous.start or segment.end < previous.end:
                raise ValueError(f"segment {segment.id} is not monotonic with segment {previous.id}")

    def _check_runs(self) -> None:
        run_kinds: dict[str, str] = {}
        for run in self.transcript.runs:
            if run.id in run_kinds:
                raise ValueError(f"duplicate transcript run id {run.id!r}")
            run_kinds[run.id] = run.kind
        for segment in self.segments:
            kind = run_kinds.get(segment.source.run)
            if kind is None:
                raise ValueError(f"segment {segment.id} cites run {segment.source.run!r}, absent from transcript.runs")
            if kind != segment.source.kind:
                raise ValueError(
                    f"segment {segment.id} is {segment.source.kind} but run {segment.source.run!r} is {kind}"
                )

    def _check_membership(self) -> None:
        for segment in self.segments:
            paragraph = self.paragraphs[segment.paragraph] if segment.paragraph < len(self.paragraphs) else None
            if paragraph is None or not paragraph.first_segment <= segment.id <= paragraph.last_segment:
                raise ValueError(f"segment {segment.id} names paragraph {segment.paragraph}, which does not contain it")
            if not self.chapters:
                if segment.chapter is not None:
                    raise ValueError(f"segment {segment.id} names chapter {segment.chapter} but the document has none")
                continue
            chapter = (
                self.chapters[segment.chapter]
                if segment.chapter is not None and segment.chapter < len(self.chapters)
                else None
            )
            if chapter is None or not chapter.first_segment <= segment.id <= chapter.last_segment:
                raise ValueError(f"segment {segment.id} names chapter {segment.chapter}, which does not contain it")
