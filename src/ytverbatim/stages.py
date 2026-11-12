"""The pure stages of `ytverbatim run`: source transcript → stretches → text stages → contract document.

A *stretch* is the unit every text stage works on: one sentence-like piece of the source transcript, with its time
range, the engine's timed words inside it, and the flags and edits raised on it so far. The source is cut into
stretches once (at its own punctuation, or at pauses and a size limit where it has none). Text stages only replace a
stretch's text; flags raised earlier are moved through each replacement by a diff, and every edit is kept.

`build_document` then re-times each stretch's final text against its words, re-cuts the whole transcript into
sentences (punctuation restored by an LLM can merge or split stretches) and maps every flag and edit onto the
sentence that holds it. Nothing here performs I/O.
"""

import collections
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from ytverbatim import __version__, ids
from ytverbatim.acquire.captions import CaptionTrack
from ytverbatim.acquire.metadata import VideoMetadata
from ytverbatim.asr import chunks
from ytverbatim.asr.models import ChunkHypothesis
from ytverbatim.asr.protocol import EngineDef
from ytverbatim.contract import (
    Access,
    Chapter,
    ContextDocument,
    Edit,
    Flag,
    LLMUsage,
    Paragraph,
    Quality,
    Segment,
    SourceRef,
    Transcript,
    TranscriptRun,
    Video,
    Word,
)
from ytverbatim.contract import Generator as GeneratorInfo
from ytverbatim.contract.models import SCHEMA_VERSION
from ytverbatim.errors import ContractError
from ytverbatim.text import clean, offsets
from ytverbatim.text import segment as seg

# Part of every stage key that runs this module's code, and of the context record: bump it on any logic change.
VERSION = "stages-v1"
UNDETERMINED_LANGUAGE = "und"
CAPTION_MANUAL = "caption_manual"

# The chunk flag that marks a decode the pipeline cannot vouch for; the others only describe how the audio was cut.
_ANOMALY_FLAG = "decoder_anomaly"

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Stretch(_Model):
    text: str
    start: Seconds
    end: Seconds
    # The engine's timed words in this stretch, absolute times; the text is re-timed against them after every edit.
    words: tuple[Word, ...] = ()
    # Whether the source had word timings here at all; untimed text in a word-timed stretch is interpolated.
    word_timed: bool = False
    # Offsets into `text`.
    flags: tuple[Flag, ...] = ()
    # Whole-stretch before/after texts, in the order the stages made them.
    edits: tuple[Edit, ...] = ()


class Transcribed(_Model):
    """The source transcript, cut into stretches, and what the source itself says about its quality."""

    run: TranscriptRun
    language: str
    stretches: tuple[Stretch, ...]
    # None when the source has no speech estimate of its own (whole-file engines, captions).
    speech_seconds: Seconds | None
    decoder_anomalies: int
    chunk_flags: dict[str, int]
    warnings: tuple[str, ...]


def run_id(definition: EngineDef) -> str:
    return f"asr:{definition.spec.id}"


def from_asr_rows(rows: Sequence[ChunkHypothesis], definition: EngineDef, metadata: VideoMetadata) -> Transcribed:
    """Turn one engine's chunk rows (sorted by chunk) into stretches; no I/O."""
    spec = definition.spec
    language = _language(rows, metadata.language)
    tokens, warnings = _tokens(rows, language)
    sentences = seg.segment_tokens(tokens, language)
    worded = {row.chunk for row in rows if any(word.text.strip() for word in row.words)}
    anomalies = {row.chunk: row for row in rows if _ANOMALY_FLAG in row.flags}
    stretches = tuple(
        Stretch(
            text=sentence.text,
            start=sentence.start,
            end=sentence.end,
            words=tuple(
                Word(text=t.text.strip(), start=t.start, end=t.end, conf=t.conf)
                for t in sentence.tokens
                if t.timed and t.text.strip()
            ),
            word_timed=any(t.origin in worded for t in sentence.tokens),
            flags=_anomaly_flags(sentence, anomalies),
        )
        for sentence in sentences
    )
    run = TranscriptRun(
        id=run_id(definition),
        kind="asr",
        engine=spec.id,
        model_repo=spec.repo,
        model_revision=spec.revision,
        params={**definition.variant, "language_hint": metadata.language, "sample_rate": spec.sample_rate},
        vad=_vad_info() if spec.max_chunk_s is not None else None,
        chunking=_chunking_info(spec.max_chunk_s, rows),
        aligner=None,
    )
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        counts.update(row.flags)
        if not row.text.strip() and "empty_output" not in row.flags:
            counts["empty_output"] += 1
    return Transcribed(
        run=run,
        language=language,
        stretches=stretches,
        # Without a VAD (whole-file engines) there is no separate speech estimate; the transcribed span stands in.
        speech_seconds=sum(row.end - row.start for row in rows) if spec.max_chunk_s is not None else None,
        decoder_anomalies=counts[_ANOMALY_FLAG],
        chunk_flags=dict(counts),
        warnings=tuple(warnings),
    )


def caption_run_id(track: CaptionTrack) -> str:
    return f"{CAPTION_MANUAL}:{track.language}"


def from_captions(track: CaptionTrack, metadata: VideoMetadata) -> Transcribed:
    """Turn a manual caption track into stretches; cue times are spread over each cue's characters."""
    language = metadata.language or track.language.split("-")[0]
    zh = seg.uses_zh_rules(" ".join(cue.text for cue in track.segments), language)
    tokens: list[seg.Token] = []
    for index, cue in enumerate(track.segments):
        piece = seg.pseudo_tokens(cue.text, cue.start, max(cue.end, cue.start), language=language, origin=index)
        tokens += _joined(tokens, piece, zh=zh)
    stretches = tuple(
        Stretch(text=sentence.text, start=sentence.start, end=sentence.end)
        for sentence in seg.segment_tokens(_monotonic(tokens), language)
    )
    run = TranscriptRun(
        id=caption_run_id(track),
        kind=CAPTION_MANUAL,
        engine=None,
        model_repo=None,
        model_revision=None,
        params={"language": track.language, "format": track.source_format},
        vad=None,
        chunking=None,
        aligner=None,
    )
    return Transcribed(
        run=run,
        language=language,
        stretches=stretches,
        speech_seconds=None,
        decoder_anomalies=0,
        chunk_flags={},
        warnings=(),
    )


def normalize(
    stretches: Sequence[Stretch], language: str, *, script: clean.Script, fillers: clean.Fillers
) -> tuple[Stretch, ...]:
    """Convert script and remove fillers; flags move with the text and no edit is recorded (nothing was misheard)."""
    out: list[Stretch] = []
    for stretch in stretches:
        result = clean.normalize(stretch.text, language, script=script, fillers=fillers)
        words = stretch.words
        if seg.is_zh(language) and script == "zh-Hans":
            # The words are matched against the text by characters, so they must be in the same script.
            words = tuple(word.model_copy(update={"text": clean.to_simplified(word.text)}) for word in words)
        out.append(
            stretch.model_copy(
                update={
                    "text": result.text,
                    "words": words,
                    "flags": offsets.remap_flags(stretch.flags, result.steps),
                }
            )
        )
    return tuple(out)


def with_results(
    stretches: Sequence[Stretch],
    texts: Sequence[str],
    edits: Sequence[Sequence[Edit]],
    flags: Sequence[Sequence[Flag]] = (),
    *,
    record: bool = True,
) -> tuple[Stretch, ...]:
    """Apply one text stage's per-stretch results: earlier flags follow the edits, the stage's own flags are added.

    With `record=False` the edits only move the flags and are not kept: punctuation changes no word, the guard proves
    it, and hundreds of inserted commas per video would bury the edits a reader needs.
    """
    out: list[Stretch] = []
    for index, stretch in enumerate(stretches):
        new_edits = tuple(edits[index])
        moved = offsets.remap_flags(stretch.flags, ((edit.before, edit.after) for edit in new_edits))
        added = tuple(flags[index]) if flags else ()
        kept = stretch.edits + new_edits if record else stretch.edits
        out.append(stretch.model_copy(update={"text": texts[index], "flags": moved + added, "edits": kept}))
    return tuple(out)


def stretch_flag(stretch: Stretch, kind: str, note: str) -> Flag:
    return Flag.model_validate({"kind": kind, "char_start": 0, "char_end": len(stretch.text), "note": note})


def build_document(  # noqa: PLR0913
    *,
    metadata: VideoMetadata,
    tags: Sequence[str],
    access: Access,
    source: Transcribed,
    stretches: Sequence[Stretch],
    chapters: bool,
    created_at: datetime,
    correction_llm: str | None = None,
    llm_punctuated: bool = False,
    warnings: Sequence[str] = (),
    llm_usage: Sequence[LLMUsage] = (),
) -> ContextDocument:
    """Assemble and validate the document from the final stretches; `llm_punctuated` says an LLM added punctuation.

    Raises `ContractError` when the inputs cannot make a valid document, such as a chapter that ends before it starts.
    """
    try:
        return _build(
            metadata,
            tags,
            access,
            source,
            stretches,
            chapters=chapters,
            created_at=created_at,
            correction_llm=correction_llm,
            llm_punctuated=llm_punctuated,
            warnings=warnings,
            llm_usage=llm_usage,
        )
    except ValidationError as exc:
        raise ContractError(f"{metadata.video_id}: the transcript does not fit the contract:\n{exc}") from exc


@dataclass(frozen=True, slots=True)
class _Placed:
    """Where one token of the final token stream came from and where it landed."""

    stretch: int
    # The token's first character as an offset into its stretch's text; -1 for a joining space added before it.
    stretch_start: int
    stretch_end: int
    sentence: int
    sentence_start: int


@dataclass(frozen=True, slots=True)
class _Layout:
    sentences: list[seg.Sentence]
    # Per stretch, its tokens in order.
    placed: dict[int, list[_Placed]]
    interpolated: set[int]

    def spans(self, stretch: int, start: int, end: int) -> dict[int, tuple[int, int]]:
        """Find the sentence ranges that hold `[start, end)` of a stretch, by sentence index."""
        hits: dict[int, tuple[int, int]] = {}
        for token in self.placed.get(stretch, ()):
            lo, hi = max(start, token.stretch_start), min(end, token.stretch_end)
            if hi <= lo:
                continue
            length = len(self.sentences[token.sentence].text)
            a = min(max(token.sentence_start + lo - token.stretch_start, 0), length)
            b = min(max(token.sentence_start + hi - token.stretch_start, 0), length)
            if b <= a:
                continue
            known = hits.get(token.sentence)
            hits[token.sentence] = (min(a, known[0]), max(b, known[1])) if known else (a, b)
        return hits

    def sentence_at(self, stretch: int, offset: int) -> int | None:
        tokens = self.placed.get(stretch)
        if not tokens:
            return None
        for token in tokens:
            if offset < token.stretch_end:
                return token.sentence
        return tokens[-1].sentence


def _build(  # noqa: PLR0913
    metadata: VideoMetadata,
    tags: Sequence[str],
    access: Access,
    source: Transcribed,
    stretches: Sequence[Stretch],
    *,
    chapters: bool,
    created_at: datetime,
    correction_llm: str | None,
    llm_punctuated: bool,
    warnings: Sequence[str],
    llm_usage: Sequence[LLMUsage],
) -> ContextDocument:
    language = source.language
    layout = _layout(stretches, language)
    sentences = layout.sentences
    notes = [*source.warnings, *warnings]
    text = " ".join(sentence.text for sentence in sentences)
    script = _script(language, text)
    if script == "zh-Hant":
        notes.append("the transcript contains Traditional Chinese characters, which are not converted")
    if not sentences:
        notes.append("no speech was transcribed")
    if layout.interpolated:
        notes.append(
            f"word timings did not cover the text in {len(layout.interpolated)} stretch(es); the uncovered words are "
            "timed between their timed neighbours and their segments are flagged timing_interpolated"
        )

    flags: list[list[Flag]] = [[] for _ in sentences]
    edits: list[list[Edit]] = [[] for _ in sentences]
    for index, stretch in enumerate(stretches):
        for flag in stretch.flags:
            for k, (a, b) in layout.spans(index, flag.char_start, flag.char_end).items():
                flags[k].append(flag.model_copy(update={"char_start": a, "char_end": b}))
        for k, edit in _placed_edits(layout, index, stretch.edits):
            edits[k].append(edit)

    chapter_of, chapter_list = _chapters(metadata, sentences, enabled=chapters)
    paragraph_ranges = seg.split_paragraphs([s.start for s in sentences], [s.end for s in sentences], chapter_of)
    paragraph_of = [index for index, (first, last) in enumerate(paragraph_ranges) for _ in range(first, last + 1)]
    run = source.run
    segments = tuple(
        _segment(
            index,
            sentence,
            run,
            paragraph=paragraph_of[index],
            chapter=chapter_of[index],
            flags=flags[index],
            edits=edits[index],
            interpolated=layout.interpolated,
        )
        for index, sentence in enumerate(sentences)
    )
    paragraphs = tuple(
        Paragraph(
            index=index,
            start=sentences[first].start,
            end=max(s.end for s in sentences[first : last + 1]),
            first_segment=first,
            last_segment=last,
        )
        for index, (first, last) in enumerate(paragraph_ranges)
    )
    return ContextDocument(
        schema_version=SCHEMA_VERSION,
        generator=GeneratorInfo(name="ytverbatim", version=__version__),
        created_at=created_at,
        video=_video(metadata, tags, source.stretches),
        access=access,
        transcript=Transcript(
            language=language,
            script=script,
            runs=(run,),
            punctuation=_punctuation(text, language, run, llm_punctuated=llm_punctuated),
            correction_llm=correction_llm,
        ),
        chapters=chapter_list,
        paragraphs=paragraphs,
        segments=segments,
        quality=_quality(source, segments, notes),
        llm_usage=tuple(llm_usage),
    )


def _layout(stretches: Sequence[Stretch], language: str) -> _Layout:
    zh = seg.uses_zh_rules(" ".join(stretch.text for stretch in stretches), language)
    tokens: list[seg.Token] = []
    origins: list[tuple[int, int]] = []
    interpolated: set[int] = set()
    for index, stretch in enumerate(stretches):
        text = stretch.text.strip()
        if not text:
            continue
        piece, complete = _retimed(stretch, text, index, language)
        if not complete:
            interpolated.add(index)
        joined = _joined(tokens, piece, zh=zh)
        position = len(stretch.text) - len(stretch.text.lstrip()) - (len(joined[0].text) - len(piece[0].text))
        for token in joined:
            origins.append((index, position))
            position += len(token.text)
        tokens += joined
    tokens = _monotonic(tokens)
    kept = [(token, origin) for token, origin in zip(tokens, origins, strict=True) if token.text.strip()]
    sentences = seg.segment_tokens([token for token, _ in kept], language)
    placed: dict[int, list[_Placed]] = {}
    cursor = 0
    for number, sentence in enumerate(sentences):
        group = kept[cursor : cursor + len(sentence.tokens)]
        cursor += len(group)
        raw = "".join(token.text for token, _ in group)
        position = -(len(raw) - len(raw.lstrip()))
        for token, (stretch, start) in group:
            placed.setdefault(stretch, []).append(_Placed(stretch, start, start + len(token.text), number, position))
            position += len(token.text)
    return _Layout(sentences, placed, interpolated)


def _retimed(stretch: Stretch, text: str, index: int, language: str) -> tuple[list[seg.Token], bool]:
    """Tokens of the stretch's current text, timed by its words; False when some of it had to be interpolated."""
    if stretch.words:
        piece, complete = seg.aligned_tokens(
            text, stretch.words, stretch.start, stretch.end, language=language, origin=index
        )
        if "".join(token.text for token in piece) == text:
            return piece, complete
    piece = seg.pseudo_tokens(text, stretch.start, stretch.end, language=language, origin=index)
    return piece, not stretch.word_timed


def _joined(tokens: Sequence[seg.Token], piece: list[seg.Token], *, zh: bool) -> list[seg.Token]:
    # Stretch texts are stripped, so "end." and "Next" would otherwise fuse and hide the sentence break.
    if tokens and piece and not zh and not tokens[-1].text[-1:].isspace() and not piece[0].text[:1].isspace():
        return [replace(piece[0], text=f" {piece[0].text}"), *piece[1:]]
    return piece


def _placed_edits(layout: _Layout, stretch: int, edits: Sequence[Edit]) -> list[tuple[int, Edit]]:
    """Each edit as its changed word runs, positioned in the final text and attached to the sentence holding them."""
    placed: list[tuple[int, Edit]] = []
    for number, edit in enumerate(edits):
        later = [offsets.OffsetMap.between(e.before, e.after) for e in edits[number + 1 :]]
        for run in offsets.changed_runs(edit.before, edit.after):
            if not (run.before.strip() or run.after.strip()):
                continue
            position = run.start
            for mapping in later:
                position = mapping.start(position)
            sentence = layout.sentence_at(stretch, position)
            if sentence is not None:
                placed.append((sentence, Edit(stage=edit.stage, before=run.before, after=run.after, llm=edit.llm)))
    return placed


def _punctuation(
    text: str, language: str, run: TranscriptRun, *, llm_punctuated: bool
) -> Literal["engine", "llm", "caption", "none"]:
    if not seg.is_punctuated(text, language):
        return "none"
    if llm_punctuated:
        return "llm"
    return "caption" if run.kind != "asr" else "engine"


def _anomaly_flags(sentence: seg.Sentence, anomalies: Mapping[int, ChunkHypothesis]) -> tuple[Flag, ...]:
    flags: list[Flag] = []
    for chunk, row in anomalies.items():
        span = sentence.span_of(chunk)
        if span is not None:
            flags.append(
                Flag(
                    kind="decoder_anomaly",
                    char_start=span[0],
                    char_end=span[1],
                    note=f"the decoder looped or stalled in the chunk at {_clock(row.start)}-{_clock(row.end)}",
                )
            )
    return tuple(flags)


def _language(rows: Sequence[ChunkHypothesis], declared: str | None) -> str:
    """Return the language the engine reports for most of the text, else the one YouTube declares."""
    weight: collections.Counter[str] = collections.Counter()
    for row in rows:
        if row.language and row.text.strip():
            weight[row.language] += len(row.text)
    if weight:
        return weight.most_common(1)[0][0]
    return declared or UNDETERMINED_LANGUAGE


def _tokens(rows: Sequence[ChunkHypothesis], language: str) -> tuple[list[seg.Token], list[str]]:
    """Return the tokens of every chunk, timed by its words where they line up with its text, and timing warnings.

    A chunk without any word timings is spread over the chunk, which the warning covers rather than a flag on every
    segment.
    """
    zh = seg.uses_zh_rules(" ".join(row.text for row in rows), language)
    tokens: list[seg.Token] = []
    chunk_level = 0
    for row in rows:
        text = row.text.strip()
        if not text:
            continue
        words = [word for word in row.words if word.text.strip()]
        if words:
            piece, _ = seg.aligned_tokens(text, words, row.start, row.end, language=language, origin=row.chunk)
        else:
            chunk_level += 1
            piece = seg.pseudo_tokens(text, row.start, row.end, language=language, origin=row.chunk)
        tokens += _joined(tokens, piece, zh=zh)
    warnings: list[str] = []
    if chunk_level:
        warnings.append(
            f"timing is chunk-level for {chunk_level} chunk(s): segment times are spread over each chunk "
            "in proportion to characters"
        )
    return _monotonic(tokens), warnings


def _monotonic(tokens: Sequence[seg.Token]) -> list[seg.Token]:
    """Segment starts must not go backwards, and engines occasionally emit a word that does.

    Word times are already clamped to their chunk, so a bogus late start can hold later ones back only to its
    chunk's end.
    """
    out: list[seg.Token] = []
    floor = 0.0
    for token in tokens:
        start = max(token.start, floor)
        out.append(replace(token, start=start, end=max(token.end, start)))
        floor = start
    return out


def _script(language: str, text: str) -> Literal["zh-Hans", "zh-Hant", "n/a"]:
    if not seg.is_zh(language):
        return "n/a"
    return "zh-Hans" if clean.to_simplified(text) == text else "zh-Hant"


def _chapters(
    metadata: VideoMetadata, sentences: Sequence[seg.Sentence], *, enabled: bool
) -> tuple[list[int | None], tuple[Chapter, ...]]:
    """Map sentences to YouTube chapters by start time, dropping chapters that no sentence starts in."""
    if not enabled or not metadata.chapters or not sentences:
        return [None] * len(sentences), ()
    youtube = sorted(metadata.chapters, key=lambda c: c.start)
    raw = seg.chapter_indexes([s.start for s in sentences], [c.start for c in youtube])
    order = list(dict.fromkeys(raw))
    compact = {old: new for new, old in enumerate(order)}
    chapters = tuple(
        Chapter(
            index=compact[old],
            title=youtube[old].title,
            start=youtube[old].start,
            end=youtube[old].end,
            source="youtube",
            first_segment=raw.index(old),
            last_segment=len(raw) - 1 - raw[::-1].index(old),
        )
        for old in order
    )
    return [compact[old] for old in raw], chapters


def _segment(
    index: int,
    sentence: seg.Sentence,
    run: TranscriptRun,
    *,
    paragraph: int,
    chapter: int | None,
    flags: Sequence[Flag],
    edits: Sequence[Edit],
    interpolated: Collection[int],
) -> Segment:
    words = tuple(
        Word(text=t.text.strip(), start=t.start, end=t.end, conf=t.conf)
        for t in sentence.tokens
        if t.timed and t.text.strip()
    )
    all_flags = list(flags)
    if any(not t.timed and t.origin in interpolated for t in sentence.tokens):
        all_flags.append(
            Flag(
                kind="timing_interpolated",
                char_start=0,
                char_end=len(sentence.text),
                note="some words here had no engine timing; their times are interpolated from the words around them",
            )
        )
    return Segment(
        id=index,
        start=sentence.start,
        end=sentence.end,
        text=sentence.text,
        words=words or None,
        source=SourceRef(kind=run.kind, run=run.id),
        edits=tuple(edits),
        flags=tuple(sorted(all_flags, key=lambda f: f.char_start)),
        paragraph=paragraph,
        chapter=chapter,
    )


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def _video(metadata: VideoMetadata, tags: Sequence[str], stretches: Sequence[Stretch]) -> Video:
    duration = metadata.duration if metadata.duration is not None else max((s.end for s in stretches), default=0.0)
    return Video(
        id=metadata.video_id,
        url=ids.video_url(metadata.video_id),
        title=metadata.title,
        channel=metadata.channel,
        channel_id=metadata.channel_id or "",
        upload_date=_upload_date(metadata.upload_date),
        duration_s=duration,
        description=metadata.description,
        language=metadata.language,
        tags=tuple(tags),
    )


def _upload_date(raw: str | None) -> date | None:
    try:
        return datetime.strptime(raw, "%Y%m%d").date() if raw else None
    except ValueError:
        return None


def _vad_info() -> dict[str, JsonValue]:
    return {"repo": chunks.SILERO_REPO, "revision": chunks.SILERO_REVISION, **asdict(chunks.DEFAULT_VAD_PARAMS)}


def _chunking_info(max_chunk_s: float | None, rows: Sequence[ChunkHypothesis]) -> dict[str, JsonValue] | None:
    if max_chunk_s is None:
        return None
    return {
        "max_chunk_s": max_chunk_s,
        "tail_fraction": chunks.DEFAULT_TAIL_FRACTION,
        "min_pause_s": chunks.DEFAULT_MIN_PAUSE_S,
        "chunks": len(rows),
    }


def _quality(source: Transcribed, segments: Sequence[Segment], warnings: Sequence[str]) -> Quality:
    """Flag counts are the document's own flags by kind, plus the source's chunk diagnostics (forced cuts, ...)."""
    counts: collections.Counter[str] = collections.Counter(flag.kind for s in segments for flag in s.flags)
    for kind, count in source.chunk_flags.items():
        if kind != _ANOMALY_FLAG:
            counts[kind] += count
    transcribed = _covered_seconds([(s.start, s.end) for s in segments])
    return Quality(
        speech_seconds=source.speech_seconds if source.speech_seconds is not None else transcribed,
        transcribed_seconds=transcribed,
        decoder_anomalies=source.decoder_anomalies,
        flags_by_kind=dict(counts),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _covered_seconds(spans: Collection[tuple[float, float]]) -> float:
    total = 0.0
    reach = 0.0
    for start, end in sorted(spans):
        start = max(start, reach)
        if end > start:
            total += end - start
            reach = end
    return total
