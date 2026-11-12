from datetime import UTC, datetime

from ytverbatim.acquire.captions import CaptionKind, CaptionSegment, CaptionTrack
from ytverbatim.acquire.metadata import VideoMetadata
from ytverbatim.asr import registry
from ytverbatim.asr.models import ChunkHypothesis
from ytverbatim.contract import Access, ContextDocument, Edit, Flag, Word, dump_document, load_document
from ytverbatim.render import render_markdown
from ytverbatim.stages import (
    Stretch,
    Transcribed,
    build_document,
    from_asr_rows,
    from_captions,
    normalize,
    with_results,
    word_confidence,
)

ACCESS = Access(
    auth_source="none",
    cookies_used=False,
    logged_in=None,
    premium=None,
    member=None,
    availability="public",
    ytdlp_version="2099.01.01",
    warnings=(),
)
NOW = datetime(2030, 1, 1, tzinfo=UTC)
LLM = "fake:llm"


def metadata(language: str = "zh") -> VideoMetadata:
    return VideoMetadata(
        video_id="synthVid001",
        title="Synthetic title",
        channel="Synthetic channel",
        channel_id="UCsynthetic",
        upload_date="20300101",
        duration=100.0,
        description="",
        language=language,
        availability="public",
        live_status=None,
        chapters=(),
        subtitle_langs=(),
        auto_caption_lang=None,
    )


def flag_on(text: str, part: str, kind: str = "decoder_anomaly") -> Flag:
    start = text.index(part)
    return Flag.model_validate({"kind": kind, "char_start": start, "char_end": start + len(part), "note": part})


def source_of(rows: list[ChunkHypothesis], language: str = "zh") -> Transcribed:
    return from_asr_rows(rows, registry.get("w-prod"), metadata(language))


def row(text: str, start: float, end: float, words: tuple[Word, ...] = (), language: str = "zh") -> ChunkHypothesis:
    return ChunkHypothesis(
        hypothesis="w-prod",
        file="synthVid001",
        chunk=0,
        start=start,
        end=end,
        text=text,
        language=language,
        words=words,
    )


def document(
    source: Transcribed, stretches: tuple[Stretch, ...], language: str = "zh", *, llm_punctuated: bool = False
) -> ContextDocument:
    doc = build_document(
        metadata=metadata(language),
        tags=(),
        access=ACCESS,
        source=source,
        stretches=stretches,
        chapters=False,
        created_at=NOW,
        llm_punctuated=llm_punctuated,
    )
    assert load_document(dump_document(doc)) == doc
    return doc


def flagged(doc: ContextDocument) -> list[tuple[int, str, str]]:
    return [(s.id, f.kind, s.text[f.char_start : f.char_end]) for s in doc.segments for f in s.flags]


def test_flags_stay_on_their_characters_across_normalize_and_punctuation() -> None:
    source = source_of([row("嗯 這個GAAP數據很好 嗯 然後估值也不低", 0.0, 8.0)])
    (stretch,) = source.stretches
    stretch = stretch.model_copy(
        update={"flags": (flag_on(stretch.text, "GAAP數據"), flag_on(stretch.text, "估值", "low_confidence"))}
    )
    (normal,) = normalize((stretch,), "zh", script="zh-Hans", fillers="remove")
    assert normal.text == "这个GAAP数据很好 然后估值也不低"
    punctuated = "这个GAAP数据很好。然后估值也不低。"
    edit = Edit(stage="punctuation", before=normal.text, after=punctuated, llm=LLM)
    # As the pipeline does: punctuation moves the flags but is not recorded as edits.
    final = with_results((normal,), [punctuated], [[edit]], record=False)

    doc = document(source, final, llm_punctuated=True)
    assert [s.text for s in doc.segments] == ["这个GAAP数据很好。", "然后估值也不低。"]
    assert flagged(doc) == [(0, "decoder_anomaly", "GAAP数据"), (1, "low_confidence", "估值")]
    assert all(s.edits == () for s in doc.segments)
    assert doc.transcript.punctuation == "llm" and doc.transcript.script == "zh-Hans"


def test_a_corrected_word_is_interpolated_while_the_rest_keeps_engine_times() -> None:
    pieces = [" The", " GAP", " margin", " rose."]
    words = tuple(Word(text=p, start=1.0 + i, end=1.5 + i, conf=0.9) for i, p in enumerate(pieces))
    source = source_of([row("".join(pieces), 0.0, 10.0, words, "en")], "en")
    (stretch,) = source.stretches
    after = "The GAAP margin rose."
    corrected = Edit(stage="correction", before=stretch.text, after=after, llm=LLM)
    final = with_results((stretch,), [after], [[corrected]], [[flag_on(after, "GAAP", "entity_corrected")]])

    doc = document(source, final, "en")
    (segment,) = doc.segments
    assert segment.words is not None
    assert [(w.text, w.start) for w in segment.words] == [("The", 1.0), ("margin", 3.0), ("rose.", 4.0)]
    assert flagged(doc) == [(0, "timing_interpolated", after), (0, "entity_corrected", "GAAP")]
    assert [(e.stage, e.before, e.after) for e in segment.edits] == [("correction", "GAP", "GAAP")]
    markdown = render_markdown(doc)
    assert "GAAP[^1]" in markdown and "was GAP" in markdown and "timing_interpolated" not in markdown
    assert any("did not cover" in w for w in doc.quality.warnings)


def test_latin_and_digit_pieces_the_engine_split_keep_their_times_without_interpolation() -> None:
    pieces = ["好，", "看", "X", "LP", "的", "位", "置。", "成", "交", "17", "1.", "6", "万"]
    words = tuple(Word(text=p, start=1.0 + i, end=1.5 + i) for i, p in enumerate(pieces))
    source = source_of([row("好，看XLP的位置。成交171.6万", 0.0, 20.0, words)])
    stored = [word.text for stretch in source.stretches for word in stretch.words]
    assert stored == ["好，", "看", "XLP", "的", "位", "置。", "成", "交", "171.6", "万"]

    doc = document(source, source.stretches)
    assert flagged(doc) == []
    assert not any("did not cover" in w for w in doc.quality.warnings)
    timed = {w.text: (w.start, w.end) for segment in doc.segments for w in segment.words or ()}
    assert timed["XLP"] == (3.0, 4.5)
    assert timed["171.6"] == (10.0, 12.5)


def test_text_the_engine_never_timed_is_still_interpolated_and_flagged() -> None:
    words = tuple(Word(text=p, start=1.0 + i, end=1.5 + i) for i, p in enumerate(["好", "看", "的", "位", "置"]))
    source = source_of([row("好看XLP的位置", 0.0, 10.0, words)])
    doc = document(source, source.stretches)
    assert flagged(doc) == [(0, "timing_interpolated", "好看XLP的位置")]
    assert any("did not cover" in w for w in doc.quality.warnings)


def test_numeric_edits_after_punctuation_land_in_the_final_text() -> None:
    source = source_of([row("营收增长了十五个百分点 利润也不错", 0.0, 6.0)])
    (stretch,) = source.stretches
    punctuated = "营收增长了十五个百分点，利润也不错。"
    fixed = "营收增长了五十个百分点，利润也不错。"
    first = with_results(
        (stretch,),
        [punctuated],
        [[Edit(stage="punctuation", before=stretch.text, after=punctuated, llm=LLM)]],
        record=False,
    )
    numeric_flag = flag_on(fixed, "五十个百分点", "numeric_fixed")
    final = with_results(
        first, [fixed], [[Edit(stage="numeric", before=punctuated, after=fixed, llm=LLM)]], [[numeric_flag]]
    )
    (segment,) = document(source, final).segments
    assert [(e.stage, e.before, e.after) for e in segment.edits] == [("numeric", "十五", "五十")]
    assert segment.text[segment.flags[0].char_start : segment.flags[0].char_end] == "五十个百分点"


def test_a_manual_caption_track_is_a_source_of_its_own() -> None:
    track = CaptionTrack(
        language="en",
        kind=CaptionKind.MANUAL,
        source_format="json3",
        segments=(
            CaptionSegment(start=1.0, end=3.0, text="Markets rose"),
            CaptionSegment(start=3.0, end=5.0, text="today. Then they fell."),
        ),
    )
    source = from_captions(track, metadata("en"))
    assert source.run.kind == "caption_manual" and source.run.id == "caption_manual:en"
    doc = document(source, source.stretches, "en")
    assert [s.text for s in doc.segments] == ["Markets rose today.", "Then they fell."]
    assert all(s.words is None and s.flags == () for s in doc.segments)
    assert doc.transcript.punctuation == "caption" and doc.segments[0].start == 1.0
    assert "Transcript: manual captions" in render_markdown(doc)


def test_word_confidence_is_none_when_no_word_has_a_probability() -> None:
    plain = (Word(text="估值", start=0.0, end=1.0), Word(text="很高", start=1.0, end=2.0))
    assert word_confidence(source_of([row("估值很高", 0.0, 2.0, plain)]).stretches) is None
    assert word_confidence(source_of([row("估值很高", 0.0, 2.0)]).stretches) is None


def test_word_confidence_lists_the_probabilities_each_stretch_has() -> None:
    scored = (Word(text="估值", start=0.0, end=1.0, conf=0.9), Word(text="很高", start=1.0, end=2.0, conf=0.3))
    unscored = (Word(text="然后", start=3.0, end=4.0),)
    source = source_of([row("估值很高", 0.0, 2.0, scored), row("然后", 3.0, 4.0, unscored)])
    assert len(source.stretches) == 2
    assert word_confidence(source.stretches) == [[0.9, 0.3], None]
