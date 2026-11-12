import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from ytverbatim.acquire.creator_note import CreatorNote
from ytverbatim.acquire.metadata import Chapter, VideoMetadata
from ytverbatim.asr import registry
from ytverbatim.asr.models import ChunkHypothesis
from ytverbatim.asr.protocol import ChunkFlag
from ytverbatim.contract import Access, ContextDocument, Word, dump_document, load_document
from ytverbatim.errors import ContractError
from ytverbatim.render import render_markdown
from ytverbatim.stages import build_document, from_asr_rows

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


def metadata(*, language: str | None = "en", chapters: Sequence[Chapter] = (), duration: float | None = 100.0):
    return VideoMetadata(
        video_id="synthVid001",
        title="Synthetic title",
        channel="Synthetic channel",
        channel_id="UCsynthetic",
        upload_date="20300101",
        duration=duration,
        description="",
        language=language,
        availability="public",
        live_status=None,
        chapters=tuple(chapters),
        subtitle_langs=(),
        auto_caption_lang=None,
    )


def words(pieces: Sequence[str], start: float = 0.0, step: float = 0.4, gaps: dict[int, float] | None = None):
    out: list[Word] = []
    clock = start
    for index, piece in enumerate(pieces):
        out.append(Word(text=piece, start=clock, end=clock + step))
        clock += step + (gaps or {}).get(index, 0.0)
    return tuple(out)


def row(
    chunk: int,
    text: str,
    start: float,
    end: float,
    *,
    hypothesis: str = "w-prod",
    language: str | None = "en",
    chunk_words: tuple[Word, ...] = (),
    flags: tuple[ChunkFlag, ...] = (),
) -> ChunkHypothesis:
    return ChunkHypothesis(
        hypothesis=hypothesis,
        file="synthVid001",
        chunk=chunk,
        start=start,
        end=end,
        text=text,
        language=language,
        words=chunk_words,
        flags=flags,
    )


def build(
    rows: Sequence[ChunkHypothesis],
    *,
    engine: str = "w-prod",
    chapters: bool = True,
    meta: VideoMetadata | None = None,
) -> ContextDocument:
    meta = meta or metadata()
    source = from_asr_rows(rows, registry.get(engine), meta)
    return build_document(
        metadata=meta,
        tags=("synthetic",),
        access=ACCESS,
        source=source,
        stretches=source.stretches,
        chapters=chapters,
        created_at=NOW,
    )


def test_whisper_words_give_word_level_segments_and_a_valid_document() -> None:
    pieces = [" The", " market", " rose", " today.", " Then", " it", " fell", " sharply."]
    doc = build([row(0, "".join(pieces), 0.0, 5.0, chunk_words=words(pieces, 0.5, 0.5))])

    assert [s.text for s in doc.segments] == ["The market rose today.", "Then it fell sharply."]
    first, second = doc.segments
    assert (first.start, first.end, second.start, second.end) == (0.5, 2.5, 2.5, 4.5)
    assert first.words is not None and [w.text for w in first.words] == ["The", "market", "rose", "today."]
    assert first.words[0].start == 0.5 and first.words[-1].end == pytest.approx(2.5)
    assert doc.transcript.punctuation == "engine" and doc.transcript.language == "en"
    assert doc.transcript.script == "n/a" and doc.transcript.correction_llm is None
    assert doc.quality.warnings == ()
    assert doc.chapters == () and [s.chapter for s in doc.segments] == [None, None]
    again = load_document(dump_document(doc))
    assert again == doc
    assert "The market rose today." in render_markdown(doc)


def test_run_records_the_engine_and_a_whole_file_run_has_no_vad_or_chunking() -> None:
    doc = build([row(0, " Hi.", 0, 1, chunk_words=words([" Hi."]))])
    (run,) = doc.transcript.runs
    assert run.engine == "w-prod" and run.kind == "asr" and run.model_repo and run.model_revision
    assert run.vad is None and run.chunking is None
    assert run.params["mode"] == "whole_file" and run.params["language_hint"] == "en"
    assert {s.source.run for s in doc.segments} == {run.id}


def test_a_chunked_engine_records_vad_and_chunking() -> None:
    rows = [row(0, "Hello.", 0, 4, hypothesis="q-base"), row(1, "World.", 5, 9, hypothesis="q-base")]
    doc = build(rows, engine="q-base")
    (run,) = doc.transcript.runs
    assert run.vad is not None and run.vad["threshold"] == 0.5 and "revision" in run.vad
    assert run.chunking is not None and run.chunking["max_chunk_s"] == 30.0 and run.chunking["chunks"] == 2
    assert doc.quality.speech_seconds == pytest.approx(8.0)


def test_engines_without_word_times_get_chunk_level_timing_and_a_warning() -> None:
    text = "Alpha beta gamma. Delta epsilon zeta eta."
    doc = build([row(0, text, 10.0, 20.0, hypothesis="q-base")], engine="q-base")
    first, second = doc.segments
    assert first.words is None and second.words is None
    assert first.start == 10.0 and second.end == pytest.approx(20.0)
    # 15 of the 35 non-space characters belong to the first sentence.
    assert first.end == pytest.approx(10.0 + 10.0 * 15 / 35, abs=1e-3)
    assert second.start == pytest.approx(first.end)
    assert len(doc.quality.warnings) == 1 and "chunk-level" in doc.quality.warnings[0]
    assert doc.segments[0].flags == ()


def test_words_missing_from_the_end_are_interpolated_up_to_the_chunk_end() -> None:
    doc = build([row(0, "Alpha beta gamma.", 0, 3, chunk_words=words([" Alpha", " beta"]))])
    (segment,) = doc.segments
    assert segment.text == "Alpha beta gamma." and (segment.start, segment.end) == (0.0, 3.0)
    assert segment.words is not None and [(w.text, w.start, w.end) for w in segment.words] == [
        ("Alpha", 0.0, 0.4),
        ("beta", 0.4, 0.8),
    ]
    (flag,) = segment.flags
    assert (flag.kind, flag.char_start, flag.char_end) == ("timing_interpolated", 0, len(segment.text))
    assert any("did not cover" in warning for warning in doc.quality.warnings)


def test_one_dropped_word_mid_file_keeps_every_other_time_exact() -> None:
    pieces = [" The", " market", " rose", " today.", " Then", " it", " fell", " sharply.", " Prices", " recovered."]
    ws = words(pieces, 1.0, 0.5)
    kept = ws[:5] + ws[6:]  # the engine's words lack " it"
    doc = build([row(0, "".join(pieces), 0.0, 100.0, chunk_words=kept)])

    assert [s.text for s in doc.segments] == ["The market rose today.", "Then it fell sharply.", "Prices recovered."]
    timed = {w.text: (w.start, w.end) for s in doc.segments for w in s.words or ()}
    assert timed == {w.text.strip(): (w.start, w.end) for w in kept}
    first, second, third = doc.segments
    assert first.flags == () and third.flags == ()
    assert [f.kind for f in second.flags] == ["timing_interpolated"]
    assert second.words is not None and [w.text for w in second.words] == ["Then", "fell", "sharply."]
    assert (second.start, second.end) == (ws[4].start, ws[7].end)


def test_words_that_match_nothing_still_fall_back_to_chunk_level_timing() -> None:
    text = "Alpha beta gamma. Delta epsilon."
    doc = build([row(0, text, 10.0, 20.0, chunk_words=words([" Something", " else"], 10.0))])
    first, second = doc.segments
    assert first.words is None and second.words is None
    assert (first.start, second.end) == (10.0, 20.0)
    assert first.end == pytest.approx(10.0 + 10.0 * 15 / 28, abs=1e-3)
    assert all([f.kind for f in s.flags] == ["timing_interpolated"] for s in doc.segments)


def test_word_times_are_clamped_to_their_chunk() -> None:
    first = (Word(text=" One", start=0.5, end=1.0), Word(text=" two.", start=999.0, end=999.5))
    second = words([" Three", " four."], 11.0, 0.5)
    doc = build(
        [
            row(0, " One two.", 0, 10, hypothesis="w-vad", chunk_words=first),
            row(1, " Three four.", 10, 20, hypothesis="w-vad", chunk_words=second),
        ],
        engine="w-vad",
    )
    one, three = doc.segments
    assert one.words is not None and one.words[1].start == 10.0 and one.end == 10.0
    assert three.words is not None and [(w.start, w.end) for w in three.words] == [(11.0, 11.5), (11.5, 12.0)]


def test_a_chapter_that_ends_before_it_starts_is_a_contract_error() -> None:
    pieces = [" One.", " Two."]
    with pytest.raises(ContractError, match="synthVid001"):
        build(
            [row(0, "".join(pieces), 0, 3, chunk_words=words(pieces))],
            meta=metadata(chapters=[Chapter("Backwards", 5.0, 1.0)]),
        )


def test_zh_tagged_english_joins_chunks_with_a_space_and_uses_english_rules() -> None:
    rows = [
        row(0, "The market rose today.", 0, 4, hypothesis="q-base", language="zh"),
        row(1, "Then it fell.", 4.2, 9, hypothesis="q-base", language="zh"),
    ]
    doc = build(rows, engine="q-base", meta=metadata(language="zh"))
    assert [s.text for s in doc.segments] == ["The market rose today.", "Then it fell."]
    assert doc.transcript.punctuation == "engine"


def test_mandarin_without_punctuation_is_cut_at_pauses_and_flagged_unpunctuated() -> None:
    pieces = list("今天我们聊聊财报") + list("营收增长很快") + list("估值也不低")
    text = "".join(pieces)
    doc = build(
        [row(0, text, 0, 12, language="zh", chunk_words=words(pieces, 0, 0.25, {7: 0.8, 13: 1.0}))],
        meta=metadata(language="zh"),
    )
    assert [s.text for s in doc.segments] == ["今天我们聊聊财报", "营收增长很快", "估值也不低"]
    assert doc.transcript.punctuation == "none" and doc.transcript.language == "zh"
    assert doc.transcript.script == "zh-Hans"
    assert all(s.words is not None for s in doc.segments)


def test_traditional_characters_are_recorded_not_converted() -> None:
    doc = build([row(0, "這個軟體很好。", 0, 3, language="zh", hypothesis="q-base")], engine="q-base")
    assert doc.transcript.script == "zh-Hant"
    assert doc.segments[0].text == "這個軟體很好。"
    assert any("Traditional" in warning for warning in doc.quality.warnings)


def test_chunks_join_into_one_sentence_when_the_cut_fell_mid_sentence() -> None:
    rows = [
        row(0, "The market rose", 0, 4, hypothesis="q-base"),
        row(1, "sharply today. Then it fell.", 4.2, 9, hypothesis="q-base"),
    ]
    doc = build(rows, engine="q-base")
    assert [s.text for s in doc.segments] == ["The market rose sharply today.", "Then it fell."]


def test_decoder_anomaly_flags_the_segments_of_that_chunk_only() -> None:
    rows = [
        row(0, "First chunk here. It is fine.", 0, 6, hypothesis="q-base"),
        row(1, "Second chunk loops. Loops again.", 7, 13, hypothesis="q-base", flags=("decoder_anomaly", "forced_cut")),
        row(2, "Third chunk is fine.", 14, 18, hypothesis="q-base", flags=("token_cap_hit",)),
    ]
    doc = build(rows, engine="q-base")
    flagged = {s.text: [f.kind for f in s.flags] for s in doc.segments}
    assert flagged == {
        "First chunk here.": [],
        "It is fine.": [],
        "Second chunk loops.": ["decoder_anomaly"],
        "Loops again.": ["decoder_anomaly"],
        "Third chunk is fine.": [],
    }
    loops = doc.segments[2]
    assert (loops.flags[0].char_start, loops.flags[0].char_end) == (0, len(loops.text))
    assert "0:07-0:13" in loops.flags[0].note
    assert doc.quality.decoder_anomalies == 1
    # The document's flags by kind (two segments carry the anomaly), plus the chunk diagnostics.
    assert doc.quality.flags_by_kind == {"decoder_anomaly": 2, "forced_cut": 1, "token_cap_hit": 1}
    assert "decoder_anomaly" in render_markdown(doc)


def test_a_flag_covers_only_the_anomalous_chunks_part_of_a_shared_sentence() -> None:
    rows = [
        row(0, "The market rose", 0, 4, hypothesis="q-base"),
        row(1, "sharply today.", 4.2, 8, hypothesis="q-base", flags=("decoder_anomaly",)),
    ]
    (segment,) = build(rows, engine="q-base").segments
    (flag,) = segment.flags
    assert segment.text[flag.char_start : flag.char_end] == "sharply today."


def test_empty_chunks_are_counted_but_leave_no_segment() -> None:
    rows = [row(0, "", 0, 3), row(1, "Hello there.", 3, 6, hypothesis="q-base")]
    doc = build(rows, engine="q-base")
    assert [s.text for s in doc.segments] == ["Hello there."]
    assert doc.quality.flags_by_kind["empty_output"] == 1


def test_a_silent_video_still_yields_a_valid_empty_document() -> None:
    doc = build([row(0, "", 0, 60)], meta=metadata(chapters=[Chapter("One", 0, 60)]))
    assert doc.segments == () and doc.paragraphs == () and doc.chapters == ()
    assert "no speech" in doc.quality.warnings[0]
    assert load_document(dump_document(doc)) == doc
    assert render_markdown(doc).startswith("# Synthetic title")


def test_paragraphs_break_at_pauses_and_chapters_map_by_start_time() -> None:
    pieces = [" One.", " Two.", " Three.", " Four.", " Five."]
    ws = words(pieces, 0.0, 0.5, gaps={0: 0.1, 1: 2.5, 2: 0.1, 3: 0.1})
    chapters = [Chapter("Intro", 0.0, 3.0), Chapter("Silent", 3.0, 3.5), Chapter("Main", 3.5, 20.0)]
    doc = build([row(0, "".join(pieces), 0, 10, chunk_words=ws)], meta=metadata(chapters=chapters))

    assert [s.paragraph for s in doc.segments] == [0, 0, 1, 1, 1]
    assert [(c.title, c.first_segment, c.last_segment, c.source) for c in doc.chapters] == [
        ("Intro", 0, 1, "youtube"),
        ("Main", 2, 4, "youtube"),
    ]
    assert [c.index for c in doc.chapters] == [0, 1]
    assert [s.chapter for s in doc.segments] == [0, 0, 1, 1, 1]
    assert "## [00:00](https://youtu.be/synthVid001?t=0) Intro" in render_markdown(doc)


def test_a_paragraph_never_crosses_a_chapter_boundary() -> None:
    pieces = [" One.", " Two.", " Three.", " Four."]
    ws = words(pieces, 0.0, 0.5, gaps={0: 0.1, 1: 0.1, 2: 0.1})
    doc = build(
        [row(0, "".join(pieces), 0, 10, chunk_words=ws)],
        meta=metadata(chapters=[Chapter("A", 0.0, 1.0), Chapter("B", 1.0, 10.0)]),
    )
    assert [s.chapter for s in doc.segments] == [0, 0, 1, 1]
    assert [s.paragraph for s in doc.segments] == [0, 0, 1, 1]


def test_chapters_can_be_turned_off() -> None:
    pieces = [" One.", " Two."]
    doc = build(
        [row(0, "".join(pieces), 0, 3, chunk_words=words(pieces))],
        meta=metadata(chapters=[Chapter("A", 0.0, 3.0)]),
        chapters=False,
    )
    assert doc.chapters == () and [s.chapter for s in doc.segments] == [None, None]


def test_non_monotonic_word_starts_are_clamped_not_rejected() -> None:
    ws = (
        Word(text=" One", start=1.0, end=1.5),
        Word(text=" two.", start=0.9, end=1.4),
        Word(text=" Three.", start=2.0, end=2.4),
    )
    doc = build([row(0, " One two. Three.", 0, 3, chunk_words=ws)])
    assert [s.start for s in doc.segments] == [1.0, 2.0]
    assert doc.segments[0].words is not None and doc.segments[0].words[1].start == 1.0


def test_language_comes_from_the_engine_else_the_declared_language_else_und() -> None:
    assert build([row(0, "Hi.", 0, 1, language="de", hypothesis="q-base")], engine="q-base").transcript.language == "de"
    declared = build([row(0, "Hi.", 0, 1, language=None, hypothesis="q-base")], engine="q-base", meta=metadata())
    assert declared.transcript.language == "en"
    unknown = build(
        [row(0, "Hi.", 0, 1, language=None, hypothesis="q-base")], engine="q-base", meta=metadata(language=None)
    )
    assert unknown.transcript.language == "und"


def test_video_and_quality_fields() -> None:
    pieces = [" Hello", " there."]
    doc = build([row(0, "".join(pieces), 0, 4, chunk_words=words(pieces, 1.0, 0.5))])
    video = doc.video
    assert (video.id, video.url, video.title, video.channel) == (
        "synthVid001",
        "https://www.youtube.com/watch?v=synthVid001",
        "Synthetic title",
        "Synthetic channel",
    )
    assert video.upload_date is not None and video.upload_date.isoformat() == "2030-01-01"
    assert video.tags == ("synthetic",) and video.duration_s == 100.0
    assert doc.quality.transcribed_seconds == pytest.approx(1.0)
    assert doc.quality.speech_seconds == doc.quality.transcribed_seconds
    assert doc.generator.name == "ytverbatim" and doc.created_at == NOW


def test_an_unparseable_upload_date_becomes_none() -> None:
    meta = metadata()
    bad = VideoMetadata(**{**meta.__dict__, "upload_date": "soon", "duration": None})
    doc = build([row(0, "Hi.", 0, 7, hypothesis="q-base")], engine="q-base", meta=bad)
    assert doc.video.upload_date is None and doc.video.duration_s == 7.0


def test_the_library_surface_is_exported_lazily() -> None:
    import ytverbatim

    assert ytverbatim.process.__module__ == "ytverbatim.pipeline"
    assert ytverbatim.ContextDocument is ContextDocument
    assert ytverbatim.Options.__module__ == "ytverbatim.config"
    assert ytverbatim.LLMClient.__module__ == "ytverbatim.llm.protocol"
    with pytest.raises(AttributeError):
        _ = ytverbatim.nonexistent  # type: ignore[attr-defined]
    assert {"process", "Options", "ContextDocument", "LLMClient", "__version__"} <= set(dir(ytverbatim))
    assert "process" in vars(ytverbatim), "a resolved name is cached in the module"


def test_importing_the_package_loads_neither_the_pipeline_nor_its_heavy_dependencies() -> None:
    probe = "import sys, ytverbatim; print(sorted({'ytverbatim.pipeline', 'yt_dlp', 'pydantic'} & set(sys.modules)))"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_correction_context_caps_the_glossary_at_twenty_terms_and_carries_the_creator_note() -> None:
    from ytverbatim.pipeline import correction_context

    note = CreatorNote(text="市盈率 not 适应率", author="Example Channel", like_count=3, published="20300101")
    tags = tuple(f"term{i}" for i in range(25))
    context = correction_context(replace(metadata(), creator_note=note), tags)
    assert context.glossary == tags[:20]
    assert context.creator_note == "市盈率 not 适应率"
    bare = correction_context(metadata(), ())
    assert (bare.glossary, bare.creator_note) == ((), None)


def test_info_tags_keeps_only_string_tags_of_a_raw_extraction() -> None:
    from ytverbatim.pipeline import info_tags

    assert info_tags({"tags": ["GAAP", 3, "市盈率", None]}) == ("GAAP", "市盈率")
    assert info_tags({"tags": "GAAP"}) == ()
    assert info_tags({}) == ()
    assert info_tags(["tags"]) == ()
