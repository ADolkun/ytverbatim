from collections.abc import Callable
from pathlib import Path

import pytest

from ytverbatim.contract import ContextDocument, Edit, Flag, Segment, TranscriptRun
from ytverbatim.render import render_markdown

MakeDoc = Callable[..., ContextDocument]
MakeSeg = Callable[..., Segment]
MakeFlag = Callable[..., Flag]

GOLDEN = Path(__file__).parent.parent / "fixtures" / "render"


def zh_doc(make_doc: MakeDoc, seg: MakeSeg, flag: MakeFlag) -> ContextDocument:
    price = "今天我们聊聊英伟达NVDA，股价涨到了150美元。"
    return make_doc(
        [
            seg("大家好，欢迎收看本期节目。", 0.0, 3.4, paragraph=0, chapter=0),
            seg(
                price,
                3.4,
                9.9,
                paragraph=0,
                chapter=0,
                flags=[
                    flag(
                        "numeric_disputed",
                        price,
                        "150",
                        note="engines disagree",
                        readings=[("asr:example-engine", "150"), ("caption_auto", "115")],
                    ),
                    flag("entity_corrected", price, "英伟达", note="company name"),
                ],
                edits=[Edit(stage="correction", before="英伟大", after="英伟达", llm="example-llm")],
            ),
            seg("去年的价格也是150美元。", 65.2, 70.0, paragraph=1, chapter=1),
        ],
        language="zh",
        duration_s=125.5,
        chapters={0: "开场", 1: "财报"},
        punctuation="engine",
        correction_llm="example-llm",
    )


def en_doc(make_doc: MakeDoc, seg: MakeSeg, flag: MakeFlag) -> ContextDocument:
    first = "Revenue grew 12% to $4.2 billion in Q3."
    second = "Guidance for next_quarter stays at *flat* [sic] and A&T."
    return make_doc(
        [
            seg(
                first,
                3.0,
                8.0,
                flags=[
                    flag("numeric_single_source", first, "12%", note="one engine only"),
                    flag("numeric_single_source", first, "$4.2 billion", note="one engine only"),
                    flag("low_confidence", first, "$4.2 billion", note="low decoder confidence"),
                ],
            ),
            seg(second, 8.0, 14.5),
            seg("Thanks for watching.", 3700.0, 3703.4, paragraph=1),
        ],
        language="en",
        duration_s=3725.0,
        punctuation="llm",
    )


@pytest.mark.parametrize(("name", "build"), [("zh", zh_doc), ("en", en_doc)])
def test_golden(name: str, build: Callable[[MakeDoc, MakeSeg, MakeFlag], ContextDocument], make_doc, seg, flag) -> None:
    rendered = render_markdown(build(make_doc, seg, flag))
    assert rendered == (GOLDEN / f"{name}.md").read_text(encoding="utf-8")


def test_footnote_sits_at_the_flagged_occurrence_and_other_occurrences_stay_plain(make_doc, seg, flag) -> None:
    text = "Buy 150 now, 150 later, 150 always."
    doc = make_doc([seg(text, 0.0, 5.0, flags=[flag("numeric_disputed", text, "150", occurrence=1, note="n")])])
    body = render_markdown(doc)
    assert "Buy 150 now, 150[^1] later, 150 always." in body
    assert body.count("[^1]") == 2


def test_numbering_follows_document_order_across_segments(make_doc, seg, flag) -> None:
    first, second = "alpha beta", "gamma delta"
    doc = make_doc(
        [
            seg(first, 0.0, 1.0, flags=[flag("low_confidence", first, "beta")]),
            seg(second, 1.0, 2.0, flags=[flag("low_confidence", second, "gamma")]),
        ]
    )
    body = render_markdown(doc)
    assert "alpha beta[^1] gamma[^2] delta" in body
    assert body.index("[^1]: 00:00") < body.index("[^2]: 00:01")


def test_links_use_floored_whole_seconds(make_doc, seg) -> None:
    doc = make_doc([seg("hello", 61.999, 63.0)], duration_s=100.0)
    assert "[01:01](https://youtu.be/AAAAAAAAAAA?t=61) hello" in render_markdown(doc)


def test_no_chapters_gets_a_single_transcript_heading(make_doc, seg) -> None:
    doc = make_doc([seg("one", 0.0, 1.0), seg("two", 1.0, 2.0, paragraph=1)])
    body = render_markdown(doc)
    assert body.count("## Transcript") == 1
    assert body.count("\n## ") == 1


def test_chapter_heading_links_to_chapter_start(make_doc, seg) -> None:
    doc = make_doc([seg("one", 5.5, 9.0, chapter=0)], chapters={0: "First"})
    assert "## [00:05](https://youtu.be/AAAAAAAAAAA?t=5) First" in render_markdown(doc)


def test_hour_long_video_uses_h_mm_ss_everywhere(make_doc, seg) -> None:
    doc = make_doc([seg("late", 3725.9, 3730.0)], duration_s=3600.0)
    body = render_markdown(doc)
    assert "[1:02:05](https://youtu.be/AAAAAAAAAAA?t=3725) late" in body
    assert "· 1:00:00 ·" in body


def test_short_video_uses_mm_ss(make_doc, seg) -> None:
    doc = make_doc([seg("early", 5.0, 6.0)], duration_s=3599.9)
    body = render_markdown(doc)
    assert "[00:05]" in body
    assert "· 59:59 ·" in body


def test_zh_segments_join_without_separator_and_en_with_a_space(make_doc, seg) -> None:
    zh = render_markdown(make_doc([seg("你好。", 0.0, 1.0), seg("再见。", 1.0, 2.0)], language="zh"))
    en = render_markdown(make_doc([seg("Hello.", 0.0, 1.0), seg("Bye.", 1.0, 2.0)], language="en"))
    assert "你好。再见。" in zh
    assert "Hello. Bye." in en


def test_numeric_single_source_needs_overlapping_low_confidence(make_doc, seg, flag) -> None:
    text = "up 12 and 34 today"
    lone = flag("numeric_single_source", text, "12", note="alone")
    covered = flag("numeric_single_source", text, "34", note="covered")
    low = flag("low_confidence", text, "34")
    body = render_markdown(make_doc([seg(text, 0.0, 2.0, flags=[lone, covered, low])]))
    assert "up 12 and 34[^1][^2] today" in body
    assert "alone" not in body
    assert "numeric_single_source: covered" in body


def test_timing_interpolated_stays_in_the_json(make_doc, seg, flag) -> None:
    text = "Prices rose 12 today"
    timing = flag("timing_interpolated", text, text, note="interpolated")
    disputed = flag("numeric_disputed", text, "12", note="heard 20")
    doc = make_doc([seg(text, 0.0, 2.0, flags=[timing, disputed])])
    body = render_markdown(doc)
    assert "timing_interpolated" not in body and "interpolated" not in body
    assert "Prices rose 12[^1] today" in body and "1 flag" in body
    assert doc.segments[0].flags == (timing, disputed)


def test_numeric_single_source_ignores_low_confidence_elsewhere_in_segment(make_doc, seg, flag) -> None:
    text = "up 12 and shaky speech"
    lone = flag("numeric_single_source", text, "12", note="alone")
    low = flag("low_confidence", text, "shaky")
    body = render_markdown(make_doc([seg(text, 0.0, 2.0, flags=[lone, low])]))
    assert "up 12 and shaky[^1] speech" in body
    assert "alone" not in body


def test_entity_corrected_and_numeric_fixed_footnotes_show_the_before_text(make_doc, seg, flag) -> None:
    text = "Nvidia hit 150 today"
    edits = [
        Edit(stage="correction", before="Envidia", after="Nvidia", llm=None),
        Edit(stage="numeric", before="115", after="150", llm=None),
    ]
    flags = [flag("entity_corrected", text, "Nvidia", note="ticker"), flag("numeric_fixed", text, "150", note="n")]
    body = render_markdown(make_doc([seg(text, 0.0, 2.0, flags=flags, edits=edits)]))
    assert "00:00 — entity_corrected: ticker; was Envidia" in body
    assert "00:00 — numeric_fixed: n; was 115" in body


def test_footnote_lists_readings_with_sources(make_doc, seg, flag) -> None:
    text = "price is 150"
    f = flag("numeric_disputed", text, "150", note="n", readings=[("asr:x", "150"), ("caption_auto", "115")])
    body = render_markdown(make_doc([seg(text, 0.0, 1.0, flags=[f])]))
    assert "[^1]: 00:00 — numeric_disputed: n; heard 150 (asr:x) / 115 (caption_auto)" in body


def test_overlapping_flags_keep_text_intact_and_order_deterministically(make_doc, seg, flag) -> None:
    text = "the quick brown fox"
    flags = [
        flag("low_confidence", text, "quick brown", note="b"),
        flag("decoder_anomaly", text, "quick", note="a"),
        flag("entity_corrected", text, "brown fox", note="c"),
        flag("low_confidence", text, "quick brown", note="a"),
    ]
    doc = make_doc([seg(text, 0.0, 2.0, flags=flags)])
    body = render_markdown(doc)
    assert "the quick[^1] brown[^2][^3] fox[^4]" in body
    shuffled = doc.segments[0].model_copy(update={"flags": tuple(reversed(flags))})
    assert render_markdown(doc.model_copy(update={"segments": (shuffled,)})) == body
    footnotes = [line for line in body.splitlines() if line.startswith("[^")]
    assert footnotes == [
        "[^1]: 00:00 — decoder_anomaly: a",
        "[^2]: 00:00 — low_confidence: a",
        "[^3]: 00:00 — low_confidence: b",
        "[^4]: 00:00 — entity_corrected: c",
    ]


def test_flag_touching_the_segment_end_and_adjacent_segment(make_doc, seg, flag) -> None:
    first, second = "ends here", "starts there"
    doc = make_doc(
        [
            seg(first, 0.0, 1.0, flags=[flag("low_confidence", first, "here")]),
            seg(second, 1.0, 2.0, flags=[flag("low_confidence", second, "starts")]),
        ]
    )
    assert "ends here[^1] starts[^2] there" in render_markdown(doc)


def test_escaping_is_minimal_and_leaves_cjk_tickers_and_numbers_alone(make_doc, seg) -> None:
    text = "英伟达NVDA涨到$1,150.50（+3.2%）, AT&T, snake_case, 2*3 [x] <b> `c` a_ _b \\ &amp;"
    body = render_markdown(make_doc([seg(text, 0.0, 1.0)]))
    expected = (
        "英伟达NVDA涨到$1,150.50（+3.2%）, AT&T, snake_case, 2\\*3 \\[x\\] \\<b\\> \\`c\\` a\\_ \\_b \\\\ \\&amp;"
    )
    assert expected in body


def test_tildes_cannot_strike_text_through(make_doc, seg) -> None:
    body = render_markdown(make_doc([seg("涨幅~~3%~~ and ~5~ points", 0.0, 1.0)]))
    assert "涨幅\\~\\~3%\\~\\~ and \\~5\\~ points" in body


def test_newlines_in_text_do_not_start_block_syntax(make_doc, seg, flag) -> None:
    text = "intro\n# not a heading\n- not a list"
    f = flag("low_confidence", text, "heading")
    body = render_markdown(make_doc([seg(text, 0.0, 1.0, flags=[f])]))
    assert "intro # not a heading[^1] - not a list" in body


def test_title_and_chapter_titles_are_escaped(make_doc, seg) -> None:
    doc = make_doc([seg("x", 0.0, 1.0, chapter=0)], chapters={0: "Q&A [live]"})
    body = render_markdown(doc)
    assert "Q&A \\[live\\]" in body


def test_provenance_line_labels(make_doc, seg, flag) -> None:
    text = "a b"
    manual = TranscriptRun(
        id="cap-1",
        kind="caption_manual",
        engine=None,
        model_repo=None,
        model_revision=None,
        params={},
        vad=None,
        chunking=None,
        aligner=None,
    )
    doc = make_doc([seg(text, 0.0, 1.0, flags=[flag("low_confidence", text, "b")])], run=manual)
    assert "Transcript: manual captions · punctuation: engine · corrected by: none · 1 flag\n" in render_markdown(doc)


def test_header_omits_unknown_upload_date(make_doc, seg) -> None:
    body = render_markdown(make_doc([seg("x", 0.0, 1.0)], upload_date=None))
    assert "Example Channel · 10:00 · https://youtu.be/AAAAAAAAAAA" in body


def test_document_without_segments_renders_header_only(make_doc) -> None:
    body = render_markdown(make_doc([]))
    assert body.startswith("# Example Title\n")
    assert body.rstrip().endswith("## Transcript")
    assert "0 flags" in body
