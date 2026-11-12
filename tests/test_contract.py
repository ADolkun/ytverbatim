import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ytverbatim.contract import (
    SCHEMA_PATH,
    ContextDocument,
    dump_document,
    export_schema,
    load_document,
)
from ytverbatim.errors import ContractError

Raw = dict[str, Any]

RUN = {
    "id": "asr-1",
    "kind": "asr",
    "engine": "example-engine",
    "model_repo": "example/model",
    "model_revision": None,
    "params": {"temperature": [0.0, 0.2], "beam": 1, "alpha": {"z": 1, "a": 2}},
    "vad": None,
    "chunking": {"max_s": 30},
    "aligner": None,
}


def _segment(index: int, text: str, start: float, end: float, *, paragraph: int, chapter: int | None) -> Raw:
    return {
        "id": index,
        "start": start,
        "end": end,
        "text": text,
        "words": None,
        "source": {"kind": "asr", "run": "asr-1"},
        "edits": [],
        "flags": [],
        "paragraph": paragraph,
        "chapter": chapter,
    }


def make_raw() -> Raw:
    segments = [
        _segment(0, "大家好，欢迎收看。", 0.0, 2.5, paragraph=0, chapter=0),
        _segment(1, "Today we cover NVDA at 150 dollars.", 2.5, 6.0, paragraph=0, chapter=0),
        _segment(2, "好的，我们开始。", 6.0, 8.2, paragraph=1, chapter=1),
    ]
    segments[1]["words"] = [
        {"text": "Today", "start": 2.5, "end": 2.9, "conf": 0.98},
        {"text": "we", "start": 2.9, "end": 3.0, "conf": None},
    ]
    segments[1]["edits"] = [{"stage": "correction", "before": "Envidia", "after": "NVDA", "llm": "example-llm"}]
    segments[1]["flags"] = [
        {
            "kind": "numeric_disputed",
            "char_start": 21,
            "char_end": 24,
            "note": "engines disagree",
            "readings": [
                {"source": "asr:example-engine", "text": "150"},
                {"source": "caption_auto", "text": "115"},
            ],
        }
    ]
    return {
        "schema_version": "1.0",
        "generator": {"name": "ytverbatim", "version": "0.0.1"},
        "created_at": "2026-01-02T03:04:05+00:00",
        "video": {
            "id": "AAAAAAAAAAA",
            "url": "https://www.youtube.com/watch?v=AAAAAAAAAAA",
            "title": "Example Title",
            "channel": "Example Channel",
            "channel_id": "UCexample",
            "upload_date": "2026-01-01",
            "duration_s": 8.2,
            "description": "An example video.",
            "language": "zh",
            "tags": ["example"],
        },
        "access": {
            "auth_source": "none",
            "cookies_used": False,
            "logged_in": None,
            "premium": None,
            "member": None,
            "availability": "public",
            "ytdlp_version": "2026.10.26",
            "warnings": ["no JS runtime found"],
        },
        "transcript": {
            "language": "zh+en",
            "script": "zh-Hans",
            "runs": [RUN],
            "punctuation": "engine",
            "correction_llm": None,
        },
        "chapters": [
            {
                "index": 0,
                "title": "Intro",
                "start": 0.0,
                "end": 6.0,
                "source": "youtube",
                "first_segment": 0,
                "last_segment": 1,
            },
            {
                "index": 1,
                "title": "Main",
                "start": 6.0,
                "end": 8.2,
                "source": "youtube",
                "first_segment": 2,
                "last_segment": 2,
            },
        ],
        "paragraphs": [
            {"index": 0, "start": 0.0, "end": 6.0, "first_segment": 0, "last_segment": 1},
            {"index": 1, "start": 6.0, "end": 8.2, "first_segment": 2, "last_segment": 2},
        ],
        "segments": segments,
        "quality": {
            "speech_seconds": 7.5,
            "transcribed_seconds": 8.2,
            "decoder_anomalies": 0,
            "flags_by_kind": {"numeric_disputed": 1},
            "warnings": [],
        },
        "llm_usage": [
            {"stage": "correction", "client": "example", "calls": 1, "input_tokens": 100, "output_tokens": None}
        ],
    }


def test_checked_in_schema_matches_export() -> None:
    checked_in = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert checked_in == export_schema(), (
        "context.v1.schema.json is stale; regenerate it with: uv run python -c "
        "'import json; from ytverbatim.contract import SCHEMA_PATH, export_schema; "
        'SCHEMA_PATH.write_text(json.dumps(export_schema(), indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")\''
    )


def test_round_trip_is_stable() -> None:
    doc = ContextDocument.model_validate(make_raw())
    text = dump_document(doc)
    assert text.endswith("}\n")
    assert "大家好，欢迎收看。" in text
    reloaded = load_document(text)
    assert reloaded == doc
    assert dump_document(reloaded) == text


def test_round_trip_through_a_file(tmp_path: Path) -> None:
    doc = ContextDocument.model_validate(make_raw())
    path = tmp_path / "context.json"
    path.write_text(dump_document(doc), encoding="utf-8")
    assert load_document(path) == doc


def test_dump_key_order_is_stable() -> None:
    raw = make_raw()
    doc = ContextDocument.model_validate(raw)
    text = dump_document(doc)
    assert list(json.loads(text))[:3] == ["schema_version", "generator", "created_at"]
    run = json.loads(text)["transcript"]["runs"][0]
    assert list(run["params"]) == ["alpha", "beam", "temperature"]
    assert list(run["params"]["alpha"]) == ["a", "z"]

    shuffled = deepcopy(raw)
    shuffled["transcript"]["runs"][0]["params"] = dict(reversed(list(RUN["params"].items())))
    assert dump_document(ContextDocument.model_validate(shuffled)) == text


def test_times_round_to_milliseconds() -> None:
    raw = make_raw()
    raw["segments"][0]["end"] = 2.50049
    raw["segments"][1]["start"] = 2.5004
    doc = ContextDocument.model_validate(raw)
    assert doc.segments[0].end == 2.5
    assert doc.segments[1].start == 2.5


def test_words_may_be_absent_without_a_flag() -> None:
    doc = ContextDocument.model_validate(make_raw())
    assert doc.segments[0].words is None
    assert doc.segments[0].flags == ()


def test_no_chapters_is_valid() -> None:
    raw = make_raw()
    raw["chapters"] = []
    for segment in raw["segments"]:
        segment["chapter"] = None
    assert ContextDocument.model_validate(raw).chapters == ()


def test_empty_document_is_valid() -> None:
    raw = make_raw()
    raw.update(chapters=[], paragraphs=[], segments=[])
    assert ContextDocument.model_validate(raw).segments == ()


def test_documents_are_frozen() -> None:
    doc = ContextDocument.model_validate(make_raw())
    with pytest.raises(ValidationError):
        doc.video.__setattr__("title", "other")


def _bad_id(raw: Raw) -> None:
    raw["segments"][1]["id"] = 5


def _bad_first_id(raw: Raw) -> None:
    for segment in raw["segments"]:
        segment["id"] += 1


def _segment_ends_before_start(raw: Raw) -> None:
    raw["segments"][1]["end"] = 1.0


def _segments_go_backwards(raw: Raw) -> None:
    raw["segments"][2]["start"] = 3.0
    raw["segments"][2]["end"] = 4.0


def _word_goes_backwards(raw: Raw) -> None:
    raw["segments"][1]["words"][1]["start"] = 2.0
    raw["segments"][1]["words"][1]["end"] = 2.1


def _word_ends_before_start(raw: Raw) -> None:
    raw["segments"][1]["words"][0]["end"] = 1.0


def _flag_past_text(raw: Raw) -> None:
    raw["segments"][1]["flags"][0]["char_end"] = 999


def _flag_empty_span(raw: Raw) -> None:
    raw["segments"][1]["flags"][0]["char_end"] = 21


def _paragraph_gap(raw: Raw) -> None:
    raw["paragraphs"][1]["first_segment"] = 2
    raw["paragraphs"][0]["last_segment"] = 0
    raw["segments"][1]["paragraph"] = 0


def _paragraph_overlap(raw: Raw) -> None:
    raw["paragraphs"][1]["first_segment"] = 1


def _paragraph_short_of_the_end(raw: Raw) -> None:
    raw["paragraphs"].pop()


def _paragraph_index_skips(raw: Raw) -> None:
    raw["paragraphs"][1]["index"] = 2


def _segment_in_wrong_paragraph(raw: Raw) -> None:
    raw["segments"][2]["paragraph"] = 0


def _paragraph_reference_out_of_range(raw: Raw) -> None:
    raw["segments"][2]["paragraph"] = 9


def _chapter_overlap(raw: Raw) -> None:
    raw["chapters"][1]["first_segment"] = 1


def _chapter_gap(raw: Raw) -> None:
    raw["chapters"][0]["last_segment"] = 0


def _chapters_without_chapter_refs(raw: Raw) -> None:
    for segment in raw["segments"]:
        segment["chapter"] = None


def _chapter_refs_without_chapters(raw: Raw) -> None:
    raw["chapters"] = []


def _segment_in_wrong_chapter(raw: Raw) -> None:
    raw["segments"][1]["chapter"] = 1


def _paragraphs_without_segments(raw: Raw) -> None:
    raw.update(chapters=[], segments=[])


def _unknown_run(raw: Raw) -> None:
    raw["segments"][0]["source"]["run"] = "missing"


def _run_kind_mismatch(raw: Raw) -> None:
    raw["segments"][0]["source"]["kind"] = "caption_auto"


def _duplicate_run(raw: Raw) -> None:
    raw["transcript"]["runs"].append(deepcopy(RUN))


def _macos_home_path(raw: Raw) -> None:
    raw["access"]["warnings"] = ["cookie db at /Users/example/Library/Cookies"]


def _linux_home_path(raw: Raw) -> None:
    raw["access"]["warnings"] = ["failed reading /home/example/.config/chromium"]


def _bare_home_path(raw: Raw) -> None:
    raw["access"]["warnings"] = ["cannot open /home/example"]


def _unknown_field(raw: Raw) -> None:
    raw["video"]["surprise"] = 1


def _naive_timestamp(raw: Raw) -> None:
    raw["created_at"] = "2026-01-02T03:04:05"


def _negative_time(raw: Raw) -> None:
    raw["segments"][0]["start"] = -1.0


def _non_finite_time(raw: Raw) -> None:
    raw["video"]["duration_s"] = float("inf")


REJECTED: dict[str, tuple[Callable[[Raw], None], str]] = {
    "dense ids": (_bad_id, "dense"),
    "ids start at zero": (_bad_first_id, "dense"),
    "segment reversed": (_segment_ends_before_start, "before it starts"),
    "segments monotonic": (_segments_go_backwards, "monotonic"),
    "words monotonic": (_word_goes_backwards, "starts before the word"),
    "word reversed": (_word_ends_before_start, "before it starts"),
    "flag past text": (_flag_past_text, "past the"),
    "flag empty span": (_flag_empty_span, "empty or reversed"),
    "paragraph gap": (_paragraph_gap, "uncovered"),
    "paragraph overlap": (_paragraph_overlap, "overlaps"),
    "paragraph short": (_paragraph_short_of_the_end, "ranges"),
    "paragraph index skips": (_paragraph_index_skips, "indexes"),
    "segment paragraph mismatch": (_segment_in_wrong_paragraph, "does not contain"),
    "paragraph ref out of range": (_paragraph_reference_out_of_range, "does not contain"),
    "chapter overlap": (_chapter_overlap, "overlaps"),
    "chapter gap": (_chapter_gap, "uncovered"),
    "chapter ref missing": (_chapters_without_chapter_refs, "does not contain"),
    "chapter ref without chapters": (_chapter_refs_without_chapters, "has none"),
    "segment chapter mismatch": (_segment_in_wrong_chapter, "does not contain"),
    "paragraphs without segments": (_paragraphs_without_segments, "no segments"),
    "unknown run": (_unknown_run, "absent from transcript.runs"),
    "run kind mismatch": (_run_kind_mismatch, "is caption_auto but run"),
    "duplicate run": (_duplicate_run, "duplicate transcript run"),
    "macOS home path": (_macos_home_path, "home path"),
    "linux home path": (_linux_home_path, "home path"),
    "bare home path": (_bare_home_path, "home path"),
    "unknown field": (_unknown_field, "surprise"),
    "naive timestamp": (_naive_timestamp, "created_at"),
    "negative time": (_negative_time, "start"),
    "non-finite time": (_non_finite_time, "duration_s"),
}


@pytest.mark.parametrize("name", REJECTED)
def test_validator_rejects(name: str) -> None:
    mutate, expected = REJECTED[name]
    raw = make_raw()
    mutate(raw)
    with pytest.raises(ValidationError, match=expected):
        ContextDocument.model_validate(raw)


def test_load_wraps_validation_errors() -> None:
    raw = make_raw()
    raw["segments"][1]["id"] = 7
    with pytest.raises(ContractError, match="dense"):
        load_document(json.dumps(raw))


def test_url_paths_are_not_mistaken_for_home_paths() -> None:
    raw = make_raw()
    raw["access"]["warnings"] = ["see https://example.com/home/user/ and ~/notes and /Users"]
    assert ContextDocument.model_validate(raw).access.warnings


@pytest.mark.parametrize("version", ["2.0", "0.9", "10.0"])
def test_unknown_major_is_refused(version: str) -> None:
    raw = make_raw()
    raw["schema_version"] = version
    with pytest.raises(ContractError, match="unsupported"):
        load_document(json.dumps(raw))


@pytest.mark.parametrize("version", [None, 1, "1", "1.x", "v1.0", "1.0.1", ""])
def test_malformed_version_is_refused(version: object) -> None:
    raw = make_raw()
    raw["schema_version"] = version
    with pytest.raises(ContractError, match="schema_version"):
        load_document(json.dumps(raw))


def test_minor_version_is_accepted_and_unknown_fields_dropped() -> None:
    raw = make_raw()
    raw["schema_version"] = "1.7"
    raw["new_top_level"] = {"anything": True}
    raw["video"]["new_field"] = 1
    raw["segments"][1]["flags"][0]["new_flag_field"] = "x"
    raw["segments"][1]["words"][0]["new_word_field"] = 1
    raw["transcript"]["runs"][0]["vad"] = {"kept": "free-form blobs are untouched"}
    doc = load_document(json.dumps(raw))
    assert doc.schema_version == "1.0"
    assert doc.transcript.runs[0].vad == {"kept": "free-form blobs are untouched"}
    assert "new_field" not in dump_document(doc)


def test_same_version_still_rejects_unknown_fields() -> None:
    raw = make_raw()
    raw["video"]["new_field"] = 1
    with pytest.raises(ContractError, match="new_field"):
        load_document(json.dumps(raw))


def test_minor_version_still_validates_contract_rules() -> None:
    raw = make_raw()
    raw["schema_version"] = "1.1"
    raw["segments"][1]["id"] = 7
    with pytest.raises(ContractError, match="dense"):
        load_document(json.dumps(raw))


@pytest.mark.parametrize("text", ["not json", "[]", '"1.0"'])
def test_non_document_input_is_refused(text: str) -> None:
    with pytest.raises(ContractError):
        load_document(text)
