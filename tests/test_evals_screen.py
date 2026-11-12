from pathlib import Path

import pytest
from evals_fixture import CHUNKS, ZH, chunk_rows, make_eval, write_pair

from evals.consensus import Segment, TimedUnit, find_disputes, timed_units
from evals.metrics import weighted_wilson, wilson_interval
from evals.screen import (
    HypothesisScreen,
    Quality,
    Share,
    Speed,
    build_screen,
    collect_spans,
    file_quality,
    format_only,
    recommend,
)
from evals.store import load_plan, row_units, write_rows
from ytverbatim.asr.models import ChunkHypothesis
from ytverbatim.contract.models import Word


def dispute_of(a: str, b: str, language: str = "zh"):
    hyps = {
        "a": timed_units([Segment(f"我们{a}了", 0, 5)], language),  # type: ignore[arg-type]
        "b": timed_units([Segment(f"我们{b}了", 0, 5)], language),  # type: ignore[arg-type]
    }
    (dispute,) = find_disputes(hyps, language)  # type: ignore[arg-type]
    return dispute


def test_written_form_only_differences_are_set_aside_but_case_errors_stay() -> None:
    assert format_only(dispute_of("买一百五十股", "买150股"), "zh")
    assert not format_only(dispute_of("买NVDA", "买nvda"), "zh")
    assert not format_only(dispute_of("买一百股", "买两百股"), "zh")


def en_dispute(a: str, b: str):
    hyps = {"a": timed_units([Segment(a, 0, 5)], "en"), "b": timed_units([Segment(b, 0, 5)], "en")}
    (dispute,) = find_disputes(hyps, "en")
    return dispute


def test_english_sentence_capitals_are_written_form_but_acronym_case_is_not() -> None:
    assert format_only(en_dispute("so They said", "so they said"), "en")
    assert not format_only(en_dispute("buy NVDA now", "buy nvda now"), "en")
    assert not en_dispute("They said it", "We said it").touches_latin
    assert en_dispute("we met Powell today", "we met Paul today").touches_latin


def test_row_units_keep_word_times_and_normalize_the_whole_text() -> None:
    row = ChunkHypothesis(
        hypothesis="w-prod", file="f", chunk=0, start=0.0, end=10.0, text="嗯好的",
        words=(Word(text="嗯", start=0.0, end=1.0), Word(text="好的", start=4.0, end=5.0)),
    )  # fmt: skip
    units = row_units(row, "zh", "raw")
    # 嗯 is a standalone filler as its own word but not inside 嗯好的; the whole text decides, as for chunk engines.
    assert [u.text for u in units] == ["嗯", "好", "的"]
    assert [(u.start_s, u.end_s) for u in units] == [(0.0, 0.0), (4.0, 4.5), (4.5, 5.0)]


def test_chunk_units_spread_over_speech_so_pauses_hold_no_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    plan = load_plan(layout, ZH)
    assert plan is not None
    # One long chunk spanning the plan's internal pauses, as a 30 s chunk with breaths in it would.
    row = ChunkHypothesis(hypothesis="q-base", file=ZH, chunk=0, start=0.0, end=19.0, text="甲" * 15)
    assert file_quality(manifest.videos[0], plan, [row]).nonspeech_units == 0
    units = row_units(row, "zh", "raw", plan.speech)
    assert all(any(s <= u.midpoint_s < e for s, e in plan.speech) for u in units)
    even = row_units(row, "zh", "raw")
    assert any(not any(s <= u.midpoint_s < e for s, e in plan.speech) for u in even)


def test_quality_counts_flags_and_redecodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    plan = load_plan(layout, ZH)
    assert plan is not None
    rows = chunk_rows(
        "q-base", ZH, ["好好好好好好好好", "今天", "明天", ""],
        flags=[("token_cap_hit",), ("redecoded",), ("decoder_anomaly", "redecoded"), ()],  # type: ignore[list-item]
    )  # fmt: skip
    q = file_quality(manifest.videos[0], plan, rows)
    assert (q.token_cap_hits, q.decoder_anomalies, q.redecoded, q.chunks) == (1, 1, 2, 4)
    assert q.redecode_share == 0.5
    assert q.repetition_runs == 1
    assert q.token_cap_per_hour == pytest.approx(1 / (plan.duration_s / 3600))


def test_whole_file_text_in_a_pause_counts_as_nonspeech(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    plan = load_plan(layout, ZH)
    assert plan is not None
    gap = (CHUNKS[1].end + CHUNKS[2].start) / 2
    row = ChunkHypothesis(
        hypothesis="w-prod", file=ZH, chunk=0, start=0.0, end=20.0, text="谢谢观看",
        words=(Word(text="谢谢观看", start=gap - 0.4, end=gap + 0.4),),
    )  # fmt: skip
    assert file_quality(manifest.videos[0], plan, [row]).nonspeech_units == 4


def test_screen_reports_reference_cer_and_recommends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evals import run_asr

    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    write_pair(layout)
    run_asr.build_captions(layout, manifest)
    write_rows(
        layout, "f-aed", chunk_rows("f-aed", ZH, ["我们今天买了股票", "然后卖出了一百股", "看nvda的财报", "大家好"])
    )
    hypotheses = ["f-aed", "q-base", "w-prod", "yt-auto"]
    spans = collect_spans(layout, manifest, hypotheses)
    result = build_screen(layout, manifest, hypotheses, spans, {}, judge_model=None, judge_prompt_version=None,
                          judge_seed=0, judge_sample=0, generated="t")  # fmt: skip
    control = result.get("w-prod")
    assert control is not None
    challenger = result.get("q-base")
    assert challenger is not None
    # The reference covers two of four chunks, so both pay insertions for the rest; 二 for 一 costs the challenger.
    assert challenger.reference_cer["zh"].errors > control.reference_cer["zh"].errors
    assert "en" not in control.reference_cer
    assert challenger.caption_wer is not None
    assert result.recommendation.control == "w-prod"


def share(wins: int, decided: int) -> Share:
    low, high = wilson_interval(wins, decided)
    return Share(wins=wins, decided=decided, needs_audio=0, low=low, high=high)


def row(name: str, judge: Share | None) -> HypothesisScreen:
    q = {"zh": Quality(), "en": Quality()}
    return HypothesisScreen(hypothesis=name, quality=q, reference_cer={}, reference_cer_raw={}, reference_mer={},
                            caption_wer=None, speed=Speed(rtf=None, load_s=None, peak_memory_bytes=None,
                            max_rss_bytes=None), judge=judge, disputes=0)  # fmt: skip


def test_recommendation_flags_overlapping_challengers() -> None:
    clear = recommend([row("w-prod", share(50, 100)), row("w-vad", share(60, 100)), row("q-ctx", share(90, 100)),
                       row("f-aed", share(40, 100))])  # fmt: skip
    assert (clear.control, clear.challenger, clear.runner_up) == ("w-vad", "q-ctx", "f-aed")
    assert not clear.challenger_ci_overlap
    close = recommend([row("w-prod", None), row("q-ctx", share(70, 100)), row("q-base", share(66, 100))])
    assert close.challenger == "q-ctx"
    assert close.challenger_ci_overlap
    assert "overlaps q-base" in close.challenger_reason


def test_weighted_wilson_reduces_to_wilson_with_equal_weights() -> None:
    weighted = weighted_wilson([(True, 2.0)] * 8 + [(False, 2.0)] * 2)
    assert (weighted.low, weighted.high) == pytest.approx(wilson_interval(8, 10))
    skewed = weighted_wilson([(True, 1.0)] * 8 + [(False, 9.0)] * 2)
    assert skewed.rate == pytest.approx(8 / 26)
    assert skewed.effective_n < 10


def test_timed_unit_midpoint() -> None:
    assert TimedUnit("a", 1.0, 3.0).midpoint_s == 2.0
