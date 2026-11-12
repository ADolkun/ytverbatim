import json
import re
from pathlib import Path

import pytest
from evals_fixture import EN, EN_CHALLENGER, EN_CONTROL, ZH, ZH_CHALLENGER, ZH_CONTROL, make_eval, write_pair

from evals import run_asr
from evals.decide import decide
from evals.listen import LoggedDecision
from evals.normalize import Language
from evals.report import ABResult, Report, WinStats, build_report, render_markdown
from evals.screen import (
    HypothesisScreen,
    Quality,
    Rate,
    Recommendation,
    ScreenResult,
    Speed,
    build_screen,
    collect_spans,
)
from evals.session import DECISIONS_FILE, build_session, load_key, write_session

HOUR = 3600.0


def stats(wins: int, losses: int) -> WinStats:
    from evals.metrics import weighted_wilson

    w = weighted_wilson([(True, 1.0)] * wins + [(False, 1.0)] * losses)
    return WinStats(challenger_wins=wins, control_wins=losses, both=3, neither=1, skipped=0, rate=w.rate,
                    effective_n=w.effective_n, low=w.low, high=w.high)  # fmt: skip


def quality(**extra: float) -> dict[Language, Quality]:
    base = Quality(files=3, audio_s=2 * HOUR, units=10_000, chunks=200)
    return {"zh": base.model_copy(update=extra), "en": base.model_copy(update=extra)}


def screen_row(name: str, cer: float | None = 0.10, **extra: float) -> HypothesisScreen:
    reference = {"zh": Rate(errors=round(cer * 1000), reference_units=1000)} if cer is not None else {}
    return HypothesisScreen(hypothesis=name, quality=quality(**extra), reference_cer=reference,
                            reference_cer_raw=reference, reference_mer=reference, caption_wer=None,
                            speed=Speed(rtf=0.1, load_s=1.0, peak_memory_bytes=None, max_rss_bytes=None),
                            judge=None, disputes=10)  # fmt: skip


def report(
    zh: WinStats, en: WinStats, zh_numeric: WinStats, en_numeric: WinStats, challenger: HypothesisScreen | None = None
) -> Report:
    rows = (screen_row("w-prod"), challenger or screen_row("q-ctx"))
    screen = ScreenResult(
        generated="t", hypotheses=rows, spans=100, format_only_spans=0, judge_model=None, judge_prompt_version=None,
        judge_seed=0, judge_sample=0, judge_answered=0,
        recommendation=Recommendation(control="w-prod", control_reason="", challenger="q-ctx", challenger_reason="",
                                      runner_up=None, challenger_ci_overlap=False),
    )  # fmt: skip
    ab = ABResult(session="main", control="w-prod", challenger="q-ctx", seed=0, spans=150, answered=150,
                  overall=stats(1, 1), by_language={"zh": zh, "en": en}, numeric=stats(1, 1),
                  numeric_by_language={"zh": zh_numeric, "en": en_numeric}, by_file={})  # fmt: skip
    return Report(generated="t", files=(), screen=screen, ab=ab)


WIN = stats(60, 15)
TIE = stats(40, 35)
NUMERIC_OK = stats(60, 40)
NUMERIC_BAD = stats(12, 18)


def failed(decision_report: Report, language: Language) -> list[str]:
    result = decide(decision_report)
    (lang,) = [d for d in result.languages if d.language == language]
    return [c.name for c in lang.checks if not c.passed]


def test_a_clear_win_everywhere_replaces_whisper() -> None:
    result = decide(report(WIN, WIN, NUMERIC_OK, NUMERIC_OK))
    assert result.engine == {"zh": "q-ctx", "en": "q-ctx"}
    assert "replaces" in result.summary


def test_a_ci_that_includes_half_keeps_whisper() -> None:
    result = decide(report(TIE, TIE, NUMERIC_OK, NUMERIC_OK))
    assert result.engine == {"zh": "w-prod", "en": "w-prod"}
    assert failed(report(TIE, TIE, NUMERIC_OK, NUMERIC_OK), "zh") == ["overall win rate"]


def test_numeric_inferiority_keeps_whisper() -> None:
    assert failed(report(WIN, WIN, NUMERIC_BAD, NUMERIC_OK), "zh") == ["numeric/entity non-inferiority"]
    # Point estimate 0.5 or better but a lower bound at or under 0.40 still fails: too few numeric spans.
    assert "numeric/entity non-inferiority" in failed(report(WIN, WIN, stats(3, 2), NUMERIC_OK), "zh")
    assert decide(report(WIN, WIN, NUMERIC_BAD, NUMERIC_BAD)).engine == {"zh": "w-prod", "en": "w-prod"}


@pytest.mark.parametrize(
    ("extra", "check"),
    [
        ({"repetition_runs": 2}, "repetition"),
        ({"token_cap_hits": 2}, "token cap"),
        ({"nonspeech_units": 100}, "non-speech output"),
    ],
)
def test_a_quality_gate_failure_keeps_whisper(extra: dict[str, float], check: str) -> None:
    worse = screen_row("q-ctx", **extra)
    assert check in failed(report(WIN, WIN, NUMERIC_OK, NUMERIC_OK, worse), "zh")
    within = screen_row("q-ctx", **{k: v / 2 for k, v in extra.items()})
    assert failed(report(WIN, WIN, NUMERIC_OK, NUMERIC_OK, within), "zh") == []


def test_reference_cer_more_than_a_point_worse_keeps_whisper() -> None:
    assert failed(report(WIN, WIN, NUMERIC_OK, NUMERIC_OK, screen_row("q-ctx", cer=0.115)), "zh") == ["reference CER"]
    assert failed(report(WIN, WIN, NUMERIC_OK, NUMERIC_OK, screen_row("q-ctx", cer=0.105)), "zh") == []
    assert failed(report(WIN, WIN, NUMERIC_OK, NUMERIC_OK, screen_row("q-ctx", cer=None)), "zh") == ["reference CER"]
    # English has no manual reference, so this check passes there.
    assert failed(report(WIN, WIN, NUMERIC_OK, NUMERIC_OK, screen_row("q-ctx", cer=None)), "en") == []


def test_a_per_language_split_needs_that_language_to_pass_alone() -> None:
    result = decide(report(WIN, TIE, NUMERIC_OK, NUMERIC_OK))
    assert result.engine == {"zh": "q-ctx", "en": "w-prod"}
    assert result.summary.startswith("per-language split")
    assert decide(report(TIE, WIN, NUMERIC_OK, NUMERIC_OK)).engine == {"zh": "w-prod", "en": "q-ctx"}


def test_no_session_keeps_the_control() -> None:
    base = report(WIN, WIN, NUMERIC_OK, NUMERIC_OK)
    result = decide(base.model_copy(update={"ab": None}))
    assert result.engine == {"zh": "w-prod", "en": "w-prod"}


# --- report end to end, scanned for transcript text ---


def test_the_report_and_results_hold_no_transcript_text_or_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    write_pair(layout)
    run_asr.build_captions(layout, manifest)
    hypotheses = ["w-prod", "q-base", "yt-auto"]
    spans = collect_spans(layout, manifest, hypotheses)
    screen = build_screen(layout, manifest, hypotheses, spans, {}, judge_model=None, judge_prompt_version=None,
                          judge_seed=0, judge_sample=0, generated="t")  # fmt: skip
    write_session(layout, "main", *build_session(layout, manifest, "w-prod", "q-base", n=10, seed=1))
    key = load_key(layout.session_dir("main"))
    with (layout.session_dir("main") / DECISIONS_FILE).open("w", encoding="utf-8") as handle:
        for i, entry in enumerate(key.entries):
            choice = str(entry.challenger_option) if i % 3 else "both"
            handle.write(LoggedDecision(span=entry.id, choice=choice, at="t").model_dump_json() + "\n")  # type: ignore[arg-type]
    result = build_report(layout, manifest, screen, "main", "t")
    assert result.ab is not None
    assert result.ab.overall.challenger_wins == sum(1 for i in range(len(key.entries)) if i % 3)
    assert result.ab.overall.both == sum(1 for i in range(len(key.entries)) if not i % 3)
    markdown = render_markdown(result)
    written, _ = run_asr.write_decision(result, tmp_path / "results", "2025-01-01")
    texts = [markdown, json.dumps(result.model_dump(mode="json"), ensure_ascii=False), written.read_text("utf-8")]

    sources = [*ZH_CONTROL, *ZH_CHALLENGER, *EN_CONTROL, *EN_CHALLENGER, "我们今天买了股票", "然后卖出了一百股"]
    han = {run for text in sources for run in re.findall(r"[一-鿿]{2,}", text)}
    words = {w for text in sources for w in text.split() if len(w) > 3}
    for text in texts:
        assert ZH not in text and EN not in text and "Example title" not in text
        assert not any(run in text for run in han)
        assert not any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)
        assert "zh-01" in text and "en-01" in text
