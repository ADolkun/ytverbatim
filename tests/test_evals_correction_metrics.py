import pytest

from evals.correction import BaseFile, CandidateFile, audible_edits
from evals.correction_metrics import (
    DELTA_UPPER_BOUND,
    MIN_DECIDED,
    OVER_CORRECTION_MAX,
    Answer,
    CandidateScreen,
    CorrectionReport,
    Estimate,
    HumanEstimates,
    JudgeStats,
    breaks_numbers,
    decide,
    human_estimate,
    human_estimates,
    judge_agreement,
    judge_stats,
    number_violations,
    pick,
    session_answers,
)
from evals.judge import Verdict
from evals.listen import Choice, LoggedDecision
from evals.normalize import Language
from evals.screen import Span, SpanReading
from evals.session import KeyEntry, SessionKey, Stratum
from ytverbatim.contract.models import Flag
from ytverbatim.errors import ConfigError
from ytverbatim.stages import Stretch

HOURS = 2.0
RESAMPLES = 400


def flag(before: str, after: str) -> Flag:
    return Flag(kind="entity_corrected", char_start=0, char_end=len(after), note=f"{before} → {after}")


@pytest.mark.parametrize(
    ("language", "before", "after", "flags", "broken"),
    [
        ("zh", "我们卖出了一百股", "我们卖出了一百五十股", [], True),
        ("en", "revenue was $5 million", "revenue was $5 billion", [], True),
        ("en", "up 15% on the day", "up 15% on the day", [], False),
        ("zh", "看NVDA的财报", "看NVDIA的财报", [flag("NVDA", "NVDIA")], False),
        ("zh", "看NVDA的财报", "看NVDIA的财报", [], True),
        ("zh", "看NVDA的财报", "看NVDIA的财报", [flag("NVDA", "MSCI")], True),
        ("zh", "看NVDA的财报", "看的财报", [], True),
        ("zh", "大家意见一致", "大家意见一直", [], False),
        ("en", "they bought two thousand shares", "they bought 2,000 shares", [], False),
        ("en", "we use open AI tools", "we use OpenAI tools", [flag("AI", "OpenAI")], False),
        ("en", "we use open AI tools", "we use OpenAI tools", [], True),
        ("en", "we use open AI tools", "we use OpenAI tools", [flag("ML", "OpenAI")], True),
        ("en", "we use OpenAI tools", "we use Open AI tools", [flag("OpenAI", "AI")], False),
        ("en", "we use OpenAI tools", "we use Open AI tools", [], True),
        ("en", "open AI and GAP", "OpenAI and GAAP", [flag("AI", "OpenAI"), flag("GAP", "GAAP")], False),
        ("en", "open AI and GAP", "OpenAI and GAAP", [flag("AI", "OpenAI")], True),
        ("en", "NVDA and AMD", "NVDA", [flag("AI", "OpenAI")], True),
    ],
)
def test_breaks_numbers(language: Language, before: str, after: str, flags: list[Flag], broken: bool) -> None:
    assert breaks_numbers(before, after, flags, language) is broken


def make_base(texts: tuple[str, ...], language: str = "zh") -> BaseFile:
    return BaseFile(
        file="f", hypothesis="h", punctuation_llm="p", versions={}, language=language, input_digest="d",
        stretches=tuple(Stretch(text=t, start=float(i), end=float(i) + 1) for i, t in enumerate(texts)),
        usage=(),
    )  # fmt: skip


def make_candidate(texts: tuple[str, ...], flags: tuple[tuple[Flag, ...], ...]) -> CandidateFile:
    return CandidateFile(
        file="f", candidate="c", client="x", base_digest="d", versions={}, texts=texts,
        edits=tuple(() for _ in texts), flags=flags, rejections=(), usage=(), batches=1,
    )  # fmt: skip


def test_number_violations_counts_stretches() -> None:
    base = make_base(("买了一百股", "看NVDA", "大家意见一致", "没有变化"))
    candidate = make_candidate(
        ("买了一百五十股", "看NVDIA", "大家意见一直", "没有变化"),
        ((), (flag("NVDA", "NVDIA"),), (), ()),
    )
    assert number_violations(base, candidate) == 1
    unflagged = make_candidate(candidate.texts, ((),) * 4)
    assert number_violations(base, unflagged) == 2


def span(id: str, base: str, readings: dict[str, str], *, format_only: bool = False, language: str = "zh") -> Span:
    shown = {"base": base, **readings}
    return Span(
        id=id, file="f", language="zh" if language == "zh" else "en", window=0,
        readings=tuple(
            SpanReading(hypothesis=h, text=t, start=0.0, end=1.0, before=None, after=None) for h, t in shown.items()
        ),
        distinct=tuple(dict.fromkeys(shown.values())), left="", right="", start=0.0, end=1.0,
        touches_quantity=False, touches_latin=False, format_only=format_only,
    )  # fmt: skip


def verdict(span_id: str, choice: str | None) -> Verdict:
    return Verdict(key=span_id, choice=choice, needs_audio=choice is None, reason="", model="judge", prompt_version="v")


def judged_spans() -> tuple[list[Span], dict[str, Verdict]]:
    spans = [
        span("s1", "a", {"c1": "A", "c2": "a"}),
        span("s2", "b", {"c1": "B", "c2": "b"}),
        span("s3", "c", {"c1": "c", "c2": "C"}),
        span("s4", "d", {"c1": "D", "c2": "d"}),
        span("s5", "e", {"c1": "E", "c2": "e"}),
        span("s6", "f", {"c1": "F", "c2": "d"}, format_only=True),
        span("s7", "g", {"c1": "G", "c2": "gg"}),
    ]
    verdicts = {
        "s1": verdict("s1", "A"),
        "s2": verdict("s2", "b"),
        "s3": verdict("s3", "C"),
        "s4": verdict("s4", None),
        "s5": verdict("s5", "E"),
        "s6": verdict("s6", "F"),
    }
    return spans, verdicts


def test_judge_stats_by_hand() -> None:
    spans, verdicts = judged_spans()
    assert [s.id for s in audible_edits(spans, "c1")] == ["s1", "s2", "s4", "s5", "s7"]
    stats = judge_stats(spans, verdicts, "c1", hours=HOURS, resamples=RESAMPLES)
    # s7 has no verdict, s4 needs audio; s1 and s5 are fixes, s2 a break.
    assert (stats.edits, stats.judged, stats.needs_audio, stats.decided) == (5, 4, 1, 3)
    assert (stats.fixes, stats.breaks) == (2, 1)
    assert stats.fix_precision is not None
    assert stats.fix_precision.value == pytest.approx(2 / 3)
    assert stats.over_correction is not None
    assert stats.over_correction.value == pytest.approx(1 / 3)
    assert stats.net_per_hour is not None
    assert stats.net_per_hour.value == pytest.approx((2 - 1) / 3 * 5 / HOURS)
    assert stats.net_per_hour.low <= stats.net_per_hour.value <= stats.net_per_hour.high

    other = judge_stats(spans, verdicts, "c2", hours=HOURS, resamples=RESAMPLES)
    # c2's reading differs from base on s3 (the judge's choice: a fix) and s7; the judge chose c1's reading on s1
    # and s5, but c2 left them alone, so they are not its edits.
    assert (other.edits, other.fixes, other.breaks) == (2, 1, 0)


def test_judge_label_zero_for_another_candidates_reading() -> None:
    spans = [span("x", "a", {"c1": "A", "c2": "Z"})]
    stats = judge_stats(spans, {"x": verdict("x", "Z")}, "c1", hours=HOURS, resamples=RESAMPLES)
    assert (stats.decided, stats.fixes, stats.breaks) == (1, 0, 0)
    assert stats.net_per_hour is not None
    assert stats.net_per_hour.value == 0


def test_judge_stats_is_deterministic_and_excludes_needs_audio() -> None:
    spans, verdicts = judged_spans()
    first = judge_stats(spans, verdicts, "c1", hours=HOURS, resamples=RESAMPLES)
    assert first == judge_stats(spans, verdicts, "c1", hours=HOURS, resamples=RESAMPLES)
    assert first.decided == 3

    many = [span(f"m{i}", "a", {"c1": "A"}) for i in range(200)]
    mixed = {s.id: verdict(s.id, "A" if i % 5 < 3 else "a") for i, s in enumerate(many)}
    one = judge_stats(many, mixed, "c1", hours=HOURS, resamples=RESAMPLES)
    assert one == judge_stats(many, mixed, "c1", hours=HOURS, resamples=RESAMPLES)
    assert one != judge_stats(many, mixed, "c1", hours=HOURS, seed=1, resamples=RESAMPLES)

    only_audio = judge_stats(spans, {"s4": verdicts["s4"]}, "c1", hours=HOURS, resamples=RESAMPLES)
    assert only_audio.needs_audio == 1
    assert only_audio.net_per_hour is None
    assert only_audio.over_correction is None


def judge(net: float, over: float = 0.05, *, width: float = 1.0) -> JudgeStats:
    return JudgeStats(
        edits=100, judged=100, needs_audio=0, decided=100, fixes=60, breaks=5,
        fix_precision=Estimate(value=0.6, low=0.5, high=0.7),
        over_correction=Estimate(value=over, low=0.0, high=0.2),
        net_per_hour=Estimate(value=net, low=net - width, high=net + width),
    )  # fmt: skip


def screen(
    id: str,
    net: float = 10.0,
    *,
    local: bool = False,
    over: float = 0.05,
    violations: int = 0,
    failed: bool = False,
    complete: bool = True,
    width: float = 1.0,
) -> CandidateScreen:
    return CandidateScreen(
        id=id, local=local, complete=complete, llm_output_failed=failed, number_violations=violations,
        first_try_validity=1.0, judge=judge(net, over, width=width),
    )  # fmt: skip


def test_pick_takes_the_best_by_net() -> None:
    chosen = pick([screen("a", 8.0), screen("b", 12.0), screen("c", 10.0)])
    assert (chosen.candidate, chosen.best, chosen.eligible) == ("b", "b", ("a", "b", "c"))


@pytest.mark.parametrize(
    ("bad", "why"),
    [
        (screen("x", complete=False), "incomplete"),
        (screen("x", failed=True), "LLMOutputError"),
        (screen("x", violations=1), "number violations"),
        (screen("x", over=OVER_CORRECTION_MAX + 0.01), "over-correction"),
        (screen("x", net=0.0), "judge net"),
        (screen("x", net=-3.0), "judge net"),
    ],
)
def test_pick_gates(bad: CandidateScreen, why: str) -> None:
    chosen = pick([bad, screen("ok", 1.0)])
    assert chosen.candidate == "ok"
    assert why in " ".join(chosen.ineligible["x"])


def test_pick_gates_boundary_over_correction_is_eligible() -> None:
    assert pick([screen("x", over=OVER_CORRECTION_MAX)]).candidate == "x"


def test_pick_ineligible_without_a_decided_verdict() -> None:
    nothing = JudgeStats(
        edits=3, judged=1, needs_audio=1, decided=0, fixes=0, breaks=0,
        fix_precision=None, over_correction=None, net_per_hour=None,
    )  # fmt: skip
    silent = screen("x").model_copy(update={"judge": nothing})
    chosen = pick([silent])
    assert chosen.candidate is None
    assert "decided judge verdict" in " ".join(chosen.ineligible["x"])


def test_pick_prefers_local_when_intervals_overlap() -> None:
    chosen = pick([screen("claude", 12.0), screen("l-near", 11.0, local=True), screen("l-also", 10.5, local=True)])
    assert (chosen.candidate, chosen.best) == ("l-near", "claude")
    assert "preferred" in chosen.reason


def test_pick_keeps_best_when_local_interval_is_apart() -> None:
    chosen = pick([screen("claude", 12.0), screen("l-far", 5.0, local=True)])
    assert chosen.candidate == "claude"


def test_pick_ignores_an_ineligible_local_that_overlaps() -> None:
    chosen = pick([screen("claude", 12.0), screen("l-bad", 12.0, local=True, violations=2)])
    assert chosen.candidate == "claude"


def test_pick_none_eligible() -> None:
    chosen = pick([screen("a", failed=True), screen("b", violations=3)])
    assert (chosen.candidate, chosen.best, chosen.eligible) == (None, None, ())
    assert set(chosen.ineligible) == {"a", "b"}
    assert pick([]).candidate is None


def entry(id: str, weight: float, *, language: str = "zh", stratum: Stratum = "other", option: int = 1) -> KeyEntry:
    return KeyEntry(
        id=id, source="s", file="f", language="zh" if language == "zh" else "en", stratum=stratum,
        weight=weight, challenger_option=1 if option == 1 else 2,
    )  # fmt: skip


def key(entries: list[KeyEntry], challenger: str = "c1") -> SessionKey:
    return SessionKey(
        control="base", challenger=challenger, seed=0, n=len(entries), numeric_share=0.4, created="t",
        population={}, entries=tuple(entries),
    )  # fmt: skip


def logged(span_id: str, choice: Choice) -> LoggedDecision:
    return LoggedDecision(span=span_id, choice=choice, at="t")


# zh other (w 4.0): fix, break, both; zh numeric (w 1.5): fix, neither; en other (w 3.0): fix, fix (option 2).
ENTRIES = [
    entry("z1", 4.0), entry("z2", 4.0), entry("z3", 4.0),
    entry("n1", 1.5, stratum="numeric"), entry("n2", 1.5, stratum="numeric"),
    entry("e1", 3.0, language="en", option=2), entry("e2", 3.0, language="en", option=2),
]  # fmt: skip
DECISIONS = [
    logged("z1", "1"), logged("z2", "2"), logged("z3", "both"),
    logged("n1", "1"), logged("n2", "neither"),
    logged("e1", "2"), logged("e2", "2"),
]  # fmt: skip


def test_session_answers_maps_sides_and_counts_undecided() -> None:
    answers, undecided = session_answers(key(ENTRIES), [*DECISIONS[:2], logged("z3", "skip"), logged("e1", "1")])
    assert undecided == 4
    assert [(a.span, a.verdict) for a in answers] == [("z1", "after"), ("z2", "before"), ("e1", "before")]
    assert session_answers(key(ENTRIES), [logged("z1", "skip"), logged("z1", "1")])[1] == 6


def test_session_answers_rejects_unknown_spans() -> None:
    with pytest.raises(ConfigError, match="unknown spans"):
        session_answers(key(ENTRIES), [logged("nope", "1")])


def test_human_estimate_matches_hand_weighted_counts() -> None:
    answers, undecided = session_answers(key(ENTRIES), DECISIONS)
    assert undecided == 0
    estimate = human_estimate(answers, edits=200, hours=4.0, resamples=RESAMPLES)
    total = 3 * 4.0 + 2 * 1.5 + 2 * 3.0
    fixes = 4.0 + 1.5 + 3.0 + 3.0
    breaks = 4.0
    assert estimate.answered == 7
    assert estimate.fix_precision.value == pytest.approx(fixes / total)
    assert estimate.over_correction.value == pytest.approx(breaks / total)
    assert estimate.both_fine == pytest.approx(4.0 / total)
    assert estimate.both_wrong == pytest.approx(1.5 / total)
    assert estimate.net_per_edit == pytest.approx((fixes - breaks) / total)
    assert estimate.effective_n == pytest.approx(total**2 / (3 * 16 + 2 * 2.25 + 2 * 9))
    assert estimate.delta_per_hour.value == pytest.approx(-(fixes - breaks) / total * 200 / 4.0)
    assert estimate.delta_per_hour.low <= estimate.delta_per_hour.value <= estimate.delta_per_hour.high


def test_human_estimate_bootstrap_is_deterministic_and_stratified() -> None:
    answers, _ = session_answers(key(ENTRIES), DECISIONS)
    first = human_estimate(answers, edits=200, hours=4.0, resamples=RESAMPLES)
    assert first == human_estimate(answers, edits=200, hours=4.0, resamples=RESAMPLES)
    assert first != human_estimate(answers, edits=200, hours=4.0, seed=3, resamples=RESAMPLES)

    # Every en span is a fix and the stratum is resampled on its own, so a lone en stratum has no spread.
    only_en = [a for a in answers if a.language == "en"]
    interval = human_estimate(only_en, edits=10, hours=1.0, resamples=RESAMPLES).delta_per_hour
    assert interval.low == interval.value == interval.high == -10.0


def test_human_estimates_pools_and_splits_by_language() -> None:
    result = human_estimates(
        key(ENTRIES), DECISIONS, edits={"zh": 150, "en": 50}, hours={"zh": 3.0, "en": 1.0}, resamples=RESAMPLES
    )
    assert (result.candidate, result.spans, result.undecided) == ("c1", 7, 0)
    assert result.overall is not None
    assert (result.overall.edits, result.overall.hours) == (200, 4.0)
    assert set(result.by_language) == {"zh", "en"}
    zh = result.by_language["zh"]
    assert zh.answered == 5
    zh_total = 3 * 4.0 + 2 * 1.5
    assert zh.net_per_edit == pytest.approx((4.0 + 1.5 - 4.0) / zh_total)
    assert result.by_language["en"].delta_per_hour.value == pytest.approx(-50.0)


def test_human_estimates_with_nothing_answered() -> None:
    result = human_estimates(key(ENTRIES), [], edits={"zh": 1, "en": 1}, hours={"zh": 1.0, "en": 1.0})
    assert (result.overall, result.by_language, result.undecided) == (None, {}, 7)


def test_judge_agreement() -> None:
    answers = [
        Answer("a", "zh", "other", 1.0, "after"),
        Answer("b", "zh", "other", 1.0, "before"),
        Answer("c", "zh", "other", 1.0, "after"),
        Answer("d", "zh", "other", 1.0, "both"),
        Answer("e", "zh", "other", 1.0, "after"),
    ]
    # c: the judge sided with another reading (0); d is not decisive; e has no judge label.
    result = judge_agreement({"a": 1, "b": 1, "c": 0, "d": 1}, answers)
    assert result is not None
    assert (result.agree, result.decisive) == (1, 3)
    assert result.low <= result.rate <= result.high
    assert judge_agreement({}, answers) is None


def human(
    *,
    over: float = 0.10,
    delta: tuple[float, float, float] = (-30.0, -50.0, -10.0),
    undecided: int = 0,
    candidate: str = "c1",
    spans: int = 80,
    answered: int = MIN_DECIDED,
) -> HumanEstimates:
    value, low, high = delta
    estimate = human_estimate([Answer("a", "zh", "other", 1.0, "after")], edits=10, hours=1.0, resamples=10).model_copy(
        update={
            "answered": answered,
            "over_correction": Estimate(value=over, low=0.0, high=0.3),
            "delta_per_hour": Estimate(value=value, low=low, high=high),
        }
    )
    return HumanEstimates(candidate=candidate, spans=spans, undecided=undecided, overall=estimate, by_language={})


def report(*candidates: CandidateScreen, human: HumanEstimates | None = None) -> CorrectionReport:
    return CorrectionReport(candidates=candidates or (screen("c1"),), human=human)


def failed_checks(decision_report: CorrectionReport) -> list[str]:
    return [c.name for c in decide(decision_report).checks if not c.passed]


def test_decide_all_pass_is_on() -> None:
    decision = decide(report(human=human()))
    assert decision.default_on
    assert decision.candidate == "c1"
    assert all(c.passed for c in decision.checks)
    assert {c.name for c in decision.checks} >= {
        "every span decided",
        "enough labels",
        "over-correction",
        "errors per hour",
        "number preservation",
        "JSON after the retry",
    }


def test_decide_needs_enough_decided_spans() -> None:
    assert MIN_DECIDED == 30
    few = decide(report(human=human(answered=MIN_DECIDED - 1)))
    assert not few.default_on
    assert [c.name for c in few.checks if not c.passed] == ["enough labels"]
    assert "too few labels" in next(c.detail for c in few.checks if c.name == "enough labels")
    assert decide(report(human=human(answered=MIN_DECIDED))).default_on
    assert not decide(report(human=human(answered=1))).default_on


def test_decide_ci_including_zero_is_off() -> None:
    touching = human(delta=(-20.0, -40.0, DELTA_UPPER_BOUND))
    assert failed_checks(report(human=touching)) == ["errors per hour"]
    assert failed_checks(report(human=human(delta=(-20.0, -40.0, 5.0)))) == ["errors per hour"]
    assert not decide(report(human=touching)).default_on


def test_decide_over_correction_bar() -> None:
    assert failed_checks(report(human=human(over=0.16))) == ["over-correction"]
    assert decide(report(human=human(over=OVER_CORRECTION_MAX))).default_on


def test_decide_one_number_violation_is_off() -> None:
    off = decide(report(screen("c1", violations=1), human=human()))
    assert not off.default_on
    assert off.candidate is None
    assert "1 number violations" in off.checks[0].detail


def test_decide_llm_output_error_is_off() -> None:
    off = decide(report(screen("c1", failed=True), human=human()))
    assert not off.default_on
    assert "LLMOutputError" in off.checks[0].detail


def test_decide_first_try_validity_is_not_gated() -> None:
    sloppy = screen("c1").model_copy(update={"first_try_validity": 0.5})
    assert decide(report(sloppy, human=human())).default_on


def test_decide_skipped_span_refuses() -> None:
    with pytest.raises(ConfigError, match="3 of 80 spans without a decision"):
        decide(report(human=human(undecided=3)))


def test_decide_no_session_is_off() -> None:
    decision = decide(report())
    assert not decision.default_on
    assert decision.candidate == "c1"
    assert [c.name for c in decision.checks if not c.passed] == ["human session"]


def test_decide_no_eligible_candidate_is_off() -> None:
    decision = decide(report(screen("a", failed=True), screen("b", net=-1.0), human=human()))
    assert not decision.default_on
    assert decision.candidate is None
    assert "a:" in decision.checks[0].detail
    assert "b:" in decision.checks[0].detail
    assert not decide(CorrectionReport(candidates=(), human=None)).default_on


def test_decide_refuses_a_session_for_another_candidate() -> None:
    with pytest.raises(ConfigError, match="rebuild the session"):
        decide(report(human=human(candidate="c2")))


def test_decide_empty_session_is_off() -> None:
    empty = HumanEstimates(candidate="c1", spans=0, undecided=0, overall=None, by_language={})
    assert failed_checks(report(human=empty)) == ["human estimate"]
