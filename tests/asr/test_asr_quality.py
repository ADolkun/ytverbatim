import pytest

from ytverbatim.asr.quality import (
    Anomaly,
    NonlexicalRun,
    Unit,
    cross_segment_repetitions,
    decoder_anomalies,
    has_lexical_content,
    intra_segment_repetitions,
    is_cjk_text,
    is_nonlexical_failure,
    nonlexical_runs,
    normalize_for_repetition,
    word_timing_collapses,
)
from ytverbatim.contract.models import Word


def record(start: float, end: float, text: str) -> Unit:
    return Unit(start, end, text, (Word(text=text, start=start, end=end, conf=0.99),) if text else ())


def kinds(findings: list[Anomaly]) -> set[str]:
    return {finding.kind for finding in findings}


def bridge() -> list[Unit]:
    return [Unit(0.0, 15.0, "开始讨论条件"), Unit(15.0, 75.0, "。"), Unit(75.0, 90.0, "还有风险限制")]


def test_long_punctuation_bridge_is_a_failure_and_brief_punctuation_is_not() -> None:
    (run,) = nonlexical_runs(bridge())
    assert run == NonlexicalRun(start=15.0, end=75.0, records=1, observed_seconds=60.0)
    assert is_nonlexical_failure(run)
    brief = [Unit(0.0, 15.0, "开始讨论条件"), Unit(15.0, 15.2, "。"), Unit(15.2, 90.0, "还有风险限制")]
    (run,) = nonlexical_runs(brief)
    assert run.observed_seconds == pytest.approx(0.2)
    assert not is_nonlexical_failure(run)


@pytest.mark.parametrize("text", ["!!!", "。。。", "", " "])
def test_text_without_letters_or_digits_has_no_lexical_content(text: str) -> None:
    assert not has_lexical_content(text)


def test_separated_tiny_records_do_not_count_silence_as_output() -> None:
    units = [Unit(0, 10, "first"), Unit(10, 10.1, "!"), Unit(99.9, 100, "!"), Unit(100, 110, "last")]
    runs = nonlexical_runs(units)
    assert len(runs) == 2
    assert sum(run.observed_seconds for run in runs) == pytest.approx(0.2)
    assert not any(is_nonlexical_failure(run) for run in runs)


@pytest.mark.parametrize("count", [9, 10])
@pytest.mark.parametrize("last_end", [14.99, 15.0])
def test_dense_run_needs_ten_records_over_five_seconds(count: int, last_end: float) -> None:
    units = [Unit(10 + i * 0.5, 10 + i * 0.5 + 0.01, "!") for i in range(count)]
    units[-1] = Unit(units[-1].start, last_end, "!")
    (run,) = nonlexical_runs([Unit(0, 10, "first"), *units, Unit(15, 30, "last")])
    assert run.records == count
    assert run.observed_seconds < run.end - run.start
    assert is_nonlexical_failure(run) == (count == 10 and last_end >= 15)


LOOPS = [
    "熟" * 111 + "�",
    "风险限制" * 8,
    " ".join(["repeat risk"] * 8),
    ", ".join(["مخاطر السوق"] * 8),
    ". ".join(["риск рынка"] * 8),
    " ".join(["ＲＥＰＥＡＴ ｒｉｓｋ"] * 8),
]


@pytest.mark.parametrize("text", LOOPS)
def test_extreme_multilingual_loops_are_anomalies(text: str) -> None:
    assert decoder_anomalies([Unit(0.0, 10.0, text)])


@pytest.mark.parametrize("text", ["no no no", "哈哈哈", "风险限制" * 7, "1,234 " * 12, "1234" * 12, "risk " * 7])
def test_brief_emphasis_numeric_repeats_and_the_threshold_are_not_anomalies(text: str) -> None:
    assert decoder_anomalies([Unit(0.0, 10.0, text)]) == []


def test_eight_short_segments_of_natural_repetition_are_not_a_loop() -> None:
    assert decoder_anomalies([Unit(i, i + 1, "Repeat risk") for i in range(8)]) == []


def test_replacement_character_is_counted() -> None:
    (finding,) = decoder_anomalies([Unit(1.0, 2.0, "meaningful � damaged �")])
    assert finding == Anomaly("replacement_character", 1.0, 2.0, 2)


def test_intra_segment_count_is_the_number_of_repeats() -> None:
    assert intra_segment_repetitions(" ".join(["repeat risk"] * 9)) == [9]
    assert intra_segment_repetitions("风险限制" * 12) == [12]


def test_mandarin_loops_are_found_through_inconsistent_separators() -> None:
    text = "，".join(["风险 限制"] * 3) + " 风险限制 风险限制。" + "风险限制，" * 3
    assert "风险限制" * 8 not in text
    assert intra_segment_repetitions(text) == [8]


def test_script_detection_follows_the_cjk_share_of_letters() -> None:
    assert is_cjk_text("SPY涨了3.5%，QQQ跌了")
    assert not is_cjk_text("the S&P 500 rose 涨")
    assert not is_cjk_text("1234 !!!")
    assert normalize_for_repetition("Risk, LIMITS!") == "risk limits"
    assert normalize_for_repetition("风险，限制！") == "风险限制"


def repeated(text: str, count: int, duration: float, gap: float, empty: int) -> list[Unit]:
    units: list[Unit] = []
    for index in range(count):
        start = round(10 + index * (duration + gap), 3)
        end = round(start + duration, 3)
        units.append(record(start, end, text))
        if index + 1 < count:
            units.extend(record(end, end, "") for _ in range(empty))
    return units


def observed_loop() -> list[Unit]:
    units = [
        record(0, 781.64, "Earlier healthy source context."),
        record(781.78, 782.62, "Maybe it is."),
        record(782.66, 783.12, "Maybe it's not."),
        record(783.3, 796.98, "A distinct healthy explanation before the failure."),
        record(797, 797.32, "Maybe."),
    ]
    timings = (
        (811.74, 812.0),
        (812.02, 812.04),
        (812.06, 812.08),
        (812.1, 812.12),
        (813.12, 813.38),
        (813.74, 813.88),
        (813.88, 814.0),
        (814.58, 814.84),
        (815.06, 815.08),
        (815.98, 816.24),
        (816.36, 816.62),
    )
    for index, (start, end) in enumerate(timings):
        units.append(record(start, end, "Maybe."))
        units.extend(record(end, end, "") for _ in range({3: 11, 4: 8, 5: 1, 9: 11}.get(index, 0)))
    units.extend([record(821.58, 821.84, "Maybe."), record(821.84, 829.2, "Distinct healthy evidence.")])
    return units


def test_observed_fragmented_loop_is_found_at_high_confidence() -> None:
    assert Anomaly("cross_segment_repetition", 811.74, 816.62, 11) in decoder_anomalies(observed_loop())


@pytest.mark.parametrize(
    ("text", "count", "duration", "gap", "empty", "expected"),
    [
        ("Repeat risk", 8, 1.0, 0, 0, False),
        ("Maybe.", 8, 0.2, 0, 0, False),
        ("Maybe.", 7, 0.02, 0, 4, False),
        ("Maybe.", 8, 0.02, 2.0, 0, False),
        ("2026", 11, 0.02, 0, 4, False),
        ("Maybe.", 8, 0.02, 0, 0, True),
        ("风险限制", 8, 0.02, 0, 0, True),
        ("Maybe.", 8, 0.2, 0, 2, True),
    ],
)
def test_natural_repetition_and_decoder_timing_thresholds(
    text: str, count: int, duration: float, gap: float, empty: int, expected: bool
) -> None:
    assert bool(cross_segment_repetitions(repeated(text, count, duration, gap, empty))) is expected


@pytest.mark.parametrize(
    ("short", "empty", "trailing", "expected"),
    [(2, 0, 0, False), (3, 0, 0, True), (0, 9, 0, False), (0, 10, 0, True), (0, 0, 20, False)],
)
def test_timing_corruption_must_belong_to_the_same_repeated_run(
    short: int, empty: int, trailing: int, expected: bool
) -> None:
    units = repeated("Maybe.", 8, 0.2, 0, 0)
    units = [record(u.start, u.start + 0.05, u.text) if i < short else u for i, u in enumerate(units)]
    units[1:1] = [record(10.2, 10.2, "") for _ in range(empty)]
    units.extend(record(11.6, 11.6, "") for _ in range(trailing))
    assert bool(cross_segment_repetitions(units)) is expected


def test_distinct_speech_between_repeats_breaks_the_run() -> None:
    units = repeated("Maybe.", 8, 0.02, 0, 4)
    units.insert(10, record(10.04, 10.04, "Distinct speech."))
    assert cross_segment_repetitions(units) == []


def burst(words: list[str], *, timed: bool = False) -> list[Unit]:
    end = 10 + 0.2 * len(words) if timed else 10.16
    return [
        Unit(
            10,
            end,
            " ".join(words),
            tuple(
                Word(text=w, start=10 + 0.2 * i, end=10 + 0.2 * (i + 1)) if timed else Word(text=w, start=end, end=end)
                for i, w in enumerate(words)
            ),
        )
    ]


STRICT = ["strict", "risk", "limits", "protect", "capital"]


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        (STRICT * 2, True),
        ((STRICT * 2)[:-1], False),
        (["you", "go", "up", "you", "go", "down"] * 2, True),
        (["you", "go", "up", "you", "go", "out"] * 2, False),
        (["风险控制", "严格限制", "仓位管理", "避免损失", "保护本金"] * 2, True),
        ([f"unique{index}" for index in range(14)], True),
        (["1234"] * 12, False),
    ],
)
def test_co_timed_word_collapse_needs_a_native_burst_and_letter_evidence(words: list[str], expected: bool) -> None:
    assert bool(word_timing_collapses(burst(words))) is expected
    assert word_timing_collapses(burst(words, timed=True)) == []


def test_late_word_collapse_reports_the_unit_span_and_word_count() -> None:
    words = ["or", "you", "get", "a", "drop", "back", "down", "to", "the", "bottom", "line", "And", "then"]
    words += ["you", "get", "a", "drop", "back", "down", "to"]
    unit = Unit(1116.12, 1116.28, " ".join(words), tuple(Word(text=w, start=1116.28, end=1116.28) for w in words))
    assert word_timing_collapses([unit]) == [Anomaly("word_timing_collapse", 1116.12, 1116.28, len(words))]


def test_an_interrupted_burst_is_not_a_collapse() -> None:
    (unit,) = burst(STRICT * 2)
    words = list(unit.words)
    words[5] = Word(text=words[5].text, start=10.0, end=10.16)
    assert word_timing_collapses([Unit(unit.start, unit.end, unit.text, tuple(words))]) == []


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        (["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yzAB", "cDEF"], True),
        (["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yzAB", "cDE"], False),
        (["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yzAB"], False),
    ],
)
def test_word_timing_word_and_letter_boundaries(words: list[str], expected: bool) -> None:
    assert bool(word_timing_collapses(burst(words))) is expected


@pytest.mark.parametrize(("middle", "expected"), [("", True), ("Unknown untimed source content.", False)])
def test_word_timing_run_crosses_seams_but_not_unknown_lexical_evidence(middle: str, expected: bool) -> None:
    (unit,) = burst([f"unique{index}" for index in range(10)])
    first = Unit(10, 10.16, "", unit.words[:5])
    second = Unit(10.16, 10.16, "", unit.words[5:])
    assert bool(word_timing_collapses([first, Unit(10.16, 10.16, middle), second])) is expected


def test_long_word_runs_are_linear_and_a_late_collapse_still_counts() -> None:
    words = [f"unique{index}" for index in range(5000)]
    assert word_timing_collapses(burst(words, timed=True)) == []
    (finding,) = word_timing_collapses(burst(words))
    assert finding.count == 5000
    (late,) = burst([*words, *words[:10]], timed=True)
    end = late.end
    tail = tuple(Word(text=w.text, start=end, end=end) for w in late.words[-10:])
    collapsed = Unit(late.start, end, late.text, (*late.words[:-10], *tail))
    assert any(f.kind == "word_timing_collapse" for f in decoder_anomalies([collapsed]))


def test_decoder_anomalies_covers_every_detector_without_duplicates() -> None:
    loop = Unit(20, 30, "风险限制" * 8 + "\ufffd")
    assert kinds(decoder_anomalies([*burst(STRICT * 2), loop])) == {
        "word_timing_collapse",
        "intra_segment_repetition",
        "replacement_character",
    }
    assert len(decoder_anomalies([loop, loop])) == 2
