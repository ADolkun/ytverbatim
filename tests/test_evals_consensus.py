import pytest

from evals.consensus import (
    Segment,
    TimedUnit,
    find_disputes,
    screening_stats,
    timed_units,
    window_units,
)
from evals.normalize import Language


def hypothesis(language: Language, *segments: str, step: float = 10.0) -> list[TimedUnit]:
    return timed_units([Segment(t, i * step, (i + 1) * step) for i, t in enumerate(segments)], language)


def test_agreeing_hypotheses_have_no_disputes() -> None:
    hyps = {name: hypothesis("zh", "今天天气很好") for name in "abc"}
    assert find_disputes(hyps, "zh") == []


def test_one_hypothesis_is_not_enough() -> None:
    with pytest.raises(ValueError, match="at least two"):
        find_disputes({"a": hypothesis("zh", "好")}, "zh")


def test_substitution_dispute_lists_every_reading_and_the_context() -> None:
    hyps = {
        "a": hypothesis("zh", "我们今天买了这只股票然后卖出"),
        "b": hypothesis("zh", "我们今天买了这只股票然后卖出"),
        "c": hypothesis("zh", "我们今天卖了这只股票然后卖出"),
    }
    (dispute,) = find_disputes(hyps, "zh")
    assert dispute.distinct == ("买", "卖")
    assert {r.hypothesis: r.text for r in dispute.readings} == {"a": "买", "b": "买", "c": "卖"}
    assert dispute.left_context == "我们今天"
    assert dispute.right_context == "了这只股票然后卖出"
    assert not dispute.touches_quantity
    assert not dispute.touches_latin


def test_insertion_against_epsilon() -> None:
    hyps = {
        "a": hypothesis("en", "we sold the stock today"),
        "b": hypothesis("en", "we sold the stock today"),
        "c": hypothesis("en", "we sold the big stock today"),
    }
    (dispute,) = find_disputes(hyps, "en")
    assert dispute.distinct == ("", "big")
    by_name = {r.hypothesis: r for r in dispute.readings}
    assert by_name["a"].start_s is None
    assert by_name["c"].text == "big"
    assert dispute.left_context == "we sold the"


def test_spans_within_one_agreed_unit_merge() -> None:
    hyps = {
        "a": hypothesis("zh", "甲乙丙丁戊己庚辛"),
        "b": hypothesis("zh", "甲乙丙丁戊己庚辛"),
        "c": hypothesis("zh", "甲子丙寅戊己庚辛"),
    }
    (merged,) = find_disputes(hyps, "zh")
    assert merged.distinct == ("乙丙丁", "子丙寅")


def test_spans_two_agreed_units_apart_stay_separate() -> None:
    hyps = {
        "a": hypothesis("zh", "甲乙丙丁戊己"),
        "b": hypothesis("zh", "甲乙丙丁戊己"),
        "c": hypothesis("zh", "子乙丙丁戊寅"),
    }
    assert [d.distinct for d in find_disputes(hyps, "zh")] == [("甲", "子"), ("己", "寅")]


def test_numeric_dispute_is_tagged_and_equal_quantities_do_not_dispute() -> None:
    hyps = {
        "a": hypothesis("zh", "买了一百五十股"),
        "b": hypothesis("zh", "买了150股"),
        "c": hypothesis("zh", "买了一百六十股"),
    }
    (dispute,) = find_disputes(hyps, "zh")
    assert dispute.distinct == ("150", "160")
    assert dispute.touches_quantity


def test_chinese_numeral_is_tagged_through_its_context() -> None:
    hyps = {
        name: timed_units([Segment(text, 0, 5)], "zh", "raw")
        for name, text in {"a": "总共一百股", "b": "总共一百股", "c": "总共二百股"}.items()
    }
    (dispute,) = find_disputes(hyps, "zh")
    assert dispute.touches_quantity


def test_latin_dispute_is_tagged() -> None:
    hyps = {
        "a": hypothesis("zh", "看NVDA的财报"),
        "b": hypothesis("zh", "看NVDA的财报"),
        "c": hypothesis("zh", "看AMD的财报"),
    }
    (dispute,) = find_disputes(hyps, "zh")
    assert dispute.distinct == ("nvda", "amd")
    assert dispute.touches_latin
    assert not dispute.touches_quantity


def test_time_range_covers_every_reading() -> None:
    hyps = {
        "a": hypothesis("zh", "甲乙丙", "丁戊己"),
        "b": hypothesis("zh", "甲乙丙", "丁戊己"),
        "c": hypothesis("zh", "甲乙丙", "丁子己"),
    }
    (dispute,) = find_disputes(hyps, "zh")
    assert dispute.start_s == pytest.approx(10 + 10 / 3)
    assert dispute.end_s == pytest.approx(10 + 20 / 3)


def test_context_is_capped_per_language() -> None:
    long_zh = "字" * 100
    zh = {n: hypothesis("zh", long_zh + t + long_zh) for n, t in {"a": "甲", "b": "甲", "c": "乙"}.items()}
    (zh_dispute,) = find_disputes(zh, "zh")
    assert len(zh_dispute.left_context) == len(zh_dispute.right_context) == 40
    words = " ".join(f"w{i}" for i in range(50))
    en = {n: hypothesis("en", f"{words} {t} {words}") for n, t in {"a": "x", "b": "x", "c": "y"}.items()}
    (en_dispute,) = find_disputes(en, "en")
    assert len(en_dispute.left_context.split()) == len(en_dispute.right_context.split()) == 12


def test_pivot_is_the_medoid_and_results_do_not_depend_on_input_order() -> None:
    texts = {"a": "甲乙丙丁", "b": "甲乙丙戊", "c": "甲乙丙丁", "d": "甲己丙丁"}
    forward = find_disputes({n: hypothesis("zh", t) for n, t in texts.items()}, "zh")
    backward = find_disputes({n: hypothesis("zh", t) for n, t in reversed(texts.items())}, "zh")
    assert forward == backward
    assert [r.hypothesis for r in forward[0].readings] == ["a", "b", "c", "d"]


def test_four_way_readings_are_deduplicated() -> None:
    hyps = {n: hypothesis("zh", f"甲{t}丙") for n, t in {"a": "乙", "b": "丁", "c": "乙", "d": "戊"}.items()}
    (dispute,) = find_disputes(hyps, "zh")
    assert dispute.distinct == ("乙", "丁", "戊")


def test_screening_stats_follow_the_plurality() -> None:
    hyps = {
        "a": hypothesis("zh", "我买了很多的甲股票"),
        "b": hypothesis("zh", "我买了很多的甲股票"),
        "c": hypothesis("zh", "我卖了很多的乙股票"),
    }
    disputes = find_disputes(hyps, "zh")
    stats = screening_stats(disputes, ["a", "b", "c"])
    assert stats["a"].agreement == 1.0
    assert stats["c"].agreement == 0.0
    assert stats["c"].disputes == 2


def test_undecided_disputes_count_for_nobody() -> None:
    hyps = {"a": hypothesis("zh", "买"), "b": hypothesis("zh", "卖")}
    stats = screening_stats(find_disputes(hyps, "zh"), ["a", "b"])
    assert stats["a"].decided == 0
    assert stats["a"].agreement is None


def test_window_units_use_the_midpoint() -> None:
    units = hypothesis("zh", "甲乙", "丙丁")
    assert [u.text for u in window_units(units, 0, 10)] == ["甲", "乙"]
    assert [u.text for u in window_units(units, 5, 15)] == ["乙", "丙"]
