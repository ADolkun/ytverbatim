import pytest

from evals.consensus import (
    UNCERTAIN,
    Segment,
    TimedUnit,
    find_disputes,
    timed_units,
    window_units,
)
from evals.normalize import Language, View


def hypothesis(language: Language, *segments: str, step: float = 10.0, view: View = "itn") -> list[TimedUnit]:
    return timed_units([Segment(t, i * step, (i + 1) * step) for i, t in enumerate(segments)], language, view)


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


def test_window_units_use_the_midpoint() -> None:
    units = hypothesis("zh", "甲乙", "丙丁")
    assert [u.text for u in window_units(units, 0, 10)] == ["甲", "乙"]
    assert [u.text for u in window_units(units, 5, 15)] == ["乙", "丙"]


def test_english_entities_and_spelled_numbers_are_flagged_on_the_raw_view() -> None:
    entity = {
        "a": hypothesis("en", "we bought NVDA today", view="raw"),
        "b": hypothesis("en", "we bought and video today", view="raw"),
    }
    (dispute,) = find_disputes(entity, "en")
    assert dispute.touches_latin
    spelled = {
        "a": hypothesis("en", "it rose fifteen percent today", view="raw"),
        "b": hypothesis("en", "it rose fifty percent today", view="raw"),
    }
    (numeric,) = find_disputes(spelled, "en")
    assert numeric.touches_quantity
    plain = {"a": hypothesis("en", "The cat sat", view="raw"), "b": hypothesis("en", "The cap sat", view="raw")}
    (neither,) = find_disputes(plain, "en")
    assert not neither.touches_latin
    assert not neither.touches_quantity


def test_the_pivot_counts_one_hypothesis_per_family() -> None:
    texts = {"w1": "甲乙丙丁", "w2": "甲乙丙丁", "q": "甲乙戊丁", "f": "甲乙戊丁", "m": "甲己丙丁"}
    hyps = {n: hypothesis("zh", t) for n, t in texts.items()}
    families = {"w1": "whisper", "w2": "whisper", "q": "qwen", "f": "fire", "m": "mimo"}
    # Counting both Whisper variants makes a Whisper the medoid; with one per family the q/f wording wins.
    unweighted = find_disputes(hyps, "zh")
    per_family = find_disputes(hyps, "zh", families=families)
    assert [r.text for r in unweighted[0].readings] == [r.text for r in per_family[0].readings]
    from evals.consensus import _medoid

    units = {n: [u.text for u in h] for n, h in hyps.items()}
    assert _medoid(units, {}) in {"w1", "w2"}
    assert _medoid(units, families) in {"q", "f"}


def test_an_explicit_pivot_aligns_two_hypotheses_directly() -> None:
    hyps = {"control": hypothesis("zh", "我们买了股票"), "challenger": hypothesis("zh", "我们卖了股票")}
    (dispute,) = find_disputes(hyps, "zh", pivot="control")
    assert set(dispute.distinct) == {"买", "卖"}
    with pytest.raises(ValueError, match="pivot"):
        find_disputes(hyps, "zh", pivot="other")


def test_context_holds_only_agreed_units_with_placeholders_for_other_disputes() -> None:
    hyps = {
        "a": hypothesis("zh", "我们今天买了这只股票然后卖出"),
        "b": hypothesis("zh", "我们明天买了这只股票然后卖出"),
        "c": hypothesis("zh", "我们今天买了这支股票然后卖掉"),
    }
    disputes = find_disputes(hyps, "zh")
    middle = next(d for d in disputes if set(d.distinct) == {"只", "支"})
    assert middle.left_context == f"我们{UNCERTAIN}天买了这"
    assert middle.right_context == f"股票然后卖{UNCERTAIN}"
    for d in disputes:
        for reading in d.distinct:
            if reading not in ("天", ""):
                assert reading not in d.left_context.replace(UNCERTAIN, "")


def test_readings_record_their_neighbouring_unit_times() -> None:
    hyps = {"a": hypothesis("en", "red green blue", view="raw"), "b": hypothesis("en", "red blue", view="raw")}
    (dispute,) = find_disputes(hyps, "en")
    empty = next(r for r in dispute.readings if r.hypothesis == "b")
    assert empty.start_s is None
    assert empty.before_s == pytest.approx(5.0)
    assert empty.after_s == pytest.approx(5.0)
    spoken = next(r for r in dispute.readings if r.hypothesis == "a")
    assert (spoken.start_s, spoken.end_s) == pytest.approx((10 / 3, 20 / 3))


def test_units_spread_over_speech_only() -> None:
    units = timed_units([Segment("甲乙丙丁", 0.0, 10.0)], "zh", speech=[(0.0, 2.0), (8.0, 10.0)])
    assert [(u.start_s, u.end_s) for u in units] == [(0.0, 1.0), (1.0, 2.0), (8.0, 9.0), (9.0, 10.0)]
    straddling = timed_units([Segment("甲乙丙", 0.0, 10.0)], "zh", speech=[(0.0, 2.0), (8.0, 10.0)])
    assert all(not 2.0 < u.midpoint_s < 8.0 for u in straddling)
    no_speech = timed_units([Segment("甲乙", 0.0, 4.0)], "zh", speech=[(10.0, 12.0)])
    assert [(u.start_s, u.end_s) for u in no_speech] == [(0.0, 2.0), (2.0, 4.0)]
