import re
from pathlib import Path

import jiwer
import pytest

from evals import metrics
from evals.consensus import TimedUnit
from evals.metrics import (
    ErrorCounts,
    _percentile_index,
    cer,
    mer,
    nonspeech_intervals,
    paired_bootstrap,
    quantity_accuracy,
    repetition_runs,
    unit_error_counts,
    units_in_nonspeech,
    wer,
    wilson_interval,
    win_rate,
)


def test_wer_against_hand_count() -> None:
    counts = wer("the cat sat on the mat", "the sit on the mat today", "en", "raw")
    assert (counts.substitutions, counts.deletions, counts.insertions, counts.hits) == (1, 1, 1, 4)
    assert counts.reference_units == 6
    assert counts.rate == pytest.approx(0.5)


def test_cer_against_hand_count() -> None:
    counts = cer("今天天气很好", "今天天汽很", "zh", "raw")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 1, 0)
    assert counts.rate == pytest.approx(2 / 6)


def test_cer_ignores_spacing_and_punctuation() -> None:
    assert cer("今天，天气 很好。", "今天天气很好", "zh", "raw").errors == 0


def test_mer_counts_latin_runs_as_one_unit() -> None:
    counts = mer("买了NVDA和AMD", "卖了nvda和AMD", "zh", "raw")
    assert (counts.substitutions, counts.deletions, counts.insertions, counts.hits) == (2, 0, 0, 3)
    assert counts.rate == pytest.approx(0.4)


def test_mer_in_the_itn_view_forgives_case_and_numeral_style() -> None:
    assert mer("买了一百五十股NVDA", "买了150股nvda", "zh", "raw").errors > 0
    assert mer("买了一百五十股NVDA", "买了150股nvda", "zh", "itn").errors == 0


def test_raw_view_counts_a_lowercased_ticker_as_an_error() -> None:
    assert mer("看NVDA", "看nvda", "zh", "raw").errors == 1


def test_unit_error_counts_on_presplit_units() -> None:
    counts = unit_error_counts(["a", "b", "c"], ["a", "c"])
    assert (counts.deletions, counts.hits) == (1, 2)


def test_empty_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="no units"):
        unit_error_counts([], ["a"])
    with pytest.raises(ValueError, match="no characters"):
        cer("。", "好", "zh", "raw")


def test_empty_hypothesis_is_all_deletions() -> None:
    assert unit_error_counts(["a", "b"], []).rate == 1.0


def test_jiwer_match_error_rate_is_never_called(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> float:
        raise AssertionError("jiwer.mer is the match error rate, not the mixed-unit error rate")

    monkeypatch.setattr(jiwer, "mer", forbidden)
    mer("买了NVDA", "卖了NVDA", "zh", "raw")
    wer("a b", "a c", "en", "raw")
    cer("好的", "好", "zh", "raw")
    source = Path(metrics.__file__).read_text(encoding="utf-8")
    assert not re.search(r"jiwer\.mer\b|import\b[^\n]*\bmer\b", source.replace("def mer(", ""))


def test_quantity_accuracy_raw_view_needs_the_same_surface() -> None:
    reference = "买了一百五十股，涨了3.5%，NVDA"
    hypothesis = "买了150股，涨了3.5%，nvda"
    raw = quantity_accuracy(reference, hypothesis, "zh", "raw")
    assert (raw.successes, raw.trials) == (1, 3)
    itn = quantity_accuracy(reference, hypothesis, "zh", "itn")
    assert (itn.successes, itn.trials) == (3, 3)
    assert itn.rate == 1.0


def test_quantity_accuracy_counts_a_multiset_and_can_filter_kinds() -> None:
    result = quantity_accuracy("3% 3% 4%", "3% 4% 4%", "en", "raw")
    assert (result.successes, result.trials) == (2, 3)
    only_tickers = quantity_accuracy("NVDA 5%", "AMD 5%", "en", "raw", frozenset({"ticker"}))
    assert (only_tickers.successes, only_tickers.trials) == (0, 1)


def test_quantity_accuracy_itn_matches_spelled_english_numbers() -> None:
    itn = quantity_accuracy("Sales rose 15% at NVDA", "sales rose fifteen percent at NVDA", "en", "itn")
    assert (itn.successes, itn.trials) == (2, 2)
    spelled_reference = quantity_accuracy("fifteen percent", "15%", "en", "itn")
    assert (spelled_reference.successes, spelled_reference.trials) == (1, 1)
    raw = quantity_accuracy("Sales rose 15%", "sales rose fifteen percent", "en", "raw")
    assert (raw.successes, raw.trials) == (0, 1)


@pytest.mark.parametrize(
    ("successes", "trials", "low", "high"),
    [(8, 10, 0.4902, 0.9433), (0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0), (50, 100, 0.4038, 0.5962)],
)
def test_wilson_interval_known_values(successes: int, trials: int, low: float, high: float) -> None:
    got_low, got_high = wilson_interval(successes, trials)
    assert got_low == pytest.approx(low, abs=1e-4)
    assert got_high == pytest.approx(high, abs=1e-4)


def test_wilson_interval_rejects_impossible_counts() -> None:
    for successes, trials in [(1, 0), (5, 4), (-1, 4)]:
        with pytest.raises(ValueError, match="trials"):
            wilson_interval(successes, trials)


def test_win_rate_excludes_parity_only_when_the_interval_clears_half() -> None:
    clear = win_rate(120, 30)
    assert clear.rate == pytest.approx(0.8)
    assert clear.excludes_parity
    assert not win_rate(80, 70).excludes_parity
    assert win_rate(10, 140).excludes_parity


def test_repetition_runs() -> None:
    units = list("xabcabcabcabcy")
    (run,) = repetition_runs(units)
    assert (run.start, run.period, run.repeats) == (1, 3, 4)
    assert repetition_runs(list("aaaabaaa")) == [metrics.RepetitionRun(0, 1, 4)]
    assert repetition_runs(list("ababab")) == []
    assert repetition_runs(list("ababab"), min_repeats=3) == [metrics.RepetitionRun(0, 2, 3)]


def test_nonspeech_intervals_include_head_tail_and_skip_short_gaps() -> None:
    speech = [(3.0, 5.0), (5.5, 8.0), (12.0, 13.0)]
    assert nonspeech_intervals(speech, 20.0) == [(0.0, 3.0), (8.0, 12.0), (13.0, 20.0)]
    assert nonspeech_intervals([(0.0, 10.0)], 10.5) == []


def test_units_in_nonspeech_uses_midpoints() -> None:
    units = [TimedUnit("a", 0.0, 1.0), TimedUnit("b", 2.2, 3.0), TimedUnit("c", 2.9, 3.5), TimedUnit("d", 9.0, 10.0)]
    assert [u.text for u in units_in_nonspeech(units, [(0.0, 3.0), (8.0, 12.0)])] == ["a", "b", "d"]
    assert units_in_nonspeech(units, []) == []


def counts(errors: int, reference: int) -> ErrorCounts:
    return ErrorCounts(substitutions=errors, deletions=0, insertions=0, hits=reference - errors)


def test_paired_bootstrap_is_deterministic_per_seed() -> None:
    a = [counts(e, 100) for e in (10, 12, 8, 15, 11, 9, 14, 13)]
    b = [counts(e, 100) for e in (22, 20, 19, 31, 21, 30, 20, 24)]
    first = paired_bootstrap(a, b, seed=7)
    assert first == paired_bootstrap(a, b, seed=7)
    assert first != paired_bootstrap(a, b, seed=8)
    assert first.low < first.difference < first.high < 0
    assert first.excludes_zero


def test_paired_bootstrap_interval_includes_zero_for_noise() -> None:
    a = [counts(e, 10) for e in (1, 3, 2, 4)]
    b = [counts(e, 10) for e in (3, 1, 4, 2)]
    assert not paired_bootstrap(a, b, seed=1).excludes_zero


def test_paired_bootstrap_pools_counts_instead_of_averaging_window_rates() -> None:
    # A one-unit window where a is wrong must not outweigh a 99-unit window where a is perfect.
    a = [counts(1, 1), counts(0, 99)]
    b = [counts(0, 1), counts(5, 99)]
    result = paired_bootstrap(a, b, seed=0)
    assert result.difference == pytest.approx(1 / 100 - 5 / 100)
    assert result.difference < 0  # a per-window mean would say (1.0 - 0.05) / 2 > 0


def test_percentile_indices_pick_the_right_ranks() -> None:
    assert _percentile_index(0.025, 2000) == 49
    assert _percentile_index(0.975, 2000) == 1949
    assert _percentile_index(0.5, 3) == 1


def test_paired_bootstrap_rejects_mismatched_samples() -> None:
    with pytest.raises(ValueError, match="equally long"):
        paired_bootstrap([counts(1, 2)], [counts(1, 2), counts(0, 2)], seed=0)
