"""Error rates, quantity accuracy, A/B win-rate intervals and hallucination screens.

Error rates need a real reference (the manual captions of one video). Nothing here builds a
reference out of the hypotheses themselves.
"""

import math
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import jiwer

from evals.consensus import TimedUnit
from evals.normalize import Language, View, canonical, normalize, to_units
from ytverbatim.text.numbers import QKind, QToken, quantity_tokens

Interval = tuple[float, float]

DEFAULT_CONFIDENCE = 0.95
PARITY = 0.5
DEFAULT_RESAMPLES = 2000
MIN_NONSPEECH_S = 1.0
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    hits: int

    @property
    def reference_units(self) -> int:
        return self.substitutions + self.deletions + self.hits

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.errors / self.reference_units


def _counts(output: jiwer.WordOutput | jiwer.CharacterOutput) -> ErrorCounts:
    return ErrorCounts(output.substitutions, output.deletions, output.insertions, output.hits)


def unit_error_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> ErrorCounts:
    """Edit counts over pre-split units. Units must not contain spaces."""
    if not reference:
        raise ValueError("the reference has no units")
    return _counts(jiwer.process_words(" ".join(reference), " ".join(hypothesis)))


def cer(reference: str, hypothesis: str, language: Language, view: View) -> ErrorCounts:
    """Character error rate on the normalized text with whitespace removed."""
    ref, hyp = (_WHITESPACE.sub("", "".join(to_units(t, language, view))) for t in (reference, hypothesis))
    if not ref:
        raise ValueError("the reference has no characters")
    return _counts(jiwer.process_characters(ref, hyp))


def wer(reference: str, hypothesis: str, language: Language, view: View) -> ErrorCounts:
    """Word error rate on whitespace-separated words of the normalized text."""
    return unit_error_counts(
        normalize(reference, language, view).split(),
        normalize(hypothesis, language, view).split(),
    )


def mer(reference: str, hypothesis: str, language: Language, view: View) -> ErrorCounts:
    """Mixed-unit error rate: each CJK character and each Latin or digit run is one unit.

    This is not jiwer's match error rate, which divides by hits plus errors and not by the
    reference length.
    """
    return unit_error_counts(to_units(reference, language, view), to_units(hypothesis, language, view))


@dataclass(frozen=True, slots=True)
class Proportion:
    successes: int
    trials: int

    @property
    def rate(self) -> float:
        return self.successes / self.trials


def _tokens(text: str, language: Language, kinds: frozenset[QKind] | None) -> list[QToken]:
    return [t for t in quantity_tokens(canonical(text, language)) if kinds is None or t.kind in kinds]


def quantity_accuracy(
    reference: str,
    hypothesis: str,
    language: Language,
    view: View,
    kinds: frozenset[QKind] | None = None,
) -> Proportion:
    """Share of the reference's quantity tokens the hypothesis reproduces, as a multiset.

    The raw view compares surfaces case-sensitively (``150期`` is not ``一百五十期``, ``nvda`` is
    not ``NVDA``); the itn view compares token values and casefolds.
    """
    expected = _tokens(reference, language, kinds)
    found = _tokens(hypothesis, language, kinds)
    if view == "raw":
        return Proportion(_overlap([t.surface for t in expected], [t.surface for t in found]), len(expected))
    values = _overlap(
        [t.value_itn.casefold() for t in expected if t.kind != "ticker"],
        [t.value_itn.casefold() for t in found if t.kind != "ticker"],
    )
    return Proportion(values + _ticker_hits(expected, hypothesis, language), len(expected))


def _overlap(expected: list[str], found: list[str]) -> int:
    return sum((Counter(expected) & Counter(found)).values())


def _ticker_hits(expected: list[QToken], hypothesis: str, language: Language) -> int:
    # A casefolded ticker (nvda) is no longer recognizable as one, so look the reference's up in the text.
    text = normalize(hypothesis, language, "itn")
    wanted = Counter(normalize(t.surface, language, "itn") for t in expected if t.kind == "ticker")
    return sum(
        min(count, len(re.findall(rf"(?<![A-Za-z0-9]){re.escape(key)}(?![A-Za-z0-9])", text)))
        for key, count in wanted.items()
    )


def wilson_interval(successes: int, trials: int, confidence: float = DEFAULT_CONFIDENCE) -> Interval:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError(f"need 0 <= successes <= trials and trials > 0, got {successes}/{trials}")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


@dataclass(frozen=True, slots=True)
class WinRate:
    challenger_wins: int
    control_wins: int
    low: float
    high: float

    @property
    def decisive(self) -> int:
        return self.challenger_wins + self.control_wins

    @property
    def rate(self) -> float:
        return self.challenger_wins / self.decisive

    @property
    def excludes_parity(self) -> bool:
        return self.low > PARITY or self.high < PARITY


def win_rate(challenger_wins: int, control_wins: int, confidence: float = DEFAULT_CONFIDENCE) -> WinRate:
    """Return the challenger's win rate among decisive A/B trials, with a Wilson interval.

    "Both wrong", "neither" and replays are not decisive and stay out of both counts.
    """
    low, high = wilson_interval(challenger_wins, challenger_wins + control_wins, confidence)
    return WinRate(challenger_wins, control_wins, low, high)


@dataclass(frozen=True, slots=True)
class RepetitionRun:
    start: int
    period: int
    repeats: int


def repetition_runs(units: Sequence[str], min_repeats: int = 4, max_period: int = 8) -> list[RepetitionRun]:
    """Maximal runs where a block of ``period`` units repeats back to back at least ``min_repeats`` times.

    At each position the block covering the most units wins, so a doubled phrase is not also
    reported as its own halves.
    """
    runs: list[RepetitionRun] = []
    i = 0
    while i < len(units):
        best = RepetitionRun(i, 1, 0)
        for period in range(1, max_period + 1):
            repeats = 1
            while units[i + repeats * period : i + (repeats + 1) * period] == units[i : i + period]:
                repeats += 1
            if repeats >= min_repeats and repeats * period > best.repeats * best.period:
                best = RepetitionRun(i, period, repeats)
        if best.repeats:
            runs.append(best)
            i += best.repeats * best.period
        else:
            i += 1
    return runs


def nonspeech_intervals(
    speech: Sequence[Interval], duration_s: float, min_length_s: float = MIN_NONSPEECH_S
) -> list[Interval]:
    """Gaps between speech regions of at least ``min_length_s``, including the file's head and tail."""
    gaps: list[Interval] = []
    cursor = 0.0
    for start, end in sorted(speech):
        if start - cursor >= min_length_s:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration_s - cursor >= min_length_s:
        gaps.append((cursor, duration_s))
    return gaps


def units_in_nonspeech(units: Sequence[TimedUnit], nonspeech: Sequence[Interval]) -> list[TimedUnit]:
    """Hypothesis units whose midpoint falls inside a non-speech interval: text the audio does not contain."""
    return [u for u in units if any(start <= u.midpoint_s < end for start, end in nonspeech)]


@dataclass(frozen=True, slots=True)
class Bootstrap:
    mean_difference: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    seed: int,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Bootstrap:
    """Percentile interval of mean(a - b), resampling the paired items together. Deterministic per seed."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and equally long")
    differences = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(resamples))
    tail = (1 - confidence) / 2
    low = means[int(tail * resamples)]
    high = means[min(resamples - 1, int((1 - tail) * resamples))]
    return Bootstrap(sum(differences) / n, low, high)
