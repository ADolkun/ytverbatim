"""Find the spans where ASR hypotheses disagree.

Consensus is not a reference. It only locates disputes (for the human A/B) and produces screening
statistics; the winner is decided by people, never by what most engines agree on.
"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from evals.normalize import Language, View, join_units, to_units
from ytverbatim.text.numbers import quantity_tokens

CONTEXT_UNITS: Mapping[Language, int] = {"zh": 40, "en": 12}
MERGE_GAP_UNITS = 1
MIN_HYPOTHESES = 2

_LATIN = re.compile(r"[A-Za-z]")


@dataclass(frozen=True, slots=True)
class Segment:
    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class TimedUnit:
    text: str
    start_s: float
    end_s: float

    @property
    def midpoint_s(self) -> float:
        return (self.start_s + self.end_s) / 2


@dataclass(frozen=True, slots=True)
class Reading:
    hypothesis: str
    text: str
    units: tuple[str, ...]
    start_s: float | None
    end_s: float | None


@dataclass(frozen=True, slots=True)
class Dispute:
    index: int
    readings: tuple[Reading, ...]
    distinct: tuple[str, ...]
    left_context: str
    right_context: str
    start_s: float
    end_s: float
    touches_quantity: bool
    touches_latin: bool


@dataclass(frozen=True, slots=True)
class HypothesisScreen:
    disputes: int
    decided: int
    agreed_with_plurality: int

    @property
    def agreement(self) -> float | None:
        return self.agreed_with_plurality / self.decided if self.decided else None


def timed_units(segments: Sequence[Segment], language: Language, view: View = "itn") -> list[TimedUnit]:
    """Split segments into units with times spread evenly across each segment.

    Chunk and caption hypotheses carry no word times, so an even spread is the honest estimate.
    """
    out: list[TimedUnit] = []
    for segment in segments:
        units = to_units(segment.text, language, view)
        step = (segment.end_s - segment.start_s) / len(units) if units else 0.0
        out.extend(
            TimedUnit(unit, segment.start_s + i * step, segment.start_s + (i + 1) * step)
            for i, unit in enumerate(units)
        )
    return out


def window_units(units: Sequence[TimedUnit], start_s: float, end_s: float) -> list[TimedUnit]:
    """Units whose midpoint lies in ``[start_s, end_s)``."""
    return [u for u in units if start_s <= u.midpoint_s < end_s]


def _medoid(texts: Mapping[str, list[str]]) -> str:
    def total(name: str) -> tuple[int, str]:
        return sum(Levenshtein.distance(texts[name], texts[other]) for other in texts if other != name), name

    return min(texts, key=total)


# Column 2k is the gap before pivot unit k (the last one is after the final unit) and column
# 2k + 1 is pivot unit k. Each cell lists the indices of the hypothesis units sitting in it.
Columns = list[dict[str, list[int]]]


def _columns(texts: Mapping[str, list[str]], pivot: str) -> Columns:
    pivot_len = len(texts[pivot])
    columns: Columns = [{name: [] for name in texts} for _ in range(2 * pivot_len + 1)]
    for k in range(pivot_len):
        columns[2 * k + 1][pivot] = [k]
    for name, units in texts.items():
        if name == pivot:
            continue
        for op in Levenshtein.opcodes(texts[pivot], units):
            if op.tag in ("equal", "replace"):
                for offset in range(op.src_end - op.src_start):
                    columns[2 * (op.src_start + offset) + 1][name] = [op.dest_start + offset]
            elif op.tag == "insert":
                columns[2 * op.src_start][name] = list(range(op.dest_start, op.dest_end))
    return columns


def _disputed_runs(columns: Columns, texts: Mapping[str, list[str]]) -> list[tuple[int, int]]:
    disputed = [
        c
        for c, cell in enumerate(columns)
        if len({tuple(texts[name][i] for i in idx) for name, idx in cell.items()}) > 1
    ]
    runs: list[tuple[int, int]] = []
    for c in disputed:
        if runs:
            first, last = runs[-1]
            if c // 2 - (last + 1) // 2 <= MERGE_GAP_UNITS:
                runs[-1] = (first, c)
                continue
        runs.append((c, c))
    return runs


def _touches_quantity(units: Sequence[str], left: Sequence[str], right: Sequence[str]) -> bool:
    # The context is scanned too, so a lone 一 followed by 百 still reads as 一百.
    mid_len = len(join_units(list(units)))
    start = len(join_units([*left, *units])) - mid_len
    text = join_units([*left, *units, *right])
    return any(t.char_start < start + mid_len and t.char_end > start for t in quantity_tokens(text))


def find_disputes(
    hypotheses: Mapping[str, Sequence[TimedUnit]],
    language: Language,
) -> list[Dispute]:
    """Align every hypothesis to the medoid hypothesis and return the disputed spans in order.

    A dispute is a maximal run of columns where the readings differ, merged with its neighbour when
    at most ``MERGE_GAP_UNITS`` agreed units separate them. The context comes from the pivot.
    """
    if len(hypotheses) < MIN_HYPOTHESES:
        raise ValueError("finding disputes needs at least two hypotheses")
    names = sorted(hypotheses)
    timed = {name: list(hypotheses[name]) for name in names}
    texts = {name: [u.text for u in timed[name]] for name in names}
    pivot = _medoid(texts)
    columns = _columns(texts, pivot)
    context = CONTEXT_UNITS[language]
    disputes: list[Dispute] = []
    for first, last in _disputed_runs(columns, texts):
        pivot_lo, pivot_hi = first // 2, (last + 1) // 2
        left = tuple(texts[pivot][max(0, pivot_lo - context) : pivot_lo])
        right = tuple(texts[pivot][pivot_hi : pivot_hi + context])
        readings = []
        for name in names:
            indices = [i for column in columns[first : last + 1] for i in column[name]]
            units = tuple(texts[name][i] for i in indices)
            readings.append(
                Reading(
                    name,
                    join_units(list(units)),
                    units,
                    timed[name][indices[0]].start_s if indices else None,
                    timed[name][indices[-1]].end_s if indices else None,
                )
            )
        starts = [r.start_s for r in readings if r.start_s is not None]
        ends = [r.end_s for r in readings if r.end_s is not None]
        disputes.append(
            Dispute(
                index=len(disputes),
                readings=tuple(readings),
                distinct=tuple(dict.fromkeys(r.text for r in readings)),
                left_context=join_units(list(left)),
                right_context=join_units(list(right)),
                start_s=min(starts),
                end_s=max(ends),
                touches_quantity=any(_touches_quantity(r.units, left, right) for r in readings if r.units),
                touches_latin=language == "zh" and any(_LATIN.search(u) for r in readings for u in r.units),
            )
        )
    return disputes


def screening_stats(disputes: Sequence[Dispute], hypotheses: Sequence[str]) -> dict[str, HypothesisScreen]:
    """Per hypothesis: how often it sides with the plurality reading across the given disputes.

    A dispute without a unique most-common reading is undecided and counts for nobody. Engines
    that share training data vote alike, so this screens candidates and never ranks the final pair.
    """
    totals = {name: [0, 0, 0] for name in hypotheses}
    for dispute in disputes:
        votes = Counter(r.text for r in dispute.readings).most_common(2)
        plurality = votes[0][0] if len(votes) == 1 or votes[0][1] > votes[1][1] else None
        for reading in dispute.readings:
            row = totals.get(reading.hypothesis)
            if row is None:
                continue
            row[0] += 1
            if plurality is not None:
                row[1] += 1
                row[2] += reading.text == plurality
    return {name: HypothesisScreen(row[0], row[1], row[2]) for name, row in totals.items()}
