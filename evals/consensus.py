"""Find the spans where ASR hypotheses disagree.

Consensus is not a reference and nothing here votes. It only locates disputes, for the blind judge
and the human A/B; the winner is decided by people, never by what most engines agree on.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from evals.normalize import Language, View, english_normalizer, join_units, to_units
from ytverbatim.text.numbers import quantity_tokens

CONTEXT_UNITS: Mapping[Language, int] = {"zh": 40, "en": 12}
MERGE_GAP_UNITS = 1
MIN_HYPOTHESES = 2
# Stands in for a disputed stretch inside a context, so the context never carries any one engine's wording.
UNCERTAIN = "[?]"

_LATIN = re.compile(r"[A-Za-z]")
_ACRONYM = re.compile(r"[A-Z].*[A-Z]")
_CAPITALIZED = re.compile(r"[A-Z]")


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
    """One hypothesis's wording of a dispute; ``before_s``/``after_s`` time its neighbouring units, even when empty."""

    hypothesis: str
    text: str
    units: tuple[str, ...]
    start_s: float | None
    end_s: float | None
    before_s: float | None = None
    after_s: float | None = None


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


def timed_units(
    segments: Sequence[Segment],
    language: Language,
    view: View = "raw",
    speech: Sequence[tuple[float, float]] = (),
) -> list[TimedUnit]:
    """Split segments into units with times spread evenly across each segment's speech.

    Chunk and caption hypotheses carry no word times, so an even spread is the honest estimate. A chunk spans
    pauses, though, so with ``speech`` (the VAD regions) the units spread over the speech inside the segment only;
    otherwise units would be timed into the pauses and read as text decoded in silence. A unit is kept inside the
    speech region that holds its midpoint. The default raw view keeps surface form and Latin case, which the
    numeric and entity flags of a dispute read.
    """
    out: list[TimedUnit] = []
    for segment in segments:
        units = to_units(segment.text, language, view)
        if not units:
            continue
        regions = _speech_within(segment, speech)
        talk = sum(end - start for start, end in regions)
        step = talk / len(units)
        for i, unit in enumerate(units):
            lo, hi = i * step, (i + 1) * step
            region = _region_at(regions, (lo + hi) / 2)
            out.append(TimedUnit(unit, max(_at(regions, lo), region[0]), min(_at(regions, hi), region[1])))
    return out


def _speech_within(segment: Segment, speech: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    inside = [
        (max(start, segment.start_s), min(end, segment.end_s))
        for start, end in sorted(speech)
        if min(end, segment.end_s) > max(start, segment.start_s)
    ]
    return inside or [(segment.start_s, segment.end_s)]


def _at(regions: Sequence[tuple[float, float]], offset: float) -> float:
    """Return the time reached after ``offset`` seconds of speech."""
    for start, end in regions:
        if offset <= end - start:
            return start + offset
        offset -= end - start
    return regions[-1][1]


def _region_at(regions: Sequence[tuple[float, float]], offset: float) -> tuple[float, float]:
    for start, end in regions:
        if offset < end - start:
            return start, end
        offset -= end - start
    return regions[-1]


def window_units(units: Sequence[TimedUnit], start_s: float, end_s: float) -> list[TimedUnit]:
    """Units whose midpoint lies in ``[start_s, end_s)``."""
    return [u for u in units if start_s <= u.midpoint_s < end_s]


def _medoid(texts: Mapping[str, list[str]], families: Mapping[str, str]) -> str:
    """Pick the hypothesis closest to all others, counting one hypothesis per engine family.

    Two variants of one engine (whole-file and chunked Whisper) are near-duplicates; counting both would pull the
    pivot toward that engine's wording.
    """
    representatives: dict[str, str] = {}
    for name in sorted(texts):
        representatives.setdefault(families.get(name, name), name)
    pool = sorted(representatives.values())

    def total(name: str) -> tuple[int, str]:
        return sum(Levenshtein.distance(texts[name], texts[other]) for other in pool if other != name), name

    return min(pool, key=total)


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


def _is_disputed(cell: dict[str, list[int]], texts: Mapping[str, list[str]]) -> bool:
    return len({tuple(texts[name][i] for i in idx) for name, idx in cell.items()}) > 1


def _disputed_runs(columns: Columns, texts: Mapping[str, list[str]]) -> list[tuple[int, int]]:
    disputed = [c for c, cell in enumerate(columns) if _is_disputed(cell, texts)]
    runs: list[tuple[int, int]] = []
    for c in disputed:
        if runs:
            first, last = runs[-1]
            if c // 2 - (last + 1) // 2 <= MERGE_GAP_UNITS:
                runs[-1] = (first, c)
                continue
        runs.append((c, c))
    return runs


def _agreed_context(columns: Columns, texts: Mapping[str, list[str]], order: range, limit: int) -> list[str]:
    """Up to ``limit`` units walking ``order`` away from a dispute: agreed units, and one placeholder per disputed run.

    Taking the pivot's units instead would show the pivot engine's wording of neighbouring disputes.
    """
    out: list[str] = []
    for c in order:
        if len(out) >= limit:
            break
        cell = columns[c]
        if _is_disputed(cell, texts):
            if not out or out[-1] != UNCERTAIN:
                out.append(UNCERTAIN)
            continue
        name, idx = next(iter(cell.items()))
        out.extend(texts[name][i] for i in idx)
    out = out[:limit]
    return out if order.step > 0 else out[::-1]


def _plain(units: list[str]) -> list[str]:
    # A placeholder must not glue two quantities together, so it becomes a blank unit for the scan.
    return [" " if u == UNCERTAIN else u for u in units]


def _touches_quantity(units: Sequence[str], left: Sequence[str], right: Sequence[str]) -> bool:
    # The context is scanned too, so a lone 一 followed by 百 still reads as 一百.
    mid_len = len(join_units(list(units)))
    start = len(join_units([*left, *units])) - mid_len
    text = join_units([*left, *units, *right])
    return any(t.char_start < start + mid_len and t.char_end > start for t in quantity_tokens(text))


def is_acronym(unit: str) -> bool:
    """Two or more capitals in one token: a ticker or acronym (``NVDA``, ``50WE``), whatever the sentence position."""
    return _ACRONYM.search(unit) is not None


def _touches_entity(units: Sequence[str], left: Sequence[str], language: Language) -> bool:
    # Raw-view units keep case. In Chinese any Latin token is a name or ticker. In English a capital also starts
    # sentences, so beyond acronyms only a capitalized word that follows a lowercase one counts; punctuation is gone,
    # so that is the best available sign of mid-sentence.
    if language == "zh":
        return any(_LATIN.search(u) for u in units)
    previous = [*left[-1:], *units]
    return any(
        is_acronym(u) or (_CAPITALIZED.match(u) and i > 0 and previous[i - 1][:1].islower())
        for i, u in enumerate(previous)
        if i >= len(previous) - len(units)
    )


def _spelled_quantity(text: str) -> bool:
    # English speakers spell numbers out ("fifteen percent"); Whisper's normalizer turns them into digits.
    return any(t.kind != "ticker" for t in quantity_tokens(english_normalizer()(text)))


def find_disputes(
    hypotheses: Mapping[str, Sequence[TimedUnit]],
    language: Language,
    *,
    families: Mapping[str, str] | None = None,
    pivot: str | None = None,
) -> list[Dispute]:
    """Align every hypothesis to a pivot hypothesis and return the disputed spans in order.

    The pivot is ``pivot`` when given (with two hypotheses that is a direct pairwise alignment), else the medoid
    over one hypothesis per family in ``families`` (default: every hypothesis its own family). Pass raw-view
    units: the numeric and entity flags read the readings' surface form.

    A dispute is a maximal run of columns where the readings differ, merged with its neighbour when
    at most ``MERGE_GAP_UNITS`` agreed units separate them. The context holds only units every
    hypothesis agrees on, with ``UNCERTAIN`` marking each other disputed run inside it.
    """
    if len(hypotheses) < MIN_HYPOTHESES:
        raise ValueError("finding disputes needs at least two hypotheses")
    names = sorted(hypotheses)
    timed = {name: list(hypotheses[name]) for name in names}
    texts = {name: [u.text for u in timed[name]] for name in names}
    if pivot is None:
        pivot = _medoid(texts, families or {})
    elif pivot not in texts:
        raise ValueError(f"pivot {pivot!r} is not one of the hypotheses")
    columns = _columns(texts, pivot)
    context = CONTEXT_UNITS[language]
    disputes: list[Dispute] = []
    for first, last in _disputed_runs(columns, texts):
        left = _agreed_context(columns, texts, range(first - 1, -1, -1), context)
        right = _agreed_context(columns, texts, range(last + 1, len(columns)), context)
        readings = []
        for name in names:
            indices = [i for column in columns[first : last + 1] for i in column[name]]
            units = tuple(texts[name][i] for i in indices)
            before = [i for column in columns[:first] for i in column[name]]
            after = [i for column in columns[last + 1 :] for i in column[name]]
            readings.append(
                Reading(
                    name,
                    join_units(list(units)),
                    units,
                    timed[name][indices[0]].start_s if indices else None,
                    timed[name][indices[-1]].end_s if indices else None,
                    timed[name][before[-1]].end_s if before else None,
                    timed[name][after[0]].start_s if after else None,
                )
            )
        starts = [r.start_s for r in readings if r.start_s is not None]
        ends = [r.end_s for r in readings if r.end_s is not None]
        disputes.append(
            Dispute(
                index=len(disputes),
                readings=tuple(readings),
                distinct=tuple(dict.fromkeys(r.text for r in readings)),
                left_context=join_units(left),
                right_context=join_units(right),
                start_s=min(starts),
                end_s=max(ends),
                touches_quantity=any(
                    _touches_quantity(r.units, _plain(left), _plain(right))
                    or (language == "en" and _spelled_quantity(r.text))
                    for r in readings
                    if r.units
                ),
                touches_latin=any(_touches_entity(r.units, left, language) for r in readings),
            )
        )
    return disputes
