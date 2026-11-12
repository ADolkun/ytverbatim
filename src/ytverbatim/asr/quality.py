"""Decoder-failure detectors: pure functions over text, timed units and word timings.

Mandarin and English are both supported. A text counts as Mandarin when at least 30% of its letters are CJK,
the same rule the aligner uses to pick a language. That decides how repetition is matched: English phrases are
compared word by word, while Mandarin has no word separators and ASR output punctuates and spaces it
inconsistently, so every separator is dropped before matching.
"""

import re
import unicodedata
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ytverbatim.contract.models import Word

DEFAULT_MAX_GAP_S = 30.0
# Whisper's own fallback threshold: text that zlib shrinks this much is a loop, not speech.
COMPRESSION_RATIO_THRESHOLD = 2.4
NONLEXICAL_DENSE_MIN_RECORDS = 10
NONLEXICAL_DENSE_MIN_SECONDS = 5.0

AnomalyKind = Literal[
    "replacement_character", "intra_segment_repetition", "cross_segment_repetition", "word_timing_collapse"
]

_CJK_SHARE = 0.3
_NONLEXICAL_JOIN_GAP_S = 1.0
_SHORT_RECORD_S = 0.05
_DECODER_REPEAT_MIN_LETTERS = 32
_DECODER_REPEAT_MIN_COUNT = 8
_DECODER_REPEAT_MIN_SHORT_RECORDS = 3
_WORD_TIMING_MIN_RECORDS = 10
_DECODER_REPEAT = re.compile(r"(.{1,80}?)\1{7,}", re.DOTALL)
_SEPARATORS = re.compile(r"[\W_]+")
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


@dataclass(frozen=True, slots=True)
class Unit:
    """A timed piece of output: a chunk, a segment or a single word. Word times share the unit's clock."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()


@dataclass(frozen=True, slots=True)
class Anomaly:
    kind: AnomalyKind
    start: float
    end: float
    count: int


@dataclass(frozen=True, slots=True)
class NonlexicalRun:
    """Consecutive units with no letters or digits. `observed_seconds` excludes the silence between them."""

    start: float
    end: float
    records: int
    observed_seconds: float


def has_lexical_content(text: str) -> bool:
    return any(character.isalnum() for character in text)


def compression_ratio(text: str) -> float:
    """UTF-8 bytes over zlib-compressed bytes, exactly as Whisper computes it."""
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data))


def is_cjk_text(text: str) -> bool:
    folded = unicodedata.normalize("NFKC", text)
    letters = sum(character.isalpha() for character in folded)
    return letters > 0 and len(_CJK.findall(folded)) >= _CJK_SHARE * letters


def normalize_for_repetition(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _SEPARATORS.sub("" if is_cjk_text(folded) else " ", folded).strip()


def nonlexical_runs(units: Sequence[Unit]) -> list[NonlexicalRun]:
    """Group letterless units into runs, split at lexical text or a gap over one second."""
    groups: list[list[tuple[float, float]]] = []
    active: list[tuple[float, float]] = []
    for unit in units:
        if has_lexical_content(unit.text):
            active = []
            continue
        start = max(0.0, unit.start)
        end = max(start, unit.end)
        if not active or start > active[-1][1] + _NONLEXICAL_JOIN_GAP_S:
            active = []
            groups.append(active)
        active.append((start, end))
    return [
        NonlexicalRun(
            start=min(start for start, _ in group),
            end=max(end for _, end in group),
            records=len(group),
            observed_seconds=round(sum(end - start for start, end in _interval_union(group)), 3),
        )
        for group in groups
    ]


def is_nonlexical_failure(run: NonlexicalRun, *, max_gap_s: float = DEFAULT_MAX_GAP_S) -> bool:
    """Tell a long bridge of punctuation, or a dense burst of it, from silence: it is decoder output, not speech."""
    dense = run.records >= NONLEXICAL_DENSE_MIN_RECORDS and run.end - run.start >= NONLEXICAL_DENSE_MIN_SECONDS
    return dense or run.observed_seconds > max_gap_s


def cross_segment_repetitions(units: Sequence[Unit]) -> list[Anomaly]:
    """Find a phrase repeated eight or more times with corrupt timing.

    Corrupt means three records of 50 ms or less, or ten instantaneous empty records between the repeats.
    Natural repetition has neither.
    """
    findings: list[Anomaly] = []
    run = _Repeat()
    pending_empty = 0
    for unit in units:
        text = normalize_for_repetition(unit.text)
        if not text:
            pending_empty += int(unit.end == unit.start)
            continue
        if text != run.phrase or unit.start > run.end + _NONLEXICAL_JOIN_GAP_S:
            if run.is_loop():
                findings.append(run.finding())
            run = _Repeat(phrase=text, start=unit.start)
        else:
            # Trailing empty records alone cannot turn natural repetition into a loop.
            run.empty += pending_empty
        pending_empty = 0
        run.end = unit.end
        run.count += 1
        run.short += int(unit.start <= unit.end <= unit.start + _SHORT_RECORD_S)
    if run.is_loop():
        findings.append(run.finding())
    return findings


def word_timing_collapses(units: Sequence[Unit]) -> list[Anomaly]:
    """Find ten or more lexical words with 32 letters, all zero-length at one instant: a decoder that lost its clock.

    A run continues across unit seams, but a unit with text and no word records is unknown evidence and ends it.
    """
    findings: list[Anomaly] = []
    burst = _Burst()
    for unit in units:
        if not unit.words and has_lexical_content(unit.text):
            if burst.is_collapse():
                findings.append(burst.finding())
            burst = _Burst()
            continue
        for word in unit.words:
            text = unicodedata.normalize("NFKC", word.text)
            if not has_lexical_content(text):
                continue
            co_timed = word.start == word.end
            if not co_timed or word.start != burst.instant:
                if burst.is_collapse():
                    findings.append(burst.finding())
                burst = _Burst()
            if co_timed:
                burst.add(unit, text, word.start)
    if burst.is_collapse():
        findings.append(burst.finding())
    return findings


def intra_segment_repetitions(text: str) -> list[int]:
    """Count the repeats of every phrase repeated at least eight times with at least 32 letters in the run."""
    # The trailing space lets a final English repeat match like the others.
    normalized = normalize_for_repetition(text) + " "
    return [
        len(match[0]) // len(match[1])
        for match in _DECODER_REPEAT.finditer(normalized)
        if _letters(match[0]) >= _DECODER_REPEAT_MIN_LETTERS
    ]


def decoder_anomalies(units: Sequence[Unit]) -> list[Anomaly]:
    """Find corrupt glyphs and extreme repetition within or across units, without duplicates."""
    findings = [*cross_segment_repetitions(units), *word_timing_collapses(units)]
    for unit in units:
        if "�" in unit.text:
            findings.append(Anomaly("replacement_character", unit.start, unit.end, unit.text.count("�")))
        findings.extend(
            Anomaly("intra_segment_repetition", unit.start, unit.end, count)
            for count in intra_segment_repetitions(unit.text)
        )
    return list(dict.fromkeys(findings))


def _letters(text: str) -> int:
    return sum(character.isalpha() for character in text)


def _interval_union(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


@dataclass(slots=True)
class _Repeat:
    phrase: str = ""
    start: float = 0.0
    end: float = 0.0
    count: int = 0
    short: int = 0
    empty: int = 0

    def is_loop(self) -> bool:
        return (
            self.count >= _DECODER_REPEAT_MIN_COUNT
            and self.count * _letters(self.phrase) >= _DECODER_REPEAT_MIN_LETTERS
            and (self.short >= _DECODER_REPEAT_MIN_SHORT_RECORDS or self.empty >= NONLEXICAL_DENSE_MIN_RECORDS)
        )

    def finding(self) -> Anomaly:
        return Anomaly("cross_segment_repetition", self.start, self.end, self.count)


@dataclass(slots=True)
class _Burst:
    start: float = 0.0
    end: float = 0.0
    count: int = 0
    letters: int = 0
    instant: float | None = None

    def add(self, unit: Unit, text: str, instant: float) -> None:
        if self.count:
            self.start, self.end = min(self.start, unit.start), max(self.end, unit.end)
        else:
            self.start, self.end = unit.start, unit.end
        self.count += 1
        self.letters += _letters(text)
        self.instant = instant

    def is_collapse(self) -> bool:
        return self.count >= _WORD_TIMING_MIN_RECORDS and self.letters >= _DECODER_REPEAT_MIN_LETTERS

    def finding(self) -> Anomaly:
        return Anomaly("word_timing_collapse", self.start, self.end, self.count)
