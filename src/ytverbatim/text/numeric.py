"""Numeric verification: check every quantity in the transcript against independent readings of the same audio.

Spans are the quantity tokens of `ytverbatim.text.numbers` and nothing else, so ordinary words that merely contain a
numeral character (一样, 一直) are never checked. Each span collects the readings that were actually heard at its time
range: the transcript itself, every caption track and an optional second-engine decode, plus the engine's word
confidence. Readings that agree under inverse text normalization need nothing. Readings that disagree go to an LLM that
may only *choose among them*. A number changes only to a value an independent source heard for that same span, with the
same kind and unit: the choice is applied only when it is confident, lands on a listed reading that was anchored to the
span on both sides, keeps the transcript's kind and unit, and changes nothing but data characters. Everything else
becomes a flag on that one occurrence, never a mask over other ones.

A span whose evidence cannot be aligned simply has fewer readings; nothing here raises on heuristic ambiguity. A
malformed LLM reply degrades to a flag rather than aborting the run, unlike correction: this pass surfaces doubt, and a
flag is the honest outcome when the judge is unusable.
"""

import hashlib
import json
import logging
import math
import re
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from rapidfuzz.distance import Levenshtein

from ytverbatim import atomic
from ytverbatim.acquire.captions import CaptionKind, CaptionTrack
from ytverbatim.asr.chunks import Chunk
from ytverbatim.asr.models import ChunkHypothesis, FileJob, WorkerRequest
from ytverbatim.asr.protocol import AsrContext, EngineDef
from ytverbatim.contract.models import Edit, Flag, LLMUsage, Reading, Word
from ytverbatim.errors import LLMOutputError
from ytverbatim.llm.jsonio import complete_json
from ytverbatim.llm.protocol import LLMClient, LLMRequest
from ytverbatim.llm.usage import UsageMeter
from ytverbatim.text import offsets
from ytverbatim.text.guards import check_numeric_fix
from ytverbatim.text.numbers import QToken, quantity_tokens

_log = logging.getLogger(__name__)

# Part of every checkpoint key: bump it when the prompt changes meaning so cached verdicts are not reused.
PROMPT_VERSION = "numeric-v1"
STAGE = "numeric"
BATCH_SIZE = 8
# Whisper word probability below which a span heard by one source only is also flagged `low_confidence`.
LOW_CONFIDENCE = 0.35
CONTEXT_NEIGHBORS = 2
# Caption and second-engine timings drift from the transcript's by up to a second or so; untimed cues and chunks are
# interpolated, which drifts more. A wider window costs only some extra unmatched units around the segment.
WINDOW_PAD_S = 2.0
SECOND_OPINION_PAD_S = 0.5
# Re-decoding the short pause between two nearby numeric segments is cheaper than a separate chunk and gives the second
# engine more context.
MERGE_GAP_S = 3.0
# Below this share of matching plain units the window is not the same speech, so pairing its numbers would be noise.
_MIN_ANCHOR_RATIO = 0.4
_MIN_ANCHOR_UNITS = 4
_MAX_OUTPUT_TOKENS = 4096
_MARK_OPEN, _MARK_CLOSE = "⟦", "⟧"
NO_LLM_NOTE = "independent readings disagree and no LLM was configured to choose among them; re-verify"

_CJK = re.compile(r"[一-鿿]")
_PLAIN_UNIT = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[一-鿿]")
_SPACE_BY_CJK = re.compile(r"\s+(?=[一-鿿])|(?<=[一-鿿])\s+")
_VALUE_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SURFACE_NUMBER = re.compile(r"[0-9０-９][0-9０-９,]*(?:\.[0-9０-９]+)?|\.[0-9０-９]+")
# Digits and Chinese numerals write the same kind of quantity; only the script differs.
_SAME_KIND = {"cjk_quantity": "number"}

_SYSTEM = (
    "You are a meticulous financial speech-to-text auditor. Each item is a number, ticker symbol, or amount in a "
    "transcript segment where independent transcriptions of the same audio disagree (e.g. 十五 vs 五十, $1500 vs "
    "$1000). The segment is shown with the disputed span marked " + _MARK_OPEN + "like this" + _MARK_CLOSE + ", "
    "between its neighbouring segments. Choose ONLY among the listed readings: return the one the surrounding spoken "
    "context unambiguously supports, copied exactly as listed. Never propose a value that is not listed, never reword "
    "prose, and never change anything other than the disputed span. If the context does not clearly favour one "
    'reading, say so with "value": null rather than guess. Use "confidence": "high" only when the context leaves no '
    "reasonable doubt."
)


Status = Literal["agreed", "disputed", "single_source"]


@dataclass(frozen=True, slots=True)
class TimedText:
    """Text with absolute start and end times in seconds: a transcript segment, a decoded chunk or a word."""

    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class SecondOpinion:
    """Another engine's decode of the same audio; ``pieces`` are its words, or its chunks when it has no words."""

    engine: str
    pieces: tuple[TimedText, ...]

    @classmethod
    def from_hypotheses(cls, rows: Iterable[ChunkHypothesis]) -> Self | None:
        """Build from the worker's ``hypotheses.jsonl`` rows of a second-opinion request; None when there are none."""
        ordered = sorted(rows, key=lambda row: row.start)
        if not ordered:
            return None
        engines = {row.hypothesis for row in ordered}
        if len(engines) != 1:
            raise ValueError(f"a second opinion comes from exactly one engine, got {sorted(engines)}")
        pieces: list[TimedText] = []
        for row in ordered:
            if row.words:
                pieces.extend(TimedText(word.text, word.start, word.end) for word in row.words)
            else:
                pieces.append(TimedText(row.text, row.start, row.end))
        return cls(engines.pop(), tuple(pieces))


@dataclass(frozen=True, slots=True)
class Evidence:
    """Independent evidence for the transcript's quantities; every part is optional.

    ``primary`` is the `Reading.source` of the transcript text itself (``asr:<engine-id>``, ``caption_manual``, ...).
    ``words`` are the primary engine's words with absolute times; only their ``conf`` is used. A caption track whose
    source equals ``primary`` is the transcript itself and is ignored.
    """

    primary: str
    words: tuple[Word, ...] = ()
    captions: tuple[CaptionTrack, ...] = ()
    second_opinion: SecondOpinion | None = None


@dataclass(frozen=True, slots=True)
class NumericStats:
    """Spans by status; ``fixed`` counts the disputed spans whose LLM choice was applied."""

    agreed: int = 0
    disputed: int = 0
    single_source: int = 0
    fixed: int = 0


@dataclass(frozen=True, slots=True)
class NumericResult:
    """Per-segment results aligned with the input; flag offsets point into ``texts``."""

    texts: tuple[str, ...]
    edits: tuple[tuple[Edit, ...], ...]
    flags: tuple[tuple[Flag, ...], ...]
    usage: tuple[LLMUsage, ...]
    stats: NumericStats


class _Verdict(BaseModel):
    model_config = ConfigDict(strict=True)
    id: int
    value: str | None
    confidence: Literal["high", "low"]
    reason: str = ""

    @field_validator("value", mode="before")
    @classmethod
    def _number_as_text(cls, value: object) -> object:
        # Models answer a bare number (50) for a numeric reading; one such slip must not discard the whole batch.
        if isinstance(value, int | float) and not isinstance(value, bool):
            return format(Decimal(str(value)).normalize(), "f")
        return value


class _Verdicts(BaseModel):
    model_config = ConfigDict(strict=True)
    verdicts: list[_Verdict]


@dataclass(frozen=True, slots=True)
class _Unit:
    """An alignment unit: a whole quantity token, a CJK character or a Latin word. Offsets index the cleaned text."""

    key: str
    start: int
    end: int
    token: QToken | None


@dataclass(frozen=True, slots=True)
class _Timeline:
    """An independent transcript as units with a time span and a confidence each; ``starts`` never decrease."""

    source: str
    text: str
    units: tuple[_Unit, ...]
    starts: tuple[float, ...]
    ends: tuple[float, ...]
    confs: tuple[float | None, ...]

    def window(self, start: float, end: float) -> range:
        return range(bisect_left(self.starts, start - WINDOW_PAD_S), bisect_right(self.starts, end + WINDOW_PAD_S))


@dataclass(frozen=True, slots=True)
class _Pair:
    """A transcript quantity (unit ``mine`` of ``segment``) and the timeline quantity unit ``theirs`` heard for it.

    ``anchored`` pairs match exactly or sit between matching units on both sides; the others sit in a window's edge
    gap, next to the span rather than pinned to it.
    """

    segment: int
    mine: int
    theirs: int
    anchored: bool


@dataclass(frozen=True, slots=True)
class _Option:
    surface: str
    token: QToken
    anchored: bool


@dataclass
class _Span:
    segment: int
    start: int
    end: int
    surface: str
    readings: list[Reading]
    tokens: list[QToken]
    anchored: list[bool]
    confidence: float | None = None
    # Some alternate source had speech in this span's window, whether or not it produced a reading for the span.
    covered: bool = False

    @property
    def status(self) -> Status:
        if len(self.readings) == 1:
            return "single_source"
        return "agreed" if len({token.value_itn for token in self.tokens}) == 1 else "disputed"

    def options(self) -> list[_Option]:
        """Distinct readings, the transcript's own first; a value is anchored when any source heard it anchored."""
        seen: dict[str, _Option] = {}
        for reading, token, anchored in zip(self.readings, self.tokens, self.anchored, strict=True):
            known = seen.get(token.value_itn)
            if known is None:
                seen[token.value_itn] = _Option(reading.text, token, anchored)
            elif anchored and not known.anchored:
                seen[token.value_itn] = _Option(known.surface, known.token, True)
        return list(seen.values())


@dataclass(frozen=True, slots=True)
class _Decision:
    replacement: str | None
    note: str


def verify_numbers(
    segments: Sequence[TimedText],
    *,
    language: str,
    evidence: Evidence,
    client: LLMClient | None,
    checkpoint_dir: Path,
) -> NumericResult:
    """Verify every quantity in ``segments`` against ``evidence`` and return the flagged, possibly fixed texts.

    Without a ``client`` nothing is adjudicated: every disputed span keeps the transcript's reading and its flag.

    A span heard by the transcript alone is flagged ``numeric_single_source`` only when some alternate source had
    speech in its window yet heard no number there; that is doubt about this one span. When nothing else covered the
    window (a captions-only transcript, or a gap in the captions) the span is flagged ``low_confidence`` alone, and only
    when the engine's word confidence is low: a flag on every number would say nothing about any one of them.
    """
    words = _timeline(evidence.primary, ((w.text, w.start, w.end, w.conf) for w in evidence.words))
    alternates = [
        _timeline(source, ((p.text, p.start, p.end, None) for p in pieces))
        for source, pieces in _alternate_pieces(evidence)
    ]
    units: list[list[_Unit]] = []
    spans: list[dict[int, _Span]] = []
    for index, segment in enumerate(segments):
        clean, offsets = _clean(segment.text)
        units.append(_units(clean))
        spans.append(_segment_spans(index, segment, units[-1], offsets, evidence.primary))
    pairs, _ = _claim(words, segments, units)
    for pair in pairs:
        spans[pair.segment][pair.mine].confidence = words.confs[pair.theirs]
    for alternate in alternates:
        pairs, covered = _claim(alternate, segments, units)
        for index in covered:
            for span in spans[index].values():
                span.covered = True
        for pair in pairs:
            span, unit = spans[pair.segment][pair.mine], alternate.units[pair.theirs]
            if unit.token is None:
                continue
            span.readings.append(Reading(source=alternate.source, text=alternate.text[unit.start : unit.end]))
            span.tokens.append(unit.token)
            span.anchored.append(pair.anchored)
    ordered = [span for by_unit in spans for span in by_unit.values()]
    meter = UsageMeter()
    disputed = [span for span in ordered if span.status == "disputed"]
    if client is None:
        decisions = {index: _Decision(None, NO_LLM_NOTE) for index in range(len(disputed))}
    else:
        decisions = _decide(
            disputed,
            segments,
            language=language,
            client=meter.metered(STAGE, client),
            checkpoint_dir=checkpoint_dir,
        )
    texts, edits, flags, fixed = _apply(segments, ordered, decisions, client_id=client.id if client else None)
    counts = Counter(span.status for span in ordered)
    stats = NumericStats(
        agreed=counts["agreed"], disputed=counts["disputed"], single_source=counts["single_source"], fixed=fixed
    )
    _log.info(
        "numeric: %d spans (%d agreed, %d disputed of which %d fixed, %d single-source)",
        len(ordered),
        stats.agreed,
        stats.disputed,
        stats.fixed,
        stats.single_source,
    )
    return NumericResult(texts=texts, edits=edits, flags=flags, usage=meter.summary(), stats=stats)


def remap_flags(flags: Sequence[Flag], edits: Sequence[Edit]) -> tuple[Flag, ...]:
    """Move one segment's ``flags`` from the text before ``edits`` to the text after them, applied in order.

    Numeric fixes change lengths, so flags an earlier stage raised (``entity_corrected`` from correction) must follow.
    A flag that overlaps a replaced run stretches over its replacement.
    """
    return offsets.remap_flags(flags, ((edit.before, edit.after) for edit in edits))


def second_opinion_request(
    segments: Sequence[TimedText],
    *,
    engine: EngineDef,
    file_id: str,
    wav: Path,
    out_dir: Path,
    language: str | None,
    context: AsrContext | None = None,
    words: Sequence[Word] = (),
) -> WorkerRequest | None:
    """Build a worker request that re-decodes only the audio around segments holding a quantity, or None if none do.

    Nearby segments share a chunk. A range longer than the engine's chunk limit is cut between two units of the
    transcript, preferring the widest pause, and never inside a quantity; ``words`` (the primary engine's timed words)
    place those cuts on real pauses, and without them unit times are interpolated across each segment. The engine must
    decode chunks: a whole-file engine cannot be pointed at just the spans.
    """
    max_chunk_s = engine.spec.max_chunk_s
    if max_chunk_s is None:
        raise ValueError(f"{engine.spec.id} decodes whole files, so it cannot re-decode just the numeric spans")
    ranges = sorted(
        (max(0.0, s.start - SECOND_OPINION_PAD_S), s.end + SECOND_OPINION_PAD_S)
        for s in segments
        if quantity_tokens(s.text)
    )
    if not ranges:
        return None
    pieces = [(w.text, w.start, w.end, None) for w in words] or [(s.text, s.start, s.end, None) for s in segments]
    chunks = _chunks(ranges, max_chunk_s, _timeline("", pieces))
    job = FileJob(id=file_id, wav=str(wav), language=language, context=context, chunks=chunks)
    return WorkerRequest(engine=engine, out_dir=str(out_dir), files=(job,))


def _chunks(ranges: list[tuple[float, float]], max_chunk_s: float, timeline: _Timeline) -> tuple[Chunk, ...]:
    merged: list[tuple[float, float]] = []
    for start, end in ranges:
        if merged and start - merged[-1][1] <= MERGE_GAP_S and max(end, merged[-1][1]) - merged[-1][0] <= max_chunk_s:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            continue
        start = max(start, merged[-1][1]) if merged else start
        if end > start:
            merged.append((start, end))
    chunks: list[Chunk] = []
    for first, end in merged:
        start = first
        while end - start > max_chunk_s:
            cut, pause = _cut(timeline, start + max_chunk_s / 2, start + max_chunk_s)
            chunks.append(Chunk(start, cut, forced_cut=not pause))
            start = cut
        chunks.append(Chunk(start, end))
    return tuple(chunks)


def _cut(timeline: _Timeline, earliest: float, latest: float) -> tuple[float, bool]:
    """Find the widest gap between two units within [earliest, latest], the latest on a tie, and whether it pauses.

    A quantity is a single unit, so no cut lands inside one. With no unit boundary in reach the cut falls at ``latest``.
    """
    best: tuple[float, float] | None = None
    first = max(1, bisect_left(timeline.starts, earliest))
    for k in range(first, bisect_right(timeline.starts, latest)):
        gap_start, gap_end = timeline.ends[k - 1], timeline.starts[k]
        cut = (gap_start + gap_end) / 2
        if earliest <= cut <= latest and (best is None or gap_end - gap_start >= best[0]):
            best = (gap_end - gap_start, cut)
    if best is None:
        return latest, False
    return best[1], best[0] > 0


def _alternate_pieces(evidence: Evidence) -> list[tuple[str, tuple[TimedText, ...]]]:
    sources: list[tuple[str, tuple[TimedText, ...]]] = []
    for track in evidence.captions:
        source = "caption_manual" if track.kind is CaptionKind.MANUAL else "caption_auto"
        if source == evidence.primary:
            continue
        pieces: list[TimedText] = []
        for cue in track.segments:
            if cue.words:
                pieces.extend(TimedText(w.text, w.start, w.end) for w in cue.words)
            else:
                pieces.append(TimedText(cue.text, cue.start, cue.end))
        sources.append((source, tuple(pieces)))
    if evidence.second_opinion is not None:
        sources.append((f"asr2:{evidence.second_opinion.engine}", evidence.second_opinion.pieces))
    return sources


def _clean(text: str) -> tuple[str, list[int]]:
    """Drop whitespace next to CJK (150 期 reads as 150期) and map each kept character to its original offset."""
    kept: list[int] = []
    cursor = 0
    for match in _SPACE_BY_CJK.finditer(text):
        kept.extend(range(cursor, match.start()))
        cursor = match.end()
    kept.extend(range(cursor, len(text)))
    return "".join(text[i] for i in kept), kept


def _units(text: str) -> list[_Unit]:
    units: list[_Unit] = []
    cursor = 0

    def plain(stop: int) -> None:
        for match in _PLAIN_UNIT.finditer(text, cursor, stop):
            units.append(_Unit(match[0].casefold(), match.start(), match.end(), None))

    for token in quantity_tokens(text):
        plain(token.char_start)
        # Keyed by value, so 一百五十期 aligns with 150期 as the same unit.
        units.append(_Unit(f"\x00{token.value_itn}", token.char_start, token.char_end, token))
        cursor = token.char_end
    plain(len(text))
    return units


def _timeline(source: str, pieces: Iterable[tuple[str, float, float, float | None]]) -> _Timeline:
    kept = sorted(
        ((piece, start, end, conf) for raw, start, end, conf in pieces if (piece := _clean(raw)[0]).strip()),
        key=lambda item: item[1],
    )
    joined = "".join(piece for piece, *_ in kept)
    # Whisper words carry their own leading space and Chinese is written without spaces, so only stripped Latin pieces
    # (caption words, chunk texts) need one; inserting it elsewhere would split 15 + 00 into two numbers.
    add_spaces = not any(piece[0].isspace() for piece, *_ in kept[1:]) and not _mostly_cjk(joined)
    text: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    confs: list[float | None] = []
    # Rolling caption cues overlap; clamping keeps character times in order so windows can bisect them.
    floor = -math.inf
    for piece, start, end, conf in kept:
        if add_spaces and text and not (text[-1][-1].isspace() or _CJK.match(text[-1][-1]) or _CJK.match(piece[0])):
            text.append(" ")
            starts.append(max(start, floor))
            ends.append(starts[-1])
            confs.append(None)
        step = max(end - start, 0.0) / len(piece)
        for k in range(len(piece)):
            floor = max(start + step * k, floor)
            starts.append(floor)
            ends.append(max(start + step * (k + 1), floor))
        text.append(piece)
        confs.extend(conf for _ in piece)
    full = "".join(text)
    units = _units(full)
    return _Timeline(
        source=source,
        text=full,
        units=tuple(units),
        starts=tuple(starts[u.start] for u in units),
        ends=tuple(ends[u.end - 1] for u in units),
        confs=tuple(min((c for c in confs[u.start : u.end] if c is not None), default=None) for u in units),
    )


def _mostly_cjk(text: str) -> bool:
    non_space = sum(1 for ch in text if not ch.isspace())
    return non_space > 0 and len(_CJK.findall(text)) >= 0.5 * non_space


def _segment_spans(
    index: int, segment: TimedText, units: list[_Unit], offsets: list[int], primary: str
) -> dict[int, _Span]:
    spans: dict[int, _Span] = {}
    for k, unit in enumerate(units):
        if unit.token is None:
            continue
        start, end = offsets[unit.start], offsets[unit.end - 1] + 1
        surface = segment.text[start:end]
        spans[k] = _Span(index, start, end, surface, [Reading(source=primary, text=surface)], [unit.token], [True])
    return spans


def _claim(
    timeline: _Timeline, segments: Sequence[TimedText], units: list[list[_Unit]]
) -> tuple[list[_Pair], set[int]]:
    """Pair every segment's quantities with ``timeline``'s; also return the segments whose window it covered.

    Windows overlap, so one heard number could otherwise back two spans: each timeline unit backs at most one. Anchored
    pairs claim their units first, in segment order; edge pairs may only take units nobody anchored.
    """
    candidates: list[_Pair] = []
    covered: set[int] = set()
    for index, segment in enumerate(segments):
        if not any(unit.token is not None for unit in units[index]):
            continue
        window = timeline.window(segment.start, segment.end)
        if not window:
            continue
        covered.add(index)
        theirs = [timeline.units[w] for w in window]
        for mine, (j, anchored) in _pair_quantities(units[index], theirs).items():
            candidates.append(_Pair(index, mine, window[j], anchored))
    claimed: set[int] = set()
    pairs: list[_Pair] = []
    for tier in (True, False):
        for pair in candidates:
            if pair.anchored is tier and pair.theirs not in claimed:
                claimed.add(pair.theirs)
                pairs.append(pair)
    return pairs, covered


def _pair_quantities(mine: Sequence[_Unit], theirs: Sequence[_Unit]) -> dict[int, tuple[int, bool]]:
    """Map quantity units of ``mine`` to the quantity units of ``theirs`` heard in their place, and whether anchored.

    Equal units pair as anchored. Between two runs of matching units, quantities pair in order, also anchored, only
    when both sides hold the same number of them; anything else is left unpaired rather than guessed. Before the first
    and after the last match, ``theirs`` also holds the window's padding, so there the quantities nearest the match pair
    up, unanchored. A window with no match, or one that barely matches, pairs nothing.
    """
    if not theirs:
        return {}
    pairs: dict[int, tuple[int, bool]] = {}
    anchors = 0
    gaps: list[tuple[list[int], list[int]]] = [([], [])]
    for op in Levenshtein.opcodes([u.key for u in mine], [u.key for u in theirs]):
        if op.tag != "equal":
            gap_mine, gap_theirs = gaps[-1]
            gap_mine.extend(i for i in range(op.src_start, op.src_end) if mine[i].token is not None)
            gap_theirs.extend(j for j in range(op.dest_start, op.dest_end) if theirs[j].token is not None)
            continue
        gaps.append(([], []))
        for offset in range(op.src_end - op.src_start):
            i = op.src_start + offset
            if mine[i].token is None:
                anchors += 1
            else:
                pairs[i] = (op.dest_start + offset, True)
    last = len(gaps) - 1
    if last == 0:
        return {}
    plain = sum(1 for u in mine if u.token is None)
    if plain >= _MIN_ANCHOR_UNITS and anchors < _MIN_ANCHOR_RATIO * plain:
        return {}
    for k, (gap_mine, gap_theirs) in enumerate(gaps):
        n = len(gap_mine)
        if 0 < k < last:
            if n == len(gap_theirs):
                pairs.update((i, (j, True)) for i, j in zip(gap_mine, gap_theirs, strict=True))
        elif 0 < n <= len(gap_theirs):
            nearest = gap_theirs[-n:] if k == 0 else gap_theirs[:n]
            pairs.update((i, (j, False)) for i, j in zip(gap_mine, nearest, strict=True))
    return pairs


def _decide(
    disputed: list[_Span],
    segments: Sequence[TimedText],
    *,
    language: str,
    client: LLMClient,
    checkpoint_dir: Path,
) -> dict[int, _Decision]:
    """Decide each disputed span (keyed by its id in ``disputed``) with batched LLM verdicts."""
    decisions: dict[int, _Decision] = {}
    for first in range(0, len(disputed), BATCH_SIZE):
        batch = disputed[first : first + BATCH_SIZE]
        user = _prompt(batch, segments, language)
        reply = _verdicts(client, user, checkpoint_dir, expected=len(batch))
        verdicts: dict[int, _Verdict] = {}
        if reply is not None:
            for verdict in reply.verdicts:
                verdicts.setdefault(verdict.id, verdict)
        for offset, span in enumerate(batch):
            decisions[first + offset] = _resolve(span, verdicts.get(offset), malformed=reply is None)
    return decisions


def _prompt(batch: list[_Span], segments: Sequence[TimedText], language: str) -> str:
    items: list[dict[str, object]] = []
    for item_id, span in enumerate(batch):
        text = segments[span.segment].text
        marked = text[: span.start] + _MARK_OPEN + text[span.start : span.end] + _MARK_CLOSE + text[span.end :]
        lo = max(0, span.segment - CONTEXT_NEIGHBORS)
        hi = min(len(segments), span.segment + CONTEXT_NEIGHBORS + 1)
        items.append(
            {
                "id": item_id,
                "readings": [option.surface for option in span.options()],
                "before": [segments[j].text for j in range(lo, span.segment)],
                "segment": marked,
                "after": [segments[j].text for j in range(span.segment + 1, hi)],
            }
        )
    return (
        f"The transcript language is {language}. For each item, choose the reading the context supports, copied "
        'exactly from its "readings" list, or null. Return JSON {"verdicts": [{"id": <item id>, "value": '
        '"<one listed reading>" | null, "confidence": "high" | "low", "reason": "<short>"}]} with one verdict per '
        "item.\n\nITEMS:\n" + json.dumps(items, ensure_ascii=False)
    )


def _verdicts(client: LLMClient, user: str, checkpoint_dir: Path, *, expected: int) -> _Verdicts | None:
    """One batch's validated verdicts, from the checkpoint when present; None when the reply stayed malformed.

    Only a reply with a verdict for each of the ``expected`` items is checkpointed, so a later run asks again for the
    ones it skipped.
    """
    key = json.dumps([PROMPT_VERSION, client.id, _SYSTEM, user], ensure_ascii=False)
    path = checkpoint_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
    if path.exists():
        try:
            return _Verdicts.model_validate_json(path.read_bytes())
        except ValidationError:
            # Only this module writes these files, atomically; an unreadable one is stale, and redoing it is safe.
            _log.warning("numeric: ignoring an unreadable checkpoint %s", path.name)
    request = LLMRequest(system=_SYSTEM, user=user, max_output_tokens=_MAX_OUTPUT_TOKENS, temperature=0.0)
    try:
        verdicts = complete_json(client, request, _Verdicts)
    except LLMOutputError as exc:
        # Not checkpointed, so a later run asks again.
        _log.warning("numeric: no usable verdicts for a batch, flagging it instead: %s", exc)
        return None
    if not set(range(expected)) <= {verdict.id for verdict in verdicts.verdicts}:
        _log.warning("numeric: a batch reply skipped some items; using it without a checkpoint")
        return verdicts
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    atomic.write_json(path, verdicts.model_dump())
    return verdicts


def _resolve(span: _Span, verdict: _Verdict | None, *, malformed: bool) -> _Decision:
    if verdict is None:
        return _Decision(None, "LLM reply malformed; re-verify" if malformed else "LLM gave no verdict; re-verify")
    reason = verdict.reason.strip()[:200]
    if verdict.value is None or verdict.confidence != "high":
        return _Decision(None, reason or "LLM could not tell the readings apart")
    chosen = _match_option(verdict.value, span.options())
    if chosen is None:
        return _Decision(None, f"LLM proposed {verdict.value!r}, which no source heard; rejected")
    mine = span.tokens[0]
    if chosen.token.value_itn == mine.value_itn:
        return _Decision(None, f"LLM kept the transcript reading: {reason}" if reason else "LLM kept the transcript")
    if not chosen.anchored:
        return _Decision(None, f"LLM chose {chosen.surface!r}, which was heard beside this span, not for it; re-verify")
    if not _same_kind_and_unit(mine, chosen.token):
        return _Decision(None, f"LLM chose {chosen.surface!r}, whose kind or unit differs from the transcript's")
    if _magnitude_only(mine, chosen.token):
        return _Decision(None, f"LLM chose {chosen.surface!r}, which differs only in magnitude; re-verify")
    return _Decision(_restyle(mine, chosen.token.value_itn) or chosen.surface, reason)


def _match_option(proposed: str, options: list[_Option]) -> _Option | None:
    proposed = proposed.strip()
    for option in options:
        if proposed == option.surface:
            return option
    tokens = quantity_tokens(proposed)
    if len(tokens) == 1 and tokens[0].surface == proposed:
        for option in options:
            if tokens[0].value_itn == option.token.value_itn:
                return option
    return None


def _same_kind_and_unit(mine: QToken, theirs: QToken) -> bool:
    """15% may become 50%, never 50; $15 billion never 15 billion; 一百五十期 may become 160期, never 160天."""
    if _SAME_KIND.get(mine.kind, mine.kind) != _SAME_KIND.get(theirs.kind, theirs.kind):
        return False
    return mine.kind == "ticker" or _unit(mine.value_itn) == _unit(theirs.value_itn)


def _magnitude_only(mine: QToken, theirs: QToken) -> bool:
    """Values a power of ten apart ($15 billion vs $15 million, 1.5万亿 vs 1.5亿, 15K vs 15M).

    A misheard scale word is the costliest numeric error and context rarely settles it, so it is never auto-fixed.
    """
    a, b = _VALUE_NUMBER.search(mine.value_itn), _VALUE_NUMBER.search(theirs.value_itn)
    if mine.kind == "ticker" or a is None or b is None or not Decimal(a[0]) or not Decimal(b[0]):
        return False
    ratio = (max(Decimal(a[0]), Decimal(b[0])) / min(Decimal(a[0]), Decimal(b[0]))).normalize()
    return ratio != 1 and ratio.as_tuple().digits == (1,)


def _unit(value: str) -> str:
    return _VALUE_NUMBER.sub("#", value.lstrip("+-"))


def _restyle(mine: QToken, value: str) -> str | None:
    """Write ``value`` in the transcript's own form: 15% heard elsewhere as 50 percent becomes 50%.

    Only a token written with digits has a form to keep; a Chinese-numeral span takes the chosen reading as heard.
    """
    written = _SURFACE_NUMBER.search(mine.surface)
    old, new = _VALUE_NUMBER.search(mine.value_itn), _VALUE_NUMBER.search(value)
    if mine.kind == "ticker" or written is None or old is None or new is None:
        return None
    digits = written[0].translate({0xFF10 + i: ord("0") + i for i in range(10)}).replace(",", "")
    if digits == old[0]:
        number = new[0]
    elif Decimal(digits) and Decimal(old[0]):
        # The surface carries a magnitude (3.5亿, $15 billion), so the digits written are the value over that scale.
        number = format((Decimal(new[0]) * Decimal(digits) / Decimal(old[0])).normalize(), "f")
    else:
        return None
    return mine.surface[: written.start()] + number + mine.surface[written.end() :]


def _apply(
    segments: Sequence[TimedText],
    spans: list[_Span],
    decisions: dict[int, _Decision],
    *,
    client_id: str | None,
) -> tuple[tuple[str, ...], tuple[tuple[Edit, ...], ...], tuple[tuple[Flag, ...], ...], int]:
    texts = [segment.text for segment in segments]
    edits: list[list[Edit]] = [[] for _ in segments]
    flags: list[list[Flag]] = [[] for _ in segments]
    shift = [0] * len(segments)
    fixed = 0
    disputed_id = 0
    for span in spans:
        index = span.segment
        start, end = span.start + shift[index], span.end + shift[index]
        status = span.status
        if status == "agreed":
            continue
        if status == "single_source":
            flags[index].extend(_single_source_flags(span, start, end))
            continue
        decision = decisions[disputed_id]
        disputed_id += 1
        current = texts[index]
        if decision.replacement is not None:
            candidate = current[:start] + decision.replacement + current[end:]
            if check_numeric_fix(current, candidate, [span.surface]):
                texts[index] = candidate
                edits[index].append(Edit(stage="numeric", before=current, after=candidate, llm=client_id))
                new_end = start + len(decision.replacement)
                note = f"{span.surface} → {decision.replacement}" + (f": {decision.note}" if decision.note else "")
                flags[index].append(
                    Flag(
                        kind="numeric_fixed",
                        char_start=start,
                        char_end=new_end,
                        note=note,
                        readings=tuple(span.readings),
                    )
                )
                shift[index] += new_end - end
                fixed += 1
                continue
            decision = _Decision(None, "the chosen reading would change more than the number; rejected")
        flags[index].append(
            Flag(
                kind="numeric_disputed",
                char_start=start,
                char_end=end,
                note=decision.note,
                readings=tuple(span.readings),
            )
        )
    return (
        tuple(texts),
        tuple(tuple(e) for e in edits),
        tuple(tuple(f) for f in flags),
        fixed,
    )


def _single_source_flags(span: _Span, start: int, end: int) -> list[Flag]:
    conf = span.confidence
    low = conf is not None and conf < LOW_CONFIDENCE
    result: list[Flag] = []
    if span.covered:
        note = "no other source heard a number here" + (f"; engine confidence {conf:.2f}" if conf is not None else "")
        result.append(
            Flag(kind="numeric_single_source", char_start=start, char_end=end, note=note, readings=tuple(span.readings))
        )
    if low:
        result.append(Flag(kind="low_confidence", char_start=start, char_end=end, note=f"engine confidence {conf:.2f}"))
    return result
