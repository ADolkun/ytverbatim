"""Numeric verification: check every quantity in the transcript against independent readings of the same audio.

Spans are the quantity tokens of `ytverbatim.text.numbers` and nothing else, so ordinary words that merely contain a
numeral character (一样, 一直) are never checked. Each span collects the readings that were actually heard at its time
range: the transcript itself, every caption track and an optional second-engine decode, plus the engine's word
confidence. Readings that agree under inverse text normalization need nothing. Readings that disagree go to an LLM that
may only *choose among them*; its choice is applied only when it is confident, lands on a listed reading and changes
nothing but data characters. Everything else becomes a flag on that one occurrence, never a mask over other ones.

A span whose evidence cannot be aligned simply has fewer readings; nothing here raises on heuristic ambiguity. A
malformed LLM reply degrades to a flag rather than aborting the run, unlike correction: this pass surfaces doubt, and a
flag is the honest outcome when the judge is unusable.
"""

import hashlib
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError
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
from ytverbatim.text.guards import (
    _BALLOON_FLOOR,
    _BALLOON_RATIO,
    _BARE_SYMBOL,
    _CJK,
    _CJK_NUMERAL,
    _CJK_NUMERAL_CHARS,
    _DIGIT_RUN,
    MIN_KEEP_RATIO,
    _is_predominantly_cjk,
)
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
# Below this share of matching plain units the window is not the same speech, so pairing its numbers would be noise.
_MIN_ANCHOR_RATIO = 0.4
_MIN_ANCHOR_UNITS = 4
_MAX_OUTPUT_TOKENS = 4096
_MARK_OPEN, _MARK_CLOSE = "⟦", "⟧"

# A numeric fix may touch only data characters; any other character in a changed run means a reword rode along.
_NON_DATA_CHAR = re.compile(rf"[^\s0-9０-９.,%％$¥€£A-Z{_CJK_NUMERAL_CHARS}]")
_PLAIN_UNIT = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[一-鿿]")
_SPACE_BY_CJK = re.compile(r"\s+(?=[一-鿿])|(?<=[一-鿿])\s+")

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
    def from_hypotheses(cls, rows: Iterable[ChunkHypothesis]) -> Self:
        """Build from the worker's ``hypotheses.jsonl`` rows of a second-opinion request."""
        ordered = sorted(rows, key=lambda row: row.start)
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
    """An independent transcript as units with a time and a confidence each."""

    source: str
    text: str
    units: tuple[_Unit, ...]
    times: tuple[float, ...]
    confs: tuple[float | None, ...]

    def window(self, start: float, end: float) -> list[int]:
        return [k for k, t in enumerate(self.times) if start - WINDOW_PAD_S <= t <= end + WINDOW_PAD_S]


@dataclass
class _Span:
    segment: int
    start: int
    end: int
    surface: str
    readings: list[Reading]
    values: list[str]
    confidence: float | None = None

    @property
    def status(self) -> Status:
        if len(self.readings) == 1:
            return "single_source"
        return "agreed" if len(set(self.values)) == 1 else "disputed"

    def options(self) -> list[tuple[str, str]]:
        """Distinct readings as (surface, value), the transcript's own first."""
        seen: dict[str, str] = {}
        for reading, value in zip(self.readings, self.values, strict=True):
            seen.setdefault(value, reading.text)
        return [(surface, value) for value, surface in seen.items()]


@dataclass(frozen=True, slots=True)
class _Decision:
    replacement: str | None
    note: str


def verify_numbers(
    segments: Sequence[TimedText],
    *,
    language: str,
    evidence: Evidence,
    client: LLMClient,
    checkpoint_dir: Path,
) -> NumericResult:
    """Verify every quantity in ``segments`` against ``evidence`` and return the flagged, possibly fixed texts.

    When the evidence holds nothing besides the transcript, every span is single-source and none is flagged: a flag on
    every number would say nothing about any one of them.
    """
    words = _timeline(evidence.primary, ((w.text, w.start, w.end, w.conf) for w in evidence.words))
    alternates = [
        _timeline(source, ((p.text, p.start, p.end, None) for p in pieces))
        for source, pieces in _alternate_pieces(evidence)
    ]
    has_evidence = bool(alternates) or any(w.conf is not None for w in evidence.words)
    spans = [
        span
        for index, segment in enumerate(segments)
        for span in _segment_spans(index, segment, evidence.primary, words, alternates)
    ]
    meter = UsageMeter()
    decisions = _decide(
        [span for span in spans if span.status == "disputed"],
        segments,
        language=language,
        client=meter.metered(STAGE, client),
        checkpoint_dir=checkpoint_dir,
    )
    texts, edits, flags, fixed = _apply(segments, spans, decisions, client_id=client.id, flag_single=has_evidence)
    counts = Counter(span.status for span in spans)
    stats = NumericStats(
        agreed=counts["agreed"], disputed=counts["disputed"], single_source=counts["single_source"], fixed=fixed
    )
    _log.info(
        "numeric: %d spans (%d agreed, %d disputed of which %d fixed, %d single-source)",
        len(spans),
        stats.agreed,
        stats.disputed,
        stats.fixed,
        stats.single_source,
    )
    return NumericResult(texts=texts, edits=edits, flags=flags, usage=meter.summary(), stats=stats)


def second_opinion_request(
    segments: Sequence[TimedText],
    *,
    engine: EngineDef,
    file_id: str,
    wav: Path,
    out_dir: Path,
    language: str | None,
    context: AsrContext | None = None,
) -> WorkerRequest | None:
    """Build a worker request that re-decodes only the audio around segments holding a quantity, or None if none do.

    The engine must decode chunks: a whole-file engine cannot be pointed at just the spans.
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
    job = FileJob(id=file_id, wav=str(wav), language=language, context=context, chunks=_chunks(ranges, max_chunk_s))
    return WorkerRequest(engine=engine, out_dir=str(out_dir), files=(job,))


def _chunks(ranges: list[tuple[float, float]], max_chunk_s: float) -> tuple[Chunk, ...]:
    merged: list[tuple[float, float]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] and end - merged[-1][0] <= max_chunk_s:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            continue
        start = max(start, merged[-1][1]) if merged else start
        if end > start:
            merged.append((start, end))
    chunks: list[Chunk] = []
    for start, end in merged:
        pieces = max(1, math.ceil((end - start) / max_chunk_s))
        step = (end - start) / pieces
        chunks.extend(
            Chunk(start + k * step, end if k == pieces - 1 else start + (k + 1) * step, forced_cut=k < pieces - 1)
            for k in range(pieces)
        )
    return tuple(chunks)


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
    text: list[str] = []
    times: list[float] = []
    confs: list[float | None] = []
    for raw, start, end, conf in pieces:
        piece, _ = _clean(raw)
        if not piece.strip():
            continue
        if text and _needs_space(text[-1][-1], piece[0]):
            text.append(" ")
            times.append(start)
            confs.append(None)
        step = max(end - start, 0.0) / len(piece)
        text.append(piece)
        times.extend(start + step * k for k in range(len(piece)))
        confs.extend(conf for _ in piece)
    joined = "".join(text)
    units = _units(joined)
    return _Timeline(
        source=source,
        text=joined,
        units=tuple(units),
        times=tuple(times[u.start] for u in units),
        confs=tuple(min((c for c in confs[u.start : u.end] if c is not None), default=None) for u in units),
    )


def _needs_space(left: str, right: str) -> bool:
    # Caption words arrive stripped; Whisper words carry their own leading space.
    return not (left.isspace() or right.isspace() or _CJK.match(left) or _CJK.match(right))


def _segment_spans(
    index: int, segment: TimedText, primary: str, words: _Timeline, alternates: list[_Timeline]
) -> list[_Span]:
    clean, offsets = _clean(segment.text)
    units = _units(clean)
    spans: dict[int, _Span] = {}
    for k, unit in enumerate(units):
        if unit.token is None:
            continue
        start, end = offsets[unit.start], offsets[unit.end - 1] + 1
        surface = segment.text[start:end]
        spans[k] = _Span(index, start, end, surface, [Reading(source=primary, text=surface)], [unit.token.value_itn])
    if not spans:
        return []
    if words.units:
        window = words.window(segment.start, segment.end)
        for k, j in _pair_quantities(units, [words.units[w] for w in window]).items():
            spans[k].confidence = words.confs[window[j]]
    for alternate in alternates:
        window = alternate.window(segment.start, segment.end)
        for k, j in _pair_quantities(units, [alternate.units[w] for w in window]).items():
            unit = alternate.units[window[j]]
            if unit.token is not None:
                spans[k].readings.append(Reading(source=alternate.source, text=alternate.text[unit.start : unit.end]))
                spans[k].values.append(unit.token.value_itn)
    return list(spans.values())


def _pair_quantities(mine: Sequence[_Unit], theirs: Sequence[_Unit]) -> dict[int, int]:
    """Map quantity units of ``mine`` to the quantity units of ``theirs`` heard in their place.

    Between two runs of matching units, quantities pair in order only when both sides hold the same number of them;
    anything else is left unpaired rather than guessed. Before the first and after the last match, ``theirs`` also
    holds the window's padding, so there the quantities nearest the match pair up. A window that barely matches pairs
    nothing.
    """
    if not theirs:
        return {}
    pairs: dict[int, int] = {}
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
                pairs[i] = op.dest_start + offset
    plain = sum(1 for u in mine if u.token is None)
    if plain >= _MIN_ANCHOR_UNITS and anchors < _MIN_ANCHOR_RATIO * plain:
        return {}
    last = len(gaps) - 1
    for k, (gap_mine, gap_theirs) in enumerate(gaps):
        n = len(gap_mine)
        if n == len(gap_theirs):
            pairs.update(zip(gap_mine, gap_theirs, strict=True))
        elif last > 0 and 0 < n < len(gap_theirs) and k in (0, last):
            pairs.update(zip(gap_mine, gap_theirs[-n:] if k == 0 else gap_theirs[:n], strict=True))
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
        reply = _verdicts(client, user, checkpoint_dir)
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
                "readings": [surface for surface, _ in span.options()],
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


def _verdicts(client: LLMClient, user: str, checkpoint_dir: Path) -> _Verdicts | None:
    """One batch's validated verdicts, from the checkpoint when present; None when the reply stayed malformed."""
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
    surface, value = chosen
    if value == span.values[0]:
        return _Decision(None, f"LLM kept the transcript reading: {reason}" if reason else "LLM kept the transcript")
    return _Decision(surface, reason)


def _match_option(proposed: str, options: list[tuple[str, str]]) -> tuple[str, str] | None:
    proposed = proposed.strip()
    for surface, value in options:
        if proposed == surface:
            return surface, value
    tokens = quantity_tokens(proposed)
    if len(tokens) == 1 and tokens[0].surface == proposed:
        for surface, value in options:
            if tokens[0].value_itn == value:
                return surface, value
    return None


def _apply(
    segments: Sequence[TimedText],
    spans: list[_Span],
    decisions: dict[int, _Decision],
    *,
    client_id: str,
    flag_single: bool,
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
            if flag_single:
                flags[index].extend(_single_source_flags(span, start, end))
            continue
        decision = decisions[disputed_id]
        disputed_id += 1
        current = texts[index]
        if decision.replacement is not None:
            candidate = current[:start] + decision.replacement + current[end:]
            if _numeric_fix_is_safe(current, candidate, [span.surface]):
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
    note = "heard by one source only" + (f"; engine confidence {conf:.2f}" if conf is not None else "")
    readings = tuple(span.readings)
    result = [Flag(kind="numeric_single_source", char_start=start, char_end=end, note=note, readings=readings)]
    if conf is not None and conf < LOW_CONFIDENCE:
        result.append(Flag(kind="low_confidence", char_start=start, char_end=end, note=f"engine confidence {conf:.2f}"))
    return result


def _numeric_fix_is_safe(original: str, corrected: str, suspects: list[str]) -> bool:
    """Accept a numeric fix only if it touches the flagged tokens and nothing else.

    The inverse of the correction guards: those forbid editing numbers, this one requires the edit be confined to the
    flagged tokens. It rejects an empty or echoed result, a truncation or balloon, a CJK-to-non-CJK rewrite, a changed
    run that touches a prose character, and a fix that also rewrites an unflagged data token.
    """
    if not corrected or corrected == original:
        return False
    if len(corrected) < MIN_KEEP_RATIO * len(original):
        return False
    if len(corrected) > max(_BALLOON_FLOOR, int(len(original) * _BALLOON_RATIO)):
        return False
    if _is_predominantly_cjk(original) and not _CJK.search(corrected):
        return False
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, original, corrected, autojunk=False).get_opcodes():
        if tag != "equal" and (_NON_DATA_CHAR.search(original[i1:i2]) or _NON_DATA_CHAR.search(corrected[j1:j2])):
            return False
    unflagged = _data_tokens(original)
    for token in _data_tokens(" ".join(suspects)):
        if token in unflagged:
            unflagged.remove(token)
    surviving = _data_tokens(corrected)
    for token in unflagged:
        if token not in surviving:
            return False
        surviving.remove(token)
    return True


def _data_tokens(text: str) -> list[str]:
    return _DIGIT_RUN.findall(text) + _BARE_SYMBOL.findall(text) + _CJK_NUMERAL.findall(text)
