"""LLM pass that fixes clear recognition errors (homophones, garbled terms) in transcript sentences.

Only per-sentence text changes; count and order never do. Every proposed fix passes the deterministic guards in
`ytverbatim.text.guards` or is dropped. A reply that is not the expected JSON is retried once and then raises, so a run
fails loudly rather than shipping an uncorrected transcript that looks corrected.

Each batch's validated LLM reply is checkpointed before the guards run, so a resume never pays twice for a finished
batch, and a guard change re-applies to cached replies for free.
"""

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ytverbatim import atomic
from ytverbatim.contract.models import Edit, Flag, LLMUsage
from ytverbatim.llm.jsonio import complete_json
from ytverbatim.llm.protocol import LLMClient, LLMRequest
from ytverbatim.llm.usage import UsageMeter
from ytverbatim.text.guards import GuardName, check_fix, symbol_changes

_log = logging.getLogger(__name__)

# Part of every checkpoint and stage key: bump it on any prompt, guard or logic change so stored results are redone.
PROMPT_VERSION = "correction-v6"
BATCH_CHARS = 1800
# Mean per-word probability below which a sentence gets the second, low-confidence pass.
SECOND_PASS_MIN_CONFIDENCE = 0.5
STAGE = "correction"
SECOND_PASS_STAGE = "correction_low_confidence"

_CREATOR_NOTE_MAX_CHARS = 600
# A batch where every sentence changes echoes about its input size back; the claude CLI also counts thinking here.
_MAX_OUTPUT_TOKENS = 8192

_SYMBOL_RULE = (
    "An uppercase acronym or ticker may be fixed only when it is a near-homophone mis-recognition of the intended "
    "one (e.g. GAP→GAAP, SMCI→MSCI); never add, drop, or swap in an unrelated one, and never change a number. "
)
_SYSTEM = (
    "You are a meticulous speech-to-text proofreader. You are given numbered ASR "
    "segments and must fix ONLY clear mis-recognitions: homophone errors and "
    "garbled domain terms (e.g. Mandarin 适应率→市盈率, 正线→证券, 前南→前瞻). "
    "Hard rules: keep the SAME language (never translate); preserve meaning and "
    "approximate length; never add, drop, split, or merge segments; do not change "
    "style, fillers, or punctuation except to fix an outright error; keep numbers, "
    "their units, negation, and tickers as spoken. " + _SYMBOL_RULE + "Return ONLY the segments you actually changed."
)
_SECOND_PASS_SYSTEM = (
    "You are a meticulous speech-to-text proofreader. The numbered segments below "
    "were transcribed from LOW-CONFIDENCE audio (the recognizer was unsure), so "
    "mis-recognitions — homophone errors and garbled domain terms — are MORE "
    "likely than usual; scrutinize each one harder. The same hard rules still "
    "apply: keep the SAME language (never translate); preserve meaning and "
    "approximate length; never add, drop, split, or merge segments; do not change "
    "style, fillers, or punctuation except to fix an outright error; keep numbers, "
    "their units, negation, and tickers exactly as spoken. "
    + _SYMBOL_RULE
    + "Return ONLY the segments you actually changed."
)
_FIRST_PASS_ASK = "Fix clear ASR mis-recognitions in these segments. "
_SECOND_PASS_ASK = (
    "These ASR segments came from LOW-CONFIDENCE audio and are likely to "
    "contain mis-recognitions. Re-check each one carefully and fix only "
    "clear errors. "
)

Pass = Literal["first", "low_confidence"]


class _Fix(BaseModel):
    model_config = ConfigDict(strict=True)
    i: int
    text: str


class _Fixes(BaseModel):
    model_config = ConfigDict(strict=True)
    fixed: list[_Fix]


class _Checkpoint(BaseModel):
    """A batch's validated reply and what it cost, so a reused batch still counts toward the document's LLM usage."""

    reply: _Fixes
    usage: tuple[LLMUsage, ...]


@dataclass(frozen=True, slots=True)
class CorrectionContext:
    """What the uploader wrote that can steer a fix: domain terms and the pinned creator note."""

    glossary: tuple[str, ...] = ()
    creator_note: str | None = None


@dataclass(frozen=True, slots=True)
class Rejection:
    """A proposed fix that was not applied; ``reason`` is the guard that refused it, or ``index`` for a bad index."""

    segment: int
    pass_: Pass
    before: str
    proposed: str
    reason: GuardName | Literal["index"]


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    """Per-segment results aligned with the input; flag offsets point into ``texts``."""

    texts: tuple[str, ...]
    edits: tuple[tuple[Edit, ...], ...]
    flags: tuple[tuple[Flag, ...], ...]
    usage: tuple[LLMUsage, ...]
    rejections: tuple[Rejection, ...]
    # Batch requests over both passes, checkpoint reuses included; the LLM calls made exceed it only by retries.
    batches: int


def correct_segments(
    segments: Sequence[str],
    *,
    client: LLMClient,
    language: str,
    context: CorrectionContext,
    checkpoint_dir: Path,
    confidence: Sequence[Sequence[float] | None] | None = None,
) -> CorrectionResult:
    """Correct ``segments`` (sentence texts) and return the guarded result.

    ``confidence`` holds per-word probabilities for each segment, as Whisper reports them. When given, segments whose
    mean is below `SECOND_PASS_MIN_CONFIDENCE` get a second pass with a prompt that asks for harder scrutiny;
    segments without probabilities are skipped there.
    """
    if confidence is not None and len(confidence) != len(segments):
        raise ValueError(f"confidence covers {len(confidence)} segments, but there are {len(segments)}")
    run = _Run(
        texts=list(segments),
        client=client,
        language=language,
        context=context,
        checkpoint_dir=checkpoint_dir,
    )
    for start, end in _batch_ranges(run.texts, BATCH_CHARS):
        run.correct_batch(list(range(start, end)), "first")
    if confidence is not None:
        low = [
            i for i, probs in enumerate(confidence) if probs and sum(probs) / len(probs) < SECOND_PASS_MIN_CONFIDENCE
        ]
        for batch in _index_batches(run.texts, low, BATCH_CHARS):
            run.correct_batch(batch, "low_confidence")
    flags = tuple(_entity_flags(before, after) for before, after in zip(segments, run.texts, strict=True))
    result = CorrectionResult(
        texts=tuple(run.texts),
        edits=tuple(tuple(edits) for edits in run.edits),
        flags=flags,
        usage=run.meter.summary(),
        rejections=tuple(run.rejections),
        batches=run.batches,
    )
    _log.info(
        "correction: %d edits applied, %d proposals rejected",
        sum(len(edits) for edits in result.edits),
        len(result.rejections),
    )
    return result


class _Run:
    def __init__(
        self,
        *,
        texts: list[str],
        client: LLMClient,
        language: str,
        context: CorrectionContext,
        checkpoint_dir: Path,
    ) -> None:
        self.texts = texts
        self._originals = tuple(texts)
        self.edits: list[list[Edit]] = [[] for _ in texts]
        self.rejections: list[Rejection] = []
        self.meter = UsageMeter()
        self.batches = 0
        self._client = client
        self._language = language
        self._context = context
        self._checkpoint_dir = checkpoint_dir

    def correct_batch(self, indices: list[int], pass_: Pass) -> None:
        self.batches += 1
        system = _SYSTEM if pass_ == "first" else _SECOND_PASS_SYSTEM
        user = self._prompt(indices, pass_)
        fixes = self._fixes(system, user, pass_)
        allowed = set(indices)
        for fix in fixes.fixed:
            if fix.i not in allowed:
                self.rejections.append(Rejection(fix.i, pass_, "", fix.text, "index"))
                continue
            allowed.discard(fix.i)
            self._apply(fix.i, fix.text.strip(), pass_)

    def _apply(self, index: int, proposed: str, pass_: Pass) -> None:
        current = self.texts[index]
        if proposed == current:
            return
        # Guarding against the original, not the previous pass, keeps two allowed steps from adding up to a forbidden
        # one (SAP→GAP→GAAP).
        verdict = check_fix(self._originals[index], proposed)
        if verdict.rejected_by is not None:
            self.rejections.append(Rejection(index, pass_, current, proposed, verdict.rejected_by))
            return
        self.texts[index] = proposed
        self.edits[index].append(Edit(stage="correction", before=current, after=proposed, llm=self._client.id))

    def _prompt(self, indices: list[int], pass_: Pass) -> str:
        items = [{"i": i, "text": self.texts[i]} for i in indices]
        glossary = self._context.glossary
        return (
            (_FIRST_PASS_ASK if pass_ == "first" else _SECOND_PASS_ASK)
            + f"The transcript language is {self._language}. "
            + (f"Prefer these domain terms when relevant: {', '.join(glossary)}. " if glossary else "")
            + _creator_note_hint(self._context.creator_note)
            + 'Return JSON {"fixed": [{"i": <index>, "text": "<corrected>"}]} '
            "containing ONLY the segments you changed.\n\nSEGMENTS:\n" + json.dumps(items, ensure_ascii=False)
        )

    def _fixes(self, system: str, user: str, pass_: Pass) -> _Fixes:
        key = json.dumps([PROMPT_VERSION, self._client.id, system, user], ensure_ascii=False)
        path = self._checkpoint_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
        if path.exists():
            try:
                checkpoint = _Checkpoint.model_validate_json(path.read_bytes())
            except ValidationError:
                # Only this module writes these files, atomically; an unreadable one is stale, and redoing it is safe.
                _log.warning("correction: ignoring an unreadable checkpoint %s", path.name)
            else:
                self.meter.merge(checkpoint.usage)
                return checkpoint.reply
        stage = STAGE if pass_ == "first" else SECOND_PASS_STAGE
        request = LLMRequest(system=system, user=user, max_output_tokens=_MAX_OUTPUT_TOKENS, temperature=0.0)
        batch = UsageMeter()
        try:
            fixes = complete_json(batch.metered(stage, self._client), request, _Fixes)
        finally:
            self.meter.merge(batch.summary())
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        atomic.write_json(path, _Checkpoint(reply=fixes, usage=batch.summary()).model_dump(mode="json"))
        return fixes


def _entity_flags(original: str, corrected: str) -> tuple[Flag, ...]:
    return tuple(
        Flag(
            kind="entity_corrected",
            char_start=change.char_start,
            char_end=change.char_end,
            note=f"{change.before} → {change.after}",
        )
        for change in symbol_changes(original, corrected)
    )


def _creator_note_hint(creator_note: str | None) -> str:
    """Frame the creator's pinned note as authoritative context; capped so it doesn't bloat every batch."""
    if not creator_note or not creator_note.strip():
        return ""
    note = creator_note.strip()
    if len(note) > _CREATOR_NOTE_MAX_CHARS:
        note = note[:_CREATOR_NOTE_MAX_CHARS] + "…"
    # Numbers stay as spoken even when the note corrects one: the note is kept for consumers, never silently applied.
    return (
        "The video creator PINNED this note with corrections/clarifications — treat "
        "it as authoritative when choosing a fix, but still keep numbers exactly as spoken "
        f'and change a ticker only under the near-homophone rule: "{note}". '
    )


def _batch_ranges(texts: Sequence[str], batch_chars: int) -> list[tuple[int, int]]:
    """Consecutive [start, end) ranges whose texts reach about ``batch_chars`` characters."""
    ranges: list[tuple[int, int]] = []
    start = 0
    running = 0
    for i, text in enumerate(texts):
        running += len(text)
        if running >= batch_chars:
            ranges.append((start, i + 1))
            start = i + 1
            running = 0
    if start < len(texts):
        ranges.append((start, len(texts)))
    return ranges


def _index_batches(texts: Sequence[str], indices: list[int], batch_chars: int) -> list[list[int]]:
    """Group possibly non-contiguous indices into batches of about ``batch_chars`` characters."""
    batches: list[list[int]] = []
    current: list[int] = []
    running = 0
    for i in indices:
        current.append(i)
        running += len(texts[i])
        if running >= batch_chars:
            batches.append(current)
            current = []
            running = 0
    if current:
        batches.append(current)
    return batches
