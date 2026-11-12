"""The default quality guard: spot a failed decode and decode the chunk again once, in two halves.

A chunk is one contiguous slice of audio, so the replacement is a whole chunk too: the guard never splices a
re-decode into the original text, and there is no anchor to check. A whole-file hypothesis is only flagged: it is
the production control, and re-decoding parts of it would make it something production never ran.
"""

import unicodedata
from dataclasses import dataclass
from typing import Literal

from ytverbatim.asr.audio import SAMPLE_RATE
from ytverbatim.asr.chunks import FRAME_S
from ytverbatim.asr.protocol import ChunkFlag, ChunkInput, Engine, GuardedChunk, RawChunk
from ytverbatim.asr.quality import (
    COMPRESSION_RATIO_THRESHOLD,
    Unit,
    compression_ratio,
    cross_segment_repetitions,
    decoder_anomalies,
    has_lexical_content,
    is_nonlexical_failure,
    nonlexical_runs,
)
from ytverbatim.contract.models import Word
from ytverbatim.errors import DependencyError

try:
    import numpy as np
    from numpy.typing import NDArray
except ImportError as exc:
    raise DependencyError("install ytverbatim[asr]") from exc

Trigger = Literal[
    "token_cap",
    "repetition_loop",
    "compression_ratio",
    "empty_output",
    "nonlexical_output",
    "word_timing_collapse",
    "replacement_character",
]

# Below this a half would be under a second, too little for an engine to decode reliably, and a speech chunk this
# short with no text is more likely noise than a failed decode.
MIN_REDECODE_S = 2.0
# Whisper applies its compression threshold to one 30 s window; over a longer text, ordinary speech repeats enough
# to cross it, so a longer decode is measured per window of its words.
COMPRESSION_WINDOW_S = 30.0
_SPLIT_FROM = 0.3
_SPLIT_TO = 0.7


def triggers(raw: RawChunk, *, duration: float, capped: bool | None = None) -> tuple[Trigger, ...]:
    """Why a decode of `duration` seconds looks failed; empty when it looks sound.

    `capped` overrides the token-cap check for a merged decode, whose token counts no longer say which half hit it.
    """
    found: list[Trigger] = []
    if _hit_cap(raw) if capped is None else capped:
        found.append("token_cap")
    words = tuple(Unit(word.start, word.end, word.text) for word in raw.words)
    kinds = {a.kind for a in decoder_anomalies([Unit(0.0, duration, raw.text, raw.words)])}
    kinds.update(a.kind for a in cross_segment_repetitions(words))
    if kinds & {"intra_segment_repetition", "cross_segment_repetition"}:
        found.append("repetition_loop")
    if _max_compression_ratio(raw, duration) > COMPRESSION_RATIO_THRESHOLD:
        found.append("compression_ratio")
    if duration >= MIN_REDECODE_S and not has_lexical_content(raw.text):
        found.append("empty_output")
    if any(is_nonlexical_failure(run) for run in nonlexical_runs(words)):
        found.append("nonlexical_output")
    if "word_timing_collapse" in kinds:
        found.append("word_timing_collapse")
    if "replacement_character" in kinds:
        found.append("replacement_character")
    return tuple(found)


class QualityGuard:
    """Re-decodes a failed chunk as two halves cut at the quietest point near its middle.

    At most one re-decode happens per chunk. The halves replace the whole chunk when the result passes; when it
    still fails the better of the two attempts is kept and flagged: `empty_output` when it has no text,
    `decoder_anomaly` for any other failure. With `redecode=False` the first decode is only flagged.
    """

    def __init__(self, *, redecode: bool = True) -> None:
        self._redecode = redecode

    def __call__(self, engine: Engine, chunk: ChunkInput, raw: RawChunk) -> GuardedChunk:
        duration = len(chunk.audio) / SAMPLE_RATE
        first = _Attempt(raw, _hit_cap(raw), triggers(raw, duration=duration))
        if not first.triggers:
            return GuardedChunk(raw)
        cut = _split_point(chunk.audio) if self._redecode else None
        if cut is None:
            return _verdict(first)
        halves = (
            engine.transcribe(chunk.audio[:cut], language=chunk.language, context=chunk.context),
            engine.transcribe(chunk.audio[cut:], language=chunk.language, context=chunk.context),
        )
        merged = _merge(*halves, offset_s=cut / SAMPLE_RATE, duration=duration, fallback_language=raw.language)
        capped = any(_hit_cap(half) for half in halves)
        again = _Attempt(merged, capped, triggers(merged, duration=duration, capped=capped))
        # On a tie the re-decode wins: its damage is bounded by half a chunk.
        verdict = _verdict(min((again, first), key=_badness))
        # Marked whichever attempt is kept, so trigger rates stay measurable with the guard on.
        return GuardedChunk(verdict.raw, ("redecoded", *verdict.flags))


@dataclass(frozen=True, slots=True)
class _Attempt:
    raw: RawChunk
    capped: bool
    triggers: tuple[Trigger, ...]


def _badness(attempt: _Attempt) -> tuple[bool, int]:
    # Text of any kind beats none, whatever else is wrong with it.
    return (not has_lexical_content(attempt.raw.text), len(attempt.triggers))


def _verdict(attempt: _Attempt) -> GuardedChunk:
    flags: list[ChunkFlag] = []
    # Empty output on speech may be music or noise the VAD let through, so it is counted apart from decoder failures.
    if set(attempt.triggers) - {"empty_output"}:
        flags.append("decoder_anomaly")
    if attempt.capped:
        flags.append("token_cap_hit")
    if "empty_output" in attempt.triggers:
        flags.append("empty_output")
    return GuardedChunk(attempt.raw, tuple(flags))


def _hit_cap(raw: RawChunk) -> bool:
    return raw.generated_tokens is not None and raw.max_tokens is not None and raw.generated_tokens >= raw.max_tokens


def _max_compression_ratio(raw: RawChunk, duration: float) -> float:
    if duration <= COMPRESSION_WINDOW_S:
        return compression_ratio(raw.text)
    # Without word times a long text cannot be windowed, and measured whole it would flag ordinary speech.
    windows: list[list[str]] = []
    window_end = -1.0
    for word in raw.words:
        if word.start >= window_end:
            windows.append([])
            window_end = word.start + COMPRESSION_WINDOW_S
        windows[-1].append(word.text)
    return max((compression_ratio("".join(texts)) for texts in windows), default=0.0)


def _split_point(audio: "NDArray[np.float32]") -> int | None:
    """Sample index of the quietest 100 ms frame in the middle 40% of the audio, the one nearest the middle on ties."""
    if len(audio) < MIN_REDECODE_S * SAMPLE_RATE:
        return None
    frame = round(FRAME_S * SAMPLE_RATE)
    first = round(_SPLIT_FROM * len(audio))
    count = (round(_SPLIT_TO * len(audio)) - first) // frame
    frames = audio[first : first + count * frame].reshape(count, frame)
    energy = np.mean(frames * frames, axis=1)
    centers = first + frame * (np.arange(count) + 0.5)
    # lexsort's last key is the primary one.
    best = np.lexsort((np.abs(centers - len(audio) / 2), energy))[0]
    return int(centers[best])


def _merge(
    first: RawChunk, second: RawChunk, *, offset_s: float, duration: float, fallback_language: str | None
) -> RawChunk:
    shifted = tuple(
        Word(text=w.text, start=w.start + offset_s, end=w.end + offset_s, conf=w.conf) for w in second.words
    )
    confidence = None
    if first.confidence is not None and second.confidence is not None:
        confidence = (first.confidence * offset_s + second.confidence * (duration - offset_s)) / duration
    return RawChunk(
        text=_join(first.text, second.text),
        language=first.language or second.language or fallback_language,
        generated_tokens=_add(first.generated_tokens, second.generated_tokens),
        # Budgets add up with the counts, so the worker's cap check on the merged row fires only when both halves hit.
        max_tokens=_add(first.max_tokens, second.max_tokens),
        confidence=confidence,
        words=(*first.words, *shifted),
    )


def _add(left: int | None, right: int | None) -> int | None:
    return None if left is None or right is None else left + right


def _join(left: str, right: str) -> str:
    left, right = left.strip(), right.strip()
    if not left or not right:
        return left or right
    wide = unicodedata.east_asian_width(left[-1]) in "WF" or unicodedata.east_asian_width(right[0]) in "WF"
    return left + right if wide else f"{left} {right}"
