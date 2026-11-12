"""Screening: find where hypotheses disagree, measure each one automatically, and let the blind judge sample.

The output only narrows the field to one control and one challenger; the blind human A/B decides between them.
"""

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from evals.consensus import Dispute, TimedUnit, find_disputes, is_acronym, window_units
from evals.judge import JudgeItem, Verdict, make_item
from evals.manifest import Manifest, Video
from evals.metrics import (
    ErrorCounts,
    cer,
    mer,
    nonspeech_intervals,
    repetition_runs,
    units_in_nonspeech,
    wer,
    wilson_interval,
)
from evals.normalize import Language, View, normalize
from evals.store import (
    CAPTION_HYPOTHESIS,
    REFERENCE,
    Layout,
    Plan,
    file_complete,
    hypothesis_units,
    load_plan,
    read_history,
    row_text,
    rows_by_file,
)
from ytverbatim.asr import registry
from ytverbatim.asr.models import ChunkHypothesis
from ytverbatim.errors import ConfigError

DISPUTE_VIEW: View = "raw"
CONTROLS = ("w-prod", "w-vad")
CHALLENGERS = ("q-base", "q-ctx", "f-aed", "m-4bit")
LANGUAGES: tuple[Language, ...] = ("zh", "en")
SECONDS_PER_HOUR = 3600.0

# The guard's re-decode flag; matched by name so screening also reads runs recorded before the flag existed.
REDECODED = "redecoded"

_LATIN_RUN = re.compile(r"[A-Za-z]+")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SpanReading(_Frozen):
    hypothesis: str
    text: str
    start: float | None
    end: float | None
    before: float | None
    after: float | None


class Span(_Frozen):
    """A disputed stretch. Its readings name engines, so a span never reaches the judge or a screen as it is."""

    id: str
    file: str
    language: Language
    window: int
    readings: tuple[SpanReading, ...]
    distinct: tuple[str, ...]
    left: str
    right: str
    start: float
    end: float
    touches_quantity: bool
    touches_latin: bool
    format_only: bool

    @property
    def numeric(self) -> bool:
        return self.touches_quantity or self.touches_latin

    def reading(self, hypothesis: str) -> SpanReading:
        return next(r for r in self.readings if r.hypothesis == hypothesis)


def format_only(readings: Sequence[str], language: Language) -> bool:
    """Whether the ``readings`` differ only in written form (``150期`` against ``一百五十期``), not in what was said.

    A listener cannot separate such readings, so they go to neither the judge nor the A/B. Latin case stays a real
    difference where it marks a ticker or acronym, ``nvda`` for ``NVDA``; in Chinese text every Latin token is one.
    English sentence-start capitals (``They`` against ``they``) are written form.
    """
    if len({normalize(text, language, "itn") for text in readings}) > 1:
        return False
    latin = {tuple(_LATIN_RUN.findall(text)) for text in readings}
    folded = {tuple(word.casefold() for word in words) for words in latin}
    case_conflict = len(folded) == 1 and len(latin) > 1
    if case_conflict and language == "en":
        case_conflict = any(is_acronym(word) for words in latin for word in words)
    return not case_conflict


def available(layout: Layout, video: Video, plan: Plan, hypotheses: Sequence[str]) -> dict[str, list[ChunkHypothesis]]:
    """Rows of every listed hypothesis whose output for this file is complete."""
    out: dict[str, list[ChunkHypothesis]] = {}
    for hypothesis in hypotheses:
        rows = rows_by_file(layout, hypothesis).get(video.id, [])
        if hypothesis == CAPTION_HYPOTHESIS:
            if rows:
                out[hypothesis] = rows
        elif rows and file_complete(registry.get(hypothesis), plan, rows):
            out[hypothesis] = rows
    return out


def family(hypothesis: str) -> str:
    return "captions" if hypothesis == CAPTION_HYPOTHESIS else registry.get(hypothesis).spec.family


def file_spans(
    video: Video, plan: Plan, rows: Mapping[str, Sequence[ChunkHypothesis]], pivot: str | None = None
) -> list[Span]:
    """Disputes per alignment window: medoid pivot over engine families, or ``pivot`` for a direct pairwise run."""
    language = video.view_language
    units = {name: hypothesis_units(r, language, DISPUTE_VIEW, plan.speech) for name, r in rows.items()}
    families = {name: family(name) for name in rows}
    spans: list[Span] = []
    for window, (start, end) in enumerate(plan.windows):
        in_window: dict[str, list[TimedUnit]] = {name: window_units(u, start, end) for name, u in units.items()}
        spans += window_spans(video, language, window, in_window, families=families, pivot=pivot)
    return spans


def window_spans(
    video: Video,
    language: Language,
    window: int,
    units: Mapping[str, Sequence[TimedUnit]],
    *,
    families: Mapping[str, str] | None = None,
    pivot: str | None = None,
) -> list[Span]:
    """Disputes among the hypotheses' units of one alignment window; ``window`` only numbers the span ids."""
    return [_span(video, language, window, d) for d in find_disputes(units, language, families=families, pivot=pivot)]


def _span(video: Video, language: Language, window: int, dispute: Dispute) -> Span:
    return Span(
        id=f"{video.id}#{window}#{dispute.index}",
        file=video.id,
        language=language,
        window=window,
        readings=tuple(
            SpanReading(
                hypothesis=r.hypothesis, text=r.text, start=r.start_s, end=r.end_s, before=r.before_s, after=r.after_s
            )
            for r in dispute.readings
        ),
        distinct=dispute.distinct,
        left=dispute.left_context,
        right=dispute.right_context,
        start=dispute.start_s,
        end=dispute.end_s,
        touches_quantity=dispute.touches_quantity,
        touches_latin=dispute.touches_latin,
        format_only=format_only(dispute.distinct, language),
    )


def collect_spans(
    layout: Layout, manifest: Manifest, hypotheses: Sequence[str], pivot: str | None = None
) -> list[Span]:
    spans: list[Span] = []
    for video in manifest.videos:
        plan = load_plan(layout, video.id)
        if plan is None:
            continue
        rows = available(layout, video, plan, hypotheses)
        if len(rows) >= 2:  # noqa: PLR2004 - a dispute needs two readings
            spans.extend(file_spans(video, plan, rows, pivot))
    return spans


class Quality(_Frozen):
    """Counts behind the quality gates, for one hypothesis over the files of one language."""

    files: int = 0
    audio_s: float = 0.0
    units: int = 0
    chunks: int = 0
    repetition_runs: int = 0
    token_cap_hits: int = 0
    redecoded: int = 0
    decoder_anomalies: int = 0
    nonspeech_units: int = 0

    def plus(self, other: "Quality") -> "Quality":
        return Quality(**{name: getattr(self, name) + getattr(other, name) for name in Quality.model_fields})

    @property
    def hours(self) -> float:
        return self.audio_s / SECONDS_PER_HOUR

    @property
    def repetition_per_hour(self) -> float:
        return self.repetition_runs / self.hours if self.hours else 0.0

    @property
    def token_cap_per_hour(self) -> float:
        return self.token_cap_hits / self.hours if self.hours else 0.0

    @property
    def redecode_share(self) -> float:
        """Share of chunks the quality guard decoded a second time."""
        return self.redecoded / self.chunks if self.chunks else 0.0

    @property
    def nonspeech_share(self) -> float:
        return self.nonspeech_units / self.units if self.units else 0.0


class Rate(_Frozen):
    errors: int
    reference_units: int

    @property
    def rate(self) -> float:
        return self.errors / self.reference_units

    @classmethod
    def of(cls, counts: ErrorCounts) -> "Rate":
        return cls(errors=counts.errors, reference_units=counts.reference_units)

    def plus(self, other: "Rate") -> "Rate":
        return Rate(errors=self.errors + other.errors, reference_units=self.reference_units + other.reference_units)


class Share(_Frozen):
    wins: int
    decided: int
    needs_audio: int
    low: float
    high: float

    @property
    def rate(self) -> float:
        return self.wins / self.decided


class Speed(_Frozen):
    rtf: float | None
    load_s: float | None
    # Footprint comes from newer runs only, so a mixed history reports it for just those; the others are older.
    peak_footprint_bytes: int | None
    peak_memory_bytes: int | None
    max_rss_bytes: int | None


NO_SPEED = Speed(rtf=None, load_s=None, peak_footprint_bytes=None, peak_memory_bytes=None, max_rss_bytes=None)


class HypothesisScreen(_Frozen):
    hypothesis: str
    quality: dict[Language, Quality]
    # Keyed by language view; only files with a manual reference contribute.
    reference_cer: dict[Language, Rate]
    reference_cer_raw: dict[Language, Rate]
    reference_mer: dict[Language, Rate]
    caption_wer: Rate | None
    speed: Speed
    judge: Share | None
    disputes: int


class Recommendation(_Frozen):
    control: str | None
    control_reason: str
    challenger: str | None
    challenger_reason: str
    runner_up: str | None
    challenger_ci_overlap: bool


class ScreenResult(_Frozen):
    generated: str
    hypotheses: tuple[HypothesisScreen, ...]
    spans: int
    format_only_spans: int
    judge_model: str | None
    judge_prompt_version: str | None
    judge_seed: int
    judge_sample: int
    judge_answered: int
    recommendation: Recommendation

    def get(self, hypothesis: str) -> HypothesisScreen | None:
        return next((h for h in self.hypotheses if h.hypothesis == hypothesis), None)


def file_quality(video: Video, plan: Plan, rows: Sequence[ChunkHypothesis]) -> Quality:
    units = hypothesis_units(rows, video.view_language, DISPUTE_VIEW, plan.speech)
    silence = nonspeech_intervals(plan.speech, plan.duration_s)
    return Quality(
        files=1,
        audio_s=plan.duration_s,
        units=len(units),
        chunks=len(rows),
        repetition_runs=len(repetition_runs([u.text for u in units])),
        token_cap_hits=sum("token_cap_hit" in row.flags for row in rows),
        redecoded=sum(REDECODED in row.flags for row in rows),
        decoder_anomalies=sum("decoder_anomaly" in row.flags for row in rows),
        nonspeech_units=len(units_in_nonspeech(units, silence)),
    )


def speed(layout: Layout, hypothesis: str) -> Speed:
    history = read_history(layout, hypothesis)
    if not history:
        return NO_SPEED
    seconds = sum(r.file_seconds for r in history)
    footprints = [r.metrics.peak_footprint_bytes for r in history if r.metrics.peak_footprint_bytes is not None]
    peaks = [r.metrics.peak_memory_bytes for r in history if r.metrics.peak_memory_bytes is not None]
    loads = [r.metrics.load_s for r in history if r.metrics.chunks_decoded]
    return Speed(
        # Decode time over the duration of the files decoded, so whole-file and chunked engines compare directly.
        rtf=sum(r.metrics.decode_s for r in history) / seconds if seconds else None,
        load_s=max(loads) if loads else None,
        peak_footprint_bytes=max(footprints) if footprints else None,
        peak_memory_bytes=max(peaks) if peaks else None,
        max_rss_bytes=max(max(r.metrics.max_rss_bytes, r.children_max_rss_bytes) for r in history),
    )


@dataclass
class _Accumulator:
    quality: dict[Language, Quality]
    reference_cer: dict[Language, Rate] = field(default_factory=dict)
    reference_cer_raw: dict[Language, Rate] = field(default_factory=dict)
    reference_mer: dict[Language, Rate] = field(default_factory=dict)
    caption_wer: Rate | None = None


def _add(total: Rate | None, counts: ErrorCounts) -> Rate:
    rate = Rate.of(counts)
    return rate if total is None else total.plus(rate)


def _add_to(rates: dict[Language, Rate], language: Language, counts: ErrorCounts) -> None:
    rates[language] = _add(rates.get(language), counts)


def automatic_metrics(layout: Layout, manifest: Manifest, hypotheses: Sequence[str]) -> dict[str, _Accumulator]:
    acc = {h: _Accumulator(quality={lang: Quality() for lang in LANGUAGES}) for h in hypotheses}
    references = rows_by_file(layout, REFERENCE)
    captions = rows_by_file(layout, CAPTION_HYPOTHESIS)
    for video in manifest.videos:
        plan = load_plan(layout, video.id)
        if plan is None:
            continue
        language = video.view_language
        for hypothesis, rows in available(layout, video, plan, hypotheses).items():
            entry = acc[hypothesis]
            entry.quality[language] = entry.quality[language].plus(file_quality(video, plan, rows))
            text = row_text(rows)
            reference = references.get(video.id)
            if reference:
                ref = row_text(reference)
                _add_to(entry.reference_cer, language, cer(ref, text, language, "itn"))
                _add_to(entry.reference_cer_raw, language, cer(ref, text, language, "raw"))
                _add_to(entry.reference_mer, language, mer(ref, text, language, "itn"))
            caption = captions.get(video.id)
            if language == "en" and caption and hypothesis != CAPTION_HYPOTHESIS:
                entry.caption_wer = _add(entry.caption_wer, wer(row_text(caption), text, language, "itn"))
    return acc


def judge_items(spans: Sequence[Span], sample: int, seed: int) -> list[JudgeItem]:
    """Draw a seeded uniform sample of the spans a listener could tell apart, with readings shuffled per span."""
    candidates = [s for s in spans if not s.format_only]
    chosen = random.Random(seed).sample(candidates, min(sample, len(candidates)))
    return [make_item(s.id, s.left, s.right, s.distinct, seed) for s in sorted(chosen, key=lambda s: s.id)]


def judge_shares(spans: Sequence[Span], verdicts: Mapping[str, Verdict], hypotheses: Sequence[str]) -> dict[str, Share]:
    """Per hypothesis: how often its reading is the one the judge chose, among spans the judge could decide."""
    counts = {h: [0, 0, 0] for h in hypotheses}
    for span in spans:
        verdict = verdicts.get(span.id)
        if verdict is None:
            continue
        for reading in span.readings:
            row = counts.get(reading.hypothesis)
            if row is None:
                continue
            if verdict.needs_audio:
                row[2] += 1
                continue
            row[1] += 1
            row[0] += reading.text == verdict.choice
    shares = {}
    for hypothesis, (wins, decided, needs_audio) in counts.items():
        if decided:
            low, high = wilson_interval(wins, decided)
            shares[hypothesis] = Share(wins=wins, decided=decided, needs_audio=needs_audio, low=low, high=high)
    return shares


def recommend(rows: Sequence[HypothesisScreen]) -> Recommendation:
    """Control: the better Whisper by judge share, then by fewer quality flags. Challenger: the best judge share."""
    by_id = {row.hypothesis: row for row in rows}
    controls = [by_id[h] for h in CONTROLS if h in by_id]
    challengers = sorted(
        (by_id[h] for h in CHALLENGERS if h in by_id and by_id[h].judge is not None),
        key=lambda row: _share(row),
        reverse=True,
    )
    control: HypothesisScreen | None = None
    control_reason = "no Whisper run on disk"
    if controls:
        control = max(controls, key=lambda row: (_share(row), -_flags(row)))
        others = [row for row in controls if row is not control]
        control_reason = _control_reason(control, others[0] if others else None)
    if not challengers:
        return Recommendation(
            control=control.hypothesis if control else None,
            control_reason=control_reason,
            challenger=None,
            challenger_reason="no challenger with judge verdicts",
            runner_up=None,
            challenger_ci_overlap=False,
        )
    best, runner = challengers[0], challengers[1] if len(challengers) > 1 else None
    overlap = runner is not None and _overlap(best, runner)
    reason = f"highest judge win share ({_share(best):.3f})"
    if runner is not None:
        reason += (
            f"; its 95% CI overlaps {runner.hypothesis}'s, so the pick between them is not settled by the judge"
            if overlap
            else f"; CI clear of {runner.hypothesis}"
        )
    return Recommendation(
        control=control.hypothesis if control else None,
        control_reason=control_reason,
        challenger=best.hypothesis,
        challenger_reason=reason,
        runner_up=runner.hypothesis if runner else None,
        challenger_ci_overlap=overlap,
    )


def _share(row: HypothesisScreen) -> float:
    return row.judge.rate if row.judge else -1.0


def _flags(row: HypothesisScreen) -> float:
    total = row.quality["zh"].plus(row.quality["en"])
    return total.repetition_per_hour + total.token_cap_per_hour + 100 * total.nonspeech_share


def _overlap(a: HypothesisScreen, b: HypothesisScreen) -> bool:
    if a.judge is None or b.judge is None:
        return False
    return a.judge.low <= b.judge.high and b.judge.low <= a.judge.high


def _control_reason(control: HypothesisScreen, other: HypothesisScreen | None) -> str:
    if other is None:
        return "the only Whisper run on disk"
    if control.judge and other.judge:
        tie = " (CIs overlap; quality flags break the tie)" if _overlap(control, other) else ""
        return f"judge win share {control.judge.rate:.3f} vs {other.judge.rate:.3f}{tie}"
    return "fewer quality flags (no judge verdicts for both)"


def build_screen(
    layout: Layout,
    manifest: Manifest,
    hypotheses: Sequence[str],
    spans: Sequence[Span],
    verdicts: Mapping[str, Verdict],
    *,
    judge_model: str | None,
    judge_prompt_version: str | None,
    judge_seed: int,
    judge_sample: int,
    generated: str,
) -> ScreenResult:
    if not hypotheses:
        raise ConfigError("no hypothesis output on disk; run `asr` or `captions` first")
    acc = automatic_metrics(layout, manifest, hypotheses)
    shares = judge_shares(spans, verdicts, hypotheses)
    rows = tuple(
        HypothesisScreen(
            hypothesis=h,
            quality=acc[h].quality,
            reference_cer=acc[h].reference_cer,
            reference_cer_raw=acc[h].reference_cer_raw,
            reference_mer=acc[h].reference_mer,
            caption_wer=acc[h].caption_wer,
            speed=speed(layout, h) if h != CAPTION_HYPOTHESIS else NO_SPEED,
            judge=shares.get(h),
            disputes=sum(1 for s in spans if any(r.hypothesis == h for r in s.readings)),
        )
        for h in hypotheses
    )
    return ScreenResult(
        generated=generated,
        hypotheses=rows,
        spans=len(spans),
        format_only_spans=sum(s.format_only for s in spans),
        judge_model=judge_model,
        judge_prompt_version=judge_prompt_version,
        judge_seed=judge_seed,
        judge_sample=judge_sample,
        judge_answered=sum(1 for s in spans if s.id in verdicts),
        recommendation=recommend(rows),
    )


def screen_table(result: ScreenResult) -> str:
    header = (
        "| hypothesis | judge win share [95% CI] | n | needs audio | ref CER | rep/h | cap/h | nonspeech "
        "| re-decoded | anomalies | RTF | peak GB (footprint / MLX / RSS) |"
    )
    lines = [header, "|" + "---|" * 12]
    for row in result.hypotheses:
        total = row.quality["zh"].plus(row.quality["en"])
        judge = f"{row.judge.rate:.3f} [{row.judge.low:.3f}, {row.judge.high:.3f}]" if row.judge else "—"
        lines.append(
            f"| {row.hypothesis} | {judge} | {row.judge.decided if row.judge else 0} "
            f"| {row.judge.needs_audio if row.judge else 0} | {_pct(row.reference_cer.get('zh'))} "
            f"| {total.repetition_per_hour:.2f} | {total.token_cap_per_hour:.2f} | {total.nonspeech_share:.2%} "
            f"| {total.redecode_share:.2%} | {total.decoder_anomalies} | {_num(row.speed.rtf)} "
            f"| {_memory(row.speed)} |"
        )
    return "\n".join(lines)


def recommendation_text(result: ScreenResult) -> str:
    rec = result.recommendation
    return (
        f"control: {rec.control or '—'} ({rec.control_reason})\n"
        f"challenger: {rec.challenger or '—'} ({rec.challenger_reason})"
    )


def _pct(rate: Rate | None) -> str:
    return f"{rate.rate:.2%}" if rate else "—"


def _num(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def _memory(speed: Speed) -> str:
    return " / ".join(_gb(v) for v in (speed.peak_footprint_bytes, speed.peak_memory_bytes, speed.max_rss_bytes))


def _gb(value: int | None) -> str:
    return f"{value / 1e9:.2f}" if value is not None else "—"
