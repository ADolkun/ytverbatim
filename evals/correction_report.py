"""Screening and the report of the correction-LLM eval.

``gather`` builds every dispute the judge and the owner see from the candidates' stored runs; ``assess`` reads the
judge's cached verdicts back onto them and measures each candidate. ``screen`` and ``report`` both go through those two
steps, so they always agree on span ids and on which candidates were compared. Only aggregates leave this module:
files appear as generic labels (``zh-01``) and candidates by id; no video id, title, reading or transcript text.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evals.correction import (
    CANDIDATES,
    PROPOSED,
    Candidate,
    RejectionSpan,
    audible_edits,
    complete_candidates,
    edit_spans,
    failure_note,
    first_try_validity,
    load_base,
    load_calls,
    load_failure,
    load_server_record,
    rejection_spans,
    require_run,
    run_dir,
)
from evals.correction_metrics import (
    Agreement,
    CandidateScreen,
    CorrectionDecision,
    CorrectionReport,
    Estimate,
    Pick,
    human_estimates,
    judge_agreement,
    judge_label,
    judge_stats,
    number_violations,
    pick,
    proportion,
    session_answers,
)
from evals.judge import JUDGE_MODEL, PROMPT_VERSION, JudgeItem, Verdict, load_verdicts, make_item
from evals.listen import read_decisions
from evals.manifest import Manifest
from evals.metrics import ErrorCounts, cer, mer
from evals.normalize import Language, View
from evals.report import FileRow, file_rows, files_table
from evals.screen import LANGUAGES, SECONDS_PER_HOUR, Span, judge_items
from evals.session import KEY_FILE, load_key
from evals.store import REFERENCE, Layout, row_text, rows_by_file, videos
from ytverbatim.errors import ConfigError

SCREEN_FILE = "screen.json"
ITEMS_FILE = "items.jsonl"
REJECTIONS_FILE = "rejections.jsonl"
REPORT_JSON = "report.json"
REPORT_MD = "report.md"
# A text judge cannot verify numbers, so its read of a numeric guard's rejections is informational only.
NUMERIC_GUARDS = frozenset({"digits", "numeric_spans", "quantities", "cjk_numerals"})


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class JudgeRun(_Frozen):
    model: str
    prompt_version: str
    seed: int
    # The uniform sample of edit disputes, and how many of it have a verdict; `session` needs all of them.
    sample: int
    answered: int
    rejection_sample: int
    rejection_answered: int
    spans: int
    format_only_spans: int

    @property
    def complete(self) -> bool:
        return self.answered == self.sample


class GuardRate(_Frozen):
    """How often the judge preferred the proposal a guard refused, among its sampled and judged rejections."""

    guard: str
    informational: bool
    sampled: int
    judged: int
    false_rejects: int
    rate: Estimate | None


class CorrectionScreen(_Frozen):
    """``screen.json``: what was compared and the judge's pick; `session` builds on it."""

    generated: str
    # The candidates whose runs were compared. The judge's options, and so every span id, depend on exactly this set.
    screened: tuple[str, ...]
    judge: JudgeRun
    candidates: tuple[CandidateScreen, ...]
    pick: Pick
    guards: tuple[GuardRate, ...]


class ErrorChange(_Frozen):
    before: int
    after: int
    units: int

    @property
    def removed_share(self) -> float | None:
        return (self.before - self.after) / self.before if self.before else None

    def plus(self, other: "ErrorChange") -> "ErrorChange":
        return ErrorChange(
            before=self.before + other.before, after=self.after + other.after, units=self.units + other.units
        )


class ReferenceScore(_Frozen):
    """A candidate's change in error against the manual captions, over the files that have them."""

    candidate: str
    cer: ErrorChange
    cer_raw: ErrorChange
    mer: ErrorChange
    judge_net_per_hour: Estimate | None


class CostRow(_Frozen):
    candidate: str
    calls: int
    batches: int
    input_tokens: int
    output_tokens: int
    wall_s: float
    peak_footprint_bytes: int | None


class Report(_Frozen):
    generated: str
    files: tuple[FileRow, ...]
    judge: JudgeRun
    correction: CorrectionReport
    agreement: Agreement | None
    guards: tuple[GuardRate, ...]
    reference: tuple[ReferenceScore, ...]
    cost: tuple[CostRow, ...]


@dataclass(frozen=True)
class Gathered:
    screened: tuple[Candidate, ...]
    # Candidates with a run directory but no finished run; listed as ineligible and left out of the disputes.
    unfinished: tuple[Candidate, ...]
    seed: int
    spans: list[Span]
    items: list[JudgeItem]
    rejections: list[RejectionSpan]
    rejection_items: list[JudgeItem]

    @property
    def to_judge(self) -> list[JudgeItem]:
        return [*self.items, *self.rejection_items]


@dataclass(frozen=True)
class Assessment:
    verdicts: dict[str, Verdict]
    judge: JudgeRun
    candidates: tuple[CandidateScreen, ...]
    guards: tuple[GuardRate, ...]


def judge_dir(store: Layout) -> Path:
    return store.correction / "judge"


def load_screen(store: Layout) -> CorrectionScreen:
    path = store.correction / SCREEN_FILE
    if not path.exists():
        raise ConfigError("no screening result; run `screen` first")
    return CorrectionScreen.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_candidates(
    store: Layout, manifest: Manifest, requested: Sequence[str]
) -> tuple[list[Candidate], list[Candidate]]:
    """Return the candidates to compare and the unfinished ones to list: the requested ids, or every finished run."""
    unknown = [c for c in requested if c not in CANDIDATES]
    if unknown:
        raise ConfigError(f"unknown candidate {', '.join(unknown)}; choose from {', '.join(sorted(CANDIDATES))}")
    if requested:
        return [CANDIDATES[c] for c in requested], []
    screened = complete_candidates(store, manifest)
    if not screened:
        raise ConfigError("no candidate has finished `correct` for every file; run it first")
    unfinished = [c for c in CANDIDATES.values() if c not in screened and run_dir(store, c).exists()]
    return screened, unfinished


def gather(
    store: Layout,
    manifest: Manifest,
    screened: Sequence[Candidate],
    unfinished: Sequence[Candidate],
    *,
    seed: int,
    sample: int,
) -> Gathered:
    spans = edit_spans(store, manifest, screened)
    rejections = rejection_spans(store, manifest, screened, seed=seed)
    return Gathered(
        screened=tuple(screened),
        unfinished=tuple(unfinished),
        seed=seed,
        spans=spans,
        items=judge_items(spans, sample, seed),
        rejections=rejections,
        rejection_items=[make_item(r.span.id, r.span.left, r.span.right, r.span.distinct, seed) for r in rejections],
    )


def judged(store: Layout, items: Sequence[JudgeItem]) -> dict[str, Verdict]:
    """Return the cached verdicts keyed by span id; the cache itself is keyed by what the judge saw."""
    cached = load_verdicts(judge_dir(store))
    return {i.span_id: cached[i.key(JUDGE_MODEL)] for i in items if i.key(JUDGE_MODEL) in cached}


def _hours(manifest: Manifest) -> dict[Language, float]:
    return {
        lang: sum(v.audio_duration_s for v in manifest.videos if v.view_language == lang) / SECONDS_PER_HOUR
        for lang in LANGUAGES
    }


def assess(store: Layout, manifest: Manifest, gathered: Gathered) -> Assessment:
    verdicts = judged(store, gathered.to_judge)
    hours = sum(_hours(manifest).values())
    screens = [_screen_candidate(store, manifest, c, gathered.spans, verdicts, hours) for c in gathered.screened]
    screens += [_unfinished(store, c, hours) for c in gathered.unfinished]
    judge = JudgeRun(
        model=JUDGE_MODEL,
        prompt_version=PROMPT_VERSION,
        seed=gathered.seed,
        sample=len(gathered.items),
        answered=sum(i.span_id in verdicts for i in gathered.items),
        rejection_sample=len(gathered.rejection_items),
        rejection_answered=sum(i.span_id in verdicts for i in gathered.rejection_items),
        spans=len(gathered.spans),
        format_only_spans=sum(s.format_only for s in gathered.spans),
    )
    return Assessment(verdicts, judge, tuple(screens), guard_rates(gathered.rejections, verdicts))


def _screen_candidate(
    store: Layout,
    manifest: Manifest,
    candidate: Candidate,
    spans: Sequence[Span],
    verdicts: Mapping[str, Verdict],
    hours: float,
) -> CandidateScreen:
    pairs = [(load_base(store, v.id), require_run(store, candidate, v.id)) for v in videos(manifest)]
    return CandidateScreen(
        id=candidate.id,
        local=candidate.local,
        complete=True,
        llm_output_failed=load_failure(store, candidate) is not None,
        number_violations=sum(number_violations(base, run) for base, run in pairs),
        first_try_validity=first_try_validity([run for _, run in pairs]),
        judge=judge_stats(spans, verdicts, candidate.id, hours=hours),
    )


def _unfinished(store: Layout, candidate: Candidate, hours: float) -> CandidateScreen:
    return CandidateScreen(
        id=candidate.id,
        local=candidate.local,
        complete=False,
        llm_output_failed=load_failure(store, candidate) is not None,
        number_violations=0,
        first_try_validity=None,
        judge=judge_stats([], {}, candidate.id, hours=hours),
    )


def guard_rates(rejections: Sequence[RejectionSpan], verdicts: Mapping[str, Verdict]) -> tuple[GuardRate, ...]:
    """Per guard, the share of judged rejected proposals whose every audible dispute went to the proposed reading.

    A proposal counts as judged when each of its disputes has a verdict that is not ``needs_audio``.
    """
    proposals: dict[tuple[str, int], list[RejectionSpan]] = {}
    for rejection in rejections:
        proposals.setdefault((rejection.candidate, rejection.rejection), []).append(rejection)
    sampled: dict[str, int] = {}
    outcomes: dict[str, list[bool]] = {}
    for group in proposals.values():
        guard = group[0].guard
        sampled[guard] = sampled.get(guard, 0) + 1
        found = [(r.span, verdicts.get(r.span.id)) for r in group]
        if all(v is not None and not v.needs_audio for _, v in found):
            outcomes.setdefault(guard, []).append(all(v.choice == s.reading(PROPOSED).text for s, v in found if v))
    return tuple(
        GuardRate(
            guard=guard,
            informational=guard in NUMERIC_GUARDS,
            sampled=sampled[guard],
            judged=len(outcomes.get(guard, [])),
            false_rejects=sum(outcomes.get(guard, [])),
            rate=proportion(sum(outcomes[guard]), len(outcomes[guard])) if guard in outcomes else None,
        )
        for guard in sorted(sampled)
    )


def build_screen(gathered: Gathered, assessment: Assessment, generated: str) -> CorrectionScreen:
    return CorrectionScreen(
        generated=generated,
        screened=tuple(c.id for c in gathered.screened),
        judge=assessment.judge,
        candidates=assessment.candidates,
        pick=pick(assessment.candidates),
        guards=assessment.guards,
    )


def _compare(
    score: Callable[[str, str, Language, View], ErrorCounts],
    view: View,
    reference: str,
    texts: tuple[str, str],
    language: Language,
) -> ErrorChange:
    before, after = (score(reference, text, language, view) for text in texts)
    return ErrorChange(before=before.errors, after=after.errors, units=before.reference_units)


def reference_scores(
    store: Layout, manifest: Manifest, gathered: Gathered, verdicts: Mapping[str, Verdict]
) -> tuple[ReferenceScore, ...]:
    """Each candidate's change in CER and MER against the manual captions; a validation of the judge, not a gate."""
    references = rows_by_file(store, REFERENCE)
    files = [v for v in videos(manifest) if v.id in references]
    if not files:
        return ()
    ids = {v.id for v in files}
    hours = sum(v.audio_duration_s for v in files) / SECONDS_PER_HOUR
    scores: list[ReferenceScore] = []
    for candidate in gathered.screened:
        changes = []
        for v in files:
            texts = (
                "\n".join(s.text for s in load_base(store, v.id).stretches),
                "\n".join(require_run(store, candidate, v.id).texts),
            )
            reference = row_text(references[v.id])
            changes.append(
                [
                    _compare(score, view, reference, texts, v.view_language)
                    for score, view in ((cer, "itn"), (cer, "raw"), (mer, "itn"))
                ]
            )
        total = [reduce(ErrorChange.plus, column) for column in zip(*changes, strict=True)]
        judge = judge_stats([s for s in gathered.spans if s.file in ids], verdicts, candidate.id, hours=hours)
        scores.append(
            ReferenceScore(
                candidate=candidate.id,
                cer=total[0],
                cer_raw=total[1],
                mer=total[2],
                judge_net_per_hour=judge.net_per_hour,
            )
        )
    return tuple(scores)


def cost_rows(store: Layout, manifest: Manifest, gathered: Gathered) -> tuple[CostRow, ...]:
    rows: list[CostRow] = []
    for candidate in gathered.screened:
        runs = [require_run(store, candidate, v.id) for v in videos(manifest)]
        usage = [u for run in runs for u in run.usage]
        server = load_server_record(store, candidate)
        rows.append(
            CostRow(
                candidate=candidate.id,
                calls=sum(u.calls for u in usage),
                batches=sum(run.batches for run in runs),
                input_tokens=sum(u.input_tokens or 0 for u in usage),
                output_tokens=sum(u.output_tokens or 0 for u in usage),
                wall_s=sum(c.elapsed_s for c in load_calls(store, candidate)),
                peak_footprint_bytes=None if server is None else server.peak_footprint_bytes,
            )
        )
    return tuple(rows)


def build_report(store: Layout, manifest: Manifest, screen: CorrectionScreen, session: str, generated: str) -> Report:
    """Rebuild the screening from the stored runs and verdicts, then add the owner's labels from ``session``."""
    unfinished = [CANDIDATES[c.id] for c in screen.candidates if not c.complete]
    gathered = gather(
        store,
        manifest,
        [CANDIDATES[c] for c in screen.screened],
        unfinished,
        seed=screen.judge.seed,
        sample=screen.judge.sample,
    )
    assessment = assess(store, manifest, gathered)
    if not assessment.judge.complete:
        raise ConfigError(
            f"the judge sample is incomplete ({assessment.judge.answered} of {assessment.judge.sample} answered); "
            "rerun `screen`, which resumes from its cache"
        )
    directory = store.session_dir(session)
    human = agreement = None
    if (directory / KEY_FILE).exists():
        key, decisions = load_key(directory), read_decisions(directory)
        by_id = {s.id: s for s in gathered.spans}
        edits = audible_edits(gathered.spans, key.challenger)
        hours = _hours(manifest)
        human = human_estimates(
            key,
            decisions,
            edits={lang: sum(s.language == lang for s in edits) for lang in LANGUAGES},
            hours=hours,
        )
        labels: dict[str, int] = {}
        for entry in key.entries:
            span, verdict = by_id.get(entry.source), assessment.verdicts.get(entry.source)
            if span is None:
                raise ConfigError(
                    f"session {session!r} holds spans the screened candidates no longer have; rebuild it with `session`"
                )
            label = None if verdict is None else judge_label(span, verdict, key.challenger)
            if label is not None:
                labels[entry.id] = label
        agreement = judge_agreement(labels, session_answers(key, decisions)[0])
    return Report(
        generated=generated,
        files=file_rows(store, manifest),
        judge=assessment.judge,
        correction=CorrectionReport(candidates=assessment.candidates, human=human),
        agreement=agreement,
        guards=assessment.guards,
        reference=reference_scores(store, manifest, gathered, assessment.verdicts),
        cost=cost_rows(store, manifest, gathered),
    )


def _estimate(value: Estimate | None, spec: str = ".3f") -> str:
    return "—" if value is None else f"{value.value:{spec}} [{value.low:{spec}}, {value.high:{spec}}]"


def _generic_run_dir(candidate_id: str) -> str:
    return f"correction/runs/{candidate_id} under $YTVERBATIM_EVAL_DIR"


def candidate_table(candidates: Sequence[CandidateScreen]) -> list[str]:
    lines = [
        "| candidate | run | audible edits | judged | needs audio | fix precision [95% CI] "
        "| over-correction [95% CI] | net errors removed/h [95% CI] | number violations "
        "| first-try valid JSON | invalid JSON after retry |",
        "|" + "---|" * 11,
    ]
    for c in candidates:
        j = c.judge
        validity = "—" if c.first_try_validity is None else f"{c.first_try_validity:.1%}"
        lines.append(
            f"| {c.id}{' (local)' if c.local else ''} | {'complete' if c.complete else 'incomplete'} | {j.edits} "
            f"| {j.decided} | {j.needs_audio} | {_estimate(j.fix_precision)} | {_estimate(j.over_correction)} "
            f"| {_estimate(j.net_per_hour, '.2f')} | {c.number_violations} | {validity} "
            f"| {'yes' if c.llm_output_failed else 'no'} |"
        )
    return lines


def pick_lines(candidates: Sequence[CandidateScreen], chosen: Pick) -> list[str]:
    head = (
        f"Judge pick: **{chosen.candidate}** ({chosen.reason})."
        if chosen.candidate
        else f"Judge pick: none ({chosen.reason})."
    )
    lines = [head]
    lines += [f"- {name} is ineligible: {', '.join(why)}." for name, why in chosen.ineligible.items()]
    lines += [f"- {failure_note(CANDIDATES[c.id], _generic_run_dir(c.id))}." for c in candidates if c.llm_output_failed]
    return lines


def screen_text(screen: CorrectionScreen) -> str:
    """Summarize a screening for the console."""
    j = screen.judge
    lines = [
        f"{j.spans} disputed spans ({j.format_only_spans} differ only in written form and are excluded); judge "
        f"{j.model}, {j.answered} of {j.sample} sampled spans answered.",
        *candidate_table(screen.candidates),
        *pick_lines(screen.candidates, screen.pick),
    ]
    if not j.complete:
        lines.append(
            f"The judge sample is incomplete ({j.sample - j.answered} spans left); rerun `screen` to resume, "
            "before `session`. The pick above is provisional."
        )
    return "\n".join(lines)


def _human_rows(report: Report) -> list[str]:
    human = report.correction.human
    if human is None:
        return ["No session yet."]
    head = (
        f"Session of **{human.candidate}** against the base: {human.spans - human.undecided} of {human.spans} "
        "spans decided."
    )
    if human.overall is None:
        return [head, "", "No decided spans."]
    lines = [
        head,
        "Fix precision and over-correction are weighted back from the stratified sample; Δ errors per hour is negative "
        "when the candidate leaves fewer errors.",
        "",
        "| slice | decided | effective n | fix precision [95% CI] | over-correction [95% CI] | both fine | both wrong "
        "| audible edits | hours | Δ errors/h [95% CI] |",
        "|" + "---|" * 10,
    ]
    for name, e in [("overall", human.overall), *human.by_language.items()]:
        lines.append(
            f"| {name} | {e.answered} | {e.effective_n:.1f} | {_estimate(e.fix_precision)} "
            f"| {_estimate(e.over_correction)} | {e.both_fine:.3f} | {e.both_wrong:.3f} | {e.edits} | {e.hours:.2f} "
            f"| {_estimate(e.delta_per_hour, '.2f')} |"
        )
    agreement = report.agreement
    lines += [""]
    if agreement is None:
        lines.append("Judge-human agreement: no decisive span was also decided by the judge.")
    else:
        lines.append(
            f"Judge-human agreement on decisive spans: {agreement.agree} of {agreement.decisive} "
            f"({agreement.rate:.3f} [{agreement.low:.3f}, {agreement.high:.3f}])."
        )
    return lines


def _share(change: ErrorChange) -> str:
    removed = change.removed_share
    return f"{change.before} → {change.after} ({'—' if removed is None else f'{removed:+.1%}'} removed)"


def render_markdown(report: Report) -> str:
    j = report.judge
    candidates = report.correction.candidates
    lines = [
        "# Correction-LLM eval",
        "",
        f"Generated {report.generated}. Aggregates only; files carry generic labels.",
        "",
        "## Files",
        "",
        *files_table(report.files),
        "",
        "## Judge screening",
        "",
        f"{j.spans} disputed spans over the base and the candidates' texts ({j.format_only_spans} differ only in "
        f"written form and are excluded). Judge: {j.model} ({j.prompt_version}), seed {j.seed}, {j.answered} of "
        f"{j.sample} sampled spans answered. The judge ranks the candidates; it does not decide.",
        "",
        *candidate_table(candidates),
        "",
        *pick_lines(candidates, pick(candidates)),
        "",
        "## Guard false-rejects",
        "",
        f"Disputes sampled from rejected proposals: {j.rejection_sample}, {j.rejection_answered} answered. A false "
        "reject is a proposal whose every audible dispute went to the proposed reading. Numeric guards are "
        "informational: a text judge cannot verify numbers.",
        "",
        "| guard | sampled proposals | judged | false rejects | share [95% CI] | note |",
        "|---|---|---|---|---|---|",
        *(
            f"| {g.guard} | {g.sampled} | {g.judged} | {g.false_rejects} | {_estimate(g.rate)} "
            f"| {'informational' if g.informational else ''} |"
            for g in report.guards
        ),
        "",
        "## Owner's labels",
        "",
        *_human_rows(report),
        "",
        "## Manual reference check",
        "",
        "Errors against the manual captions, before → after correction. Correction never touches the regions the "
        "captions rewrite, so the difference isolates its edits. One file only, so it validates the judge and gates "
        "nothing.",
        "",
        "| candidate | CER (itn) | CER (raw, case kept) | MER (itn) | judge net errors removed/h on these files |",
        "|---|---|---|---|---|",
        *(
            f"| {r.candidate} | {_share(r.cer)} | {_share(r.cer_raw)} | {_share(r.mer)} "
            f"| {_estimate(r.judge_net_per_hour, '.2f')} |"
            for r in report.reference
        ),
        "",
        "## Cost",
        "",
        "| candidate | calls | batches | input tokens | output tokens | wall time s | peak footprint GB |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {c.candidate} | {c.calls} | {c.batches} | {c.input_tokens} | {c.output_tokens} | {c.wall_s:.1f} "
            f"| {'—' if c.peak_footprint_bytes is None else f'{c.peak_footprint_bytes / 1e9:.2f}'} |"
            for c in report.cost
        ),
        "",
    ]
    return "\n".join(lines)


def render_decision(decision: CorrectionDecision) -> str:
    lines = [
        "## Decision",
        "",
        f"**{decision.summary}.**",
        "",
        "| check | result | detail |",
        "|---|---|---|",
        *(f"| {c.name} | {'pass' if c.passed else 'fail'} | {c.detail} |" for c in decision.checks),
        "",
        "Rule: the judge picks the best eligible candidate by net errors removed per hour (a local one whose interval "
        "overlaps it is preferred); only that candidate's edits go to the owner. It ships default-on only if every "
        "session span is decided, over-correction is at most 15%, the upper 95% bound of the change in errors per "
        "hour is below 0, no number changed and no correction call returned invalid JSON after its retry. "
        "Otherwise correction stays default-off.",
        "",
    ]
    return "\n".join(lines)
