"""Metrics and the default-on rule of the correction-LLM eval; pure functions over stored outputs.

Correction changes the text only where it edits, so a candidate's net change in errors is the sum over its edits of
+1 (a fix), -1 (a break) and 0 (neither). A blind LLM judge labels a sample of edits to rank the candidates and pick one
to test; the owner's labels of that candidate's edits, reweighted by stratum, decide. No candidate ships on judge
evidence alone.

Default-on needs all of: every session span decided, and at least `MIN_DECIDED` of them; over-correction at most
`OVER_CORRECTION_MAX`; the 95% upper bound of the change in errors per hour below `DELTA_UPPER_BOUND`; no number
violation; and no correction call that raised `LLMOutputError` after its retry. The first-try JSON validity is
reported and not gated.
"""

import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evals.correction import BASE, BaseFile, CandidateFile, audible_edits
from evals.decide import Check
from evals.judge import Verdict
from evals.listen import LoggedDecision
from evals.metrics import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, percentile_index, weighted_wilson, wilson_interval
from evals.normalize import Language, canonical
from evals.screen import Span
from evals.session import SessionKey, Stratum
from ytverbatim.contract.models import Flag
from ytverbatim.errors import ConfigError
from ytverbatim.text.numbers import quantity_tokens

OVER_CORRECTION_MAX = 0.15
# With few labels the percentile bootstrap collapses (no "before" answer means it almost never reaches 0), so an
# interval bound below 0 would be meaningless; this many decided spans are needed before the interval counts.
MIN_DECIDED = 30
MAX_NUMBER_VIOLATIONS = 0
# A candidate must remove errors on the judge's reading: its net per hour must exceed this.
JUDGE_NET_FLOOR_PER_HOUR = 0.0
# The human estimate is a change in errors per hour, so negative is better; the 95% upper bound must be below this.
DELTA_UPPER_BOUND = 0.0
BOOTSTRAP_SEED = 0

Reaction = Literal["after", "before", "both", "neither"]
_SIGN: dict[Reaction, int] = {"after": 1, "before": -1, "both": 0, "neither": 0}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Estimate(_Frozen):
    value: float
    low: float
    high: float


class JudgeStats(_Frozen):
    """One candidate's judge-screened edits. The estimates are None while no edit of it has a decided verdict."""

    edits: int
    judged: int
    needs_audio: int
    decided: int
    fixes: int
    breaks: int
    fix_precision: Estimate | None
    over_correction: Estimate | None
    net_per_hour: Estimate | None


class CandidateScreen(_Frozen):
    id: str
    local: bool
    # Every base file has this candidate's output.
    complete: bool
    # A correction call raised `LLMOutputError` after its retry (U3's `failure.json`).
    llm_output_failed: bool
    number_violations: int
    # Reported only; the gate is `llm_output_failed`.
    first_try_validity: float | None
    judge: JudgeStats


class Pick(_Frozen):
    candidate: str | None
    best: str | None
    eligible: tuple[str, ...]
    ineligible: dict[str, tuple[str, ...]]
    reason: str


class HumanEstimate(_Frozen):
    answered: int
    effective_n: float
    fix_precision: Estimate
    over_correction: Estimate
    both_fine: float
    both_wrong: float
    net_per_edit: float
    edits: int
    hours: float
    delta_per_hour: Estimate


class HumanEstimates(_Frozen):
    candidate: str
    spans: int
    # Spans whose latest decision is a skip, or that have none.
    undecided: int
    overall: HumanEstimate | None
    by_language: dict[Language, HumanEstimate]


class Agreement(_Frozen):
    agree: int
    decisive: int
    rate: float
    low: float
    high: float


class CorrectionReport(_Frozen):
    candidates: tuple[CandidateScreen, ...]
    human: HumanEstimates | None


class CorrectionDecision(_Frozen):
    default_on: bool
    candidate: str | None
    checks: tuple[Check, ...]
    summary: str


def _view(language: str) -> Language:
    return "en" if language == "en" else "zh"


def _respellings(flags: Sequence[Flag], language: Language) -> Counter[tuple[str, str]]:
    # `text.correct` writes an `entity_corrected` note as "before → after".
    pairs = (f.note.partition(" → ") for f in flags if f.kind == "entity_corrected")
    return Counter((canonical(before, language), canonical(after, language)) for before, _, after in pairs)


def breaks_numbers(before: str, after: str, flags: Sequence[Flag], language: Language) -> bool:
    """Whether ``after`` changes a quantity, or changes a ticker that ``flags`` do not record as a correction.

    Each flag covers one changed ticker. A ticker respelled in place needs a flag with its before and after. One that
    vanishes into a name (open AI→OpenAI) or appears out of one has no counterpart, so a flag naming it on its own side
    covers it (``AI → OpenAI``).
    """
    tokens = [quantity_tokens(canonical(text, language)) for text in (before, after)]
    quantities = [[(t.value_itn, t.kind) for t in found if t.kind != "ticker"] for found in tokens]
    if quantities[0] != quantities[1]:
        return True
    old, new = ([t.surface for t in found if t.kind == "ticker"] for found in tokens)
    unused = _respellings(flags, language)
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if i2 - i1 == j2 - j1:
            changed = [(b, a) for b, a in zip(old[i1:i2], new[j1:j2], strict=True) if b != a]
            if not all(_use(unused, before=b, after=a) for b, a in changed):
                return True
            continue
        gone, came = Counter(old[i1:i2]), Counter(new[j1:j2])
        # A flag naming both a vanished and a new ticker of the block is a respelling and covers both.
        for b, a in list(unused):
            paired = min(unused[b, a], gone[b], came[a])
            unused[b, a] -= paired
            gone[b] -= paired
            came[a] -= paired
        if not all(_use(unused, before=b) for b in gone.elements()):
            return True
        if not all(_use(unused, after=a) for a in came.elements()):
            return True
    return False


def _use(unused: Counter[tuple[str, str]], *, before: str | None = None, after: str | None = None) -> bool:
    """Spend one unused flag with this ``before`` and ``after`` (None matches any); False when there is none."""
    pair = next(
        ((b, a) for (b, a), count in unused.items() if count > 0 and before in (None, b) and after in (None, a)),
        None,
    )
    if pair is None:
        return False
    unused[pair] -= 1
    return True


def number_violations(base: BaseFile, candidate: CandidateFile) -> int:
    """Stretches where the corrected text breaks the number rule; it must be zero."""
    language = _view(base.language)
    return sum(
        breaks_numbers(stretch.text, text, flags, language)
        for stretch, text, flags in zip(base.stretches, candidate.texts, candidate.flags, strict=True)
    )


def judge_label(span: Span, verdict: Verdict, candidate: str) -> int | None:
    """+1 when the judge chose the candidate's reading, -1 for the base's, 0 for another's; None for needs_audio."""
    if verdict.needs_audio:
        return None
    if verdict.choice == span.reading(candidate).text:
        return 1
    return -1 if verdict.choice == span.reading(BASE).text else 0


def _percentile_interval(draws: list[float], confidence: float) -> tuple[float, float]:
    draws.sort()
    tail = (1 - confidence) / 2
    return draws[percentile_index(tail, len(draws))], draws[percentile_index(1 - tail, len(draws))]


def _bootstrap_mean(
    values: Sequence[int], seed: int, resamples: int, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    draws = [sum(rng.choices(values, k=n)) / n for _ in range(resamples)]
    return _percentile_interval(draws, confidence)


def judge_stats(
    spans: Sequence[Span],
    verdicts: Mapping[str, Verdict],
    candidate: str,
    *,
    hours: float,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = DEFAULT_RESAMPLES,
) -> JudgeStats:
    """Judge shares and net errors removed per hour for ``candidate``, over every dispute of the eval.

    ``verdicts`` is keyed by span id. The net scales the judged sample's mean label by the candidate's count of
    audible edits, so the sample must be uniform; ``needs_audio`` verdicts are counted and left out of the rates, which
    assumes they would have split like the decided ones.
    """
    edits = audible_edits(spans, candidate)
    judged = [(s, verdicts[s.id]) for s in edits if s.id in verdicts]
    labels = [label for s, v in judged if (label := judge_label(s, v, candidate)) is not None]
    counts = Counter(labels)
    stats = {
        "edits": len(edits),
        "judged": len(judged),
        "needs_audio": len(judged) - len(labels),
        "decided": len(labels),
        "fixes": counts[1],
        "breaks": counts[-1],
    }
    if not labels:
        return JudgeStats(**stats, fix_precision=None, over_correction=None, net_per_hour=None)
    n = len(labels)
    scale = len(edits) / hours
    low, high = _bootstrap_mean(labels, seed, resamples)
    return JudgeStats(
        **stats,
        fix_precision=proportion(counts[1], n),
        over_correction=proportion(counts[-1], n),
        net_per_hour=Estimate(value=(counts[1] - counts[-1]) / n * scale, low=low * scale, high=high * scale),
    )


def proportion(successes: int, trials: int) -> Estimate:
    low, high = wilson_interval(successes, trials)
    return Estimate(value=successes / trials, low=low, high=high)


def ineligible_reasons(candidate: CandidateScreen) -> list[str]:
    reasons: list[str] = []
    if not candidate.complete:
        reasons.append("run incomplete")
    if candidate.llm_output_failed:
        reasons.append("a correction call raised LLMOutputError after its retry")
    if candidate.number_violations > MAX_NUMBER_VIOLATIONS:
        reasons.append(f"{candidate.number_violations} number violations")
    judge = candidate.judge
    if judge.over_correction is None or judge.net_per_hour is None:
        reasons.append("no edit with a decided judge verdict")
        return reasons
    if judge.over_correction.value > OVER_CORRECTION_MAX:
        reasons.append(f"judge over-correction {judge.over_correction.value:.3f} > {OVER_CORRECTION_MAX}")
    if judge.net_per_hour.value <= JUDGE_NET_FLOOR_PER_HOUR:
        reasons.append(f"judge net {judge.net_per_hour.value:.2f}/h <= {JUDGE_NET_FLOOR_PER_HOUR}")
    return reasons


def _overlap(a: Estimate, b: Estimate) -> bool:
    return a.low <= b.high and b.low <= a.high


def pick(candidates: Sequence[CandidateScreen]) -> Pick:
    """Choose the candidate to test with the owner: the best eligible by judge net, or a local one overlapping it."""
    reasons = {c.id: tuple(r) for c in candidates if (r := ineligible_reasons(c))}
    nets = [(c, c.judge.net_per_hour) for c in candidates if c.id not in reasons and c.judge.net_per_hour]
    if not nets:
        return Pick(candidate=None, best=None, eligible=(), ineligible=reasons, reason="no eligible candidate")
    best, best_net = max(nets, key=lambda pair: pair[1].value)
    local = [(c, net) for c, net in nets if c.local and _overlap(net, best_net)]
    chosen = max(local, key=lambda pair: pair[1].value)[0] if local else best
    why = f"{best.id} has the best judge net ({best_net.value:.2f}/h)"
    if chosen is not best:
        why += f"; {chosen.id} is local and its interval overlaps, so it is preferred"
    return Pick(
        candidate=chosen.id,
        best=best.id,
        eligible=tuple(c.id for c, _ in nets),
        ineligible=reasons,
        reason=why,
    )


@dataclass(frozen=True, slots=True)
class Answer:
    span: str
    language: Language
    stratum: Stratum
    weight: float
    verdict: Reaction

    @property
    def sign(self) -> int:
        return _SIGN[self.verdict]


def session_answers(key: SessionKey, decisions: Sequence[LoggedDecision]) -> tuple[list[Answer], int]:
    """Return the decided spans of a session and how many are still skipped or unanswered."""
    entries = {e.id: e for e in key.entries}
    latest = {d.span: d for d in decisions}
    unknown = sorted(set(latest) - set(entries))
    if unknown:
        raise ConfigError(f"the session has decisions for unknown spans: {', '.join(unknown[:5])}")
    answers: list[Answer] = []
    for span, decision in latest.items():
        if decision.choice == "skip":
            continue
        entry = entries[span]
        if decision.choice in ("1", "2"):
            verdict: Reaction = "after" if int(decision.choice) == entry.challenger_option else "before"
        else:
            verdict = "both" if decision.choice == "both" else "neither"
        answers.append(Answer(span, entry.language, entry.stratum, entry.weight, verdict))
    return answers, len(entries) - len(answers)


def _stratified_mean_interval(
    answers: Sequence[Answer], seed: int, resamples: int, confidence: float
) -> tuple[float, float]:
    """Percentile interval of the weighted mean sign, resampling inside each (language, stratum) and keeping weights."""
    strata: dict[tuple[str, str], list[Answer]] = {}
    for a in answers:
        strata.setdefault((a.language, a.stratum), []).append(a)
    cells = [strata[key] for key in sorted(strata)]
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        weight = signed = 0.0
        for cell in cells:
            for a in rng.choices(cell, k=len(cell)):
                weight += a.weight
                signed += a.weight * a.sign
        draws.append(signed / weight)
    return _percentile_interval(draws, confidence)


def _weighted_share(answers: Sequence[Answer], keep: Callable[[Answer], bool]) -> float:
    return sum(a.weight for a in answers if keep(a)) / sum(a.weight for a in answers)


def human_estimate(
    answers: Sequence[Answer],
    *,
    edits: int,
    hours: float,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = DEFAULT_RESAMPLES,
) -> HumanEstimate:
    """Fix precision, over-correction and the change in errors per hour from stratified, weighted labels.

    ``edits`` is the candidate's number of audible edits in the slice and ``hours`` the slice's audio; both are fixed
    in the bootstrap, which resamples only the labels.
    """
    precision = weighted_wilson([(a.verdict == "after", a.weight) for a in answers])
    over = weighted_wilson([(a.verdict == "before", a.weight) for a in answers])
    mean = sum(a.weight * a.sign for a in answers) / sum(a.weight for a in answers)
    low, high = _stratified_mean_interval(answers, seed, resamples, DEFAULT_CONFIDENCE)
    scale = edits / hours
    return HumanEstimate(
        answered=len(answers),
        effective_n=precision.effective_n,
        fix_precision=Estimate(value=precision.rate, low=precision.low, high=precision.high),
        over_correction=Estimate(value=over.rate, low=over.low, high=over.high),
        both_fine=_weighted_share(answers, lambda a: a.verdict == "both"),
        both_wrong=_weighted_share(answers, lambda a: a.verdict == "neither"),
        net_per_edit=mean,
        edits=edits,
        hours=hours,
        delta_per_hour=Estimate(value=-mean * scale, low=-high * scale, high=-low * scale),
    )


def human_estimates(
    key: SessionKey,
    decisions: Sequence[LoggedDecision],
    *,
    edits: Mapping[Language, int],
    hours: Mapping[Language, float],
    seed: int = BOOTSTRAP_SEED,
    resamples: int = DEFAULT_RESAMPLES,
) -> HumanEstimates:
    """Pooled and per-language estimates for the session's challenger; ``edits`` and ``hours`` are per language."""
    answers, undecided = session_answers(key, decisions)
    by_language = {
        language: human_estimate(
            [a for a in answers if a.language == language],
            edits=edits[language],
            hours=hours[language],
            seed=seed,
            resamples=resamples,
        )
        for language in sorted({a.language for a in answers})
    }
    overall = (
        human_estimate(answers, edits=sum(edits.values()), hours=sum(hours.values()), seed=seed, resamples=resamples)
        if answers
        else None
    )
    return HumanEstimates(
        candidate=key.challenger,
        spans=len(key.entries),
        undecided=undecided,
        overall=overall,
        by_language=by_language,
    )


def judge_agreement(labels: Mapping[str, int], answers: Sequence[Answer]) -> Agreement | None:
    """Share of the owner's decisive spans, among those the judge also decided, where the judge chose the same side.

    ``labels`` maps span id to `judge_label` of the tested candidate, leaving out undecided ones.
    """
    decisive = [(labels[a.span], a.sign) for a in answers if a.span in labels and a.sign != 0]
    if not decisive:
        return None
    agree = sum(label == sign for label, sign in decisive)
    low, high = wilson_interval(agree, len(decisive))
    return Agreement(agree=agree, decisive=len(decisive), rate=agree / len(decisive), low=low, high=high)


def require_decided(human: HumanEstimates) -> None:
    if human.undecided:
        raise ConfigError(
            f"the session has {human.undecided} of {human.spans} spans without a decision (1, 2, b or 0); "
            "finish them with `listen`, then run `report` again"
        )


def decide(report: CorrectionReport) -> CorrectionDecision:
    """Apply the default-on rule; refuses while the owner's session has an undecided span."""
    chosen = pick(report.candidates)
    if chosen.candidate is None:
        detail = "; ".join(f"{name}: {', '.join(why)}" for name, why in chosen.ineligible.items())
        return CorrectionDecision(
            default_on=False,
            candidate=None,
            checks=(Check(name="eligible candidate", passed=False, detail=detail or "no candidates"),),
            summary="correction stays default-off: no candidate passed the screening",
        )
    screen = next(c for c in report.candidates if c.id == chosen.candidate)
    checks = [Check(name="eligible candidate", passed=True, detail=f"{chosen.candidate}: {chosen.reason}")]
    human = report.human
    if human is None:
        checks.append(Check(name="human session", passed=False, detail="no session"))
        return _decision(chosen.candidate, checks)
    if human.candidate != chosen.candidate:
        raise ConfigError(
            f"the session tests {human.candidate!r} but the judge picked {chosen.candidate!r}; "
            "rebuild the session with `session`"
        )
    require_decided(human)
    checks.append(Check(name="every span decided", passed=True, detail=f"{human.spans} spans"))
    estimate = human.overall
    if estimate is None:
        checks.append(Check(name="human estimate", passed=False, detail="the session has no spans"))
        return _decision(chosen.candidate, checks)
    over, delta = estimate.over_correction.value, estimate.delta_per_hour
    checks += [
        Check(
            name="enough labels",
            passed=estimate.answered >= MIN_DECIDED,
            detail=(
                f"{estimate.answered} decided spans (needs >= {MIN_DECIDED})"
                if estimate.answered >= MIN_DECIDED
                else f"too few labels: {estimate.answered} decided spans (needs >= {MIN_DECIDED})"
            ),
        ),
        Check(
            name="over-correction",
            passed=over <= OVER_CORRECTION_MAX,
            detail=f"{over:.3f} (needs <= {OVER_CORRECTION_MAX})",
        ),
        Check(
            name="errors per hour",
            passed=delta.high < DELTA_UPPER_BOUND,
            detail=f"{delta.value:.2f} [{delta.low:.2f}, {delta.high:.2f}] (upper bound needs < {DELTA_UPPER_BOUND})",
        ),
        Check(
            name="number preservation",
            passed=screen.number_violations <= MAX_NUMBER_VIOLATIONS,
            detail=f"{screen.number_violations} violations (needs <= {MAX_NUMBER_VIOLATIONS})",
        ),
        Check(
            name="JSON after the retry",
            passed=not screen.llm_output_failed,
            detail="a correction call raised LLMOutputError" if screen.llm_output_failed else "no LLMOutputError",
        ),
    ]
    return _decision(chosen.candidate, checks)


def _decision(candidate: str, checks: Sequence[Check]) -> CorrectionDecision:
    on = all(c.passed for c in checks)
    failed = ", ".join(c.name for c in checks if not c.passed)
    return CorrectionDecision(
        default_on=on,
        candidate=candidate,
        checks=tuple(checks),
        summary=(
            f"correction ships default-on with {candidate}"
            if on
            else f"correction stays default-off: {candidate} failed {failed}"
        ),
    )
