"""The amended decision rule, applied per language view; anything short of a clear win keeps Whisper.

Nothing is decided until every A/B span has a decision (1, 2, both or neither); skips must be revisited.

A challenger replaces the control anywhere only if its overall blind A/B win rate, over both languages, has a
Wilson 95% lower bound above 0.5. It then replaces the control in a language only if, on that language's data:
1. its blind A/B win rate has a Wilson 95% lower bound above 0.5;
2. on numeric/entity spans its win rate is at least 0.5 with a lower bound above 0.40 (non-inferiority);
3. its repetition, token-cap and non-speech rates are not worse than the control's beyond the tolerances below;
4. its CER against the manual reference is at most 1 point above the control's, where a reference exists.
Languages are decided separately, so a per-language split happens only when one language passes on its own.
"""

from pydantic import BaseModel, ConfigDict

from evals.normalize import Language
from evals.report import Report, WinStats
from evals.screen import LANGUAGES, HypothesisScreen
from ytverbatim.errors import ConfigError

WIN_LOWER_BOUND = 0.5
NUMERIC_POINT = 0.5
NUMERIC_LOWER_BOUND = 0.40
# Small absolute tolerances: about two extra events over the 4.7 h set for the per-hour rates, and half a percent
# of the output for text decoded inside VAD non-speech.
REPETITION_TOLERANCE_PER_HOUR = 0.5
TOKEN_CAP_TOLERANCE_PER_HOUR = 0.5
NONSPEECH_TOLERANCE = 0.005
CER_TOLERANCE = 0.01


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Check(_Frozen):
    name: str
    passed: bool
    detail: str


class LanguageDecision(_Frozen):
    language: Language
    challenger_wins: bool
    checks: tuple[Check, ...]


class Decision(_Frozen):
    control: str | None
    challenger: str | None
    languages: tuple[LanguageDecision, ...]
    engine: dict[Language, str | None]
    summary: str


def _win_check(stats: WinStats | None, name: str) -> Check:
    if stats is None or stats.low is None or stats.rate is None:
        return Check(name=name, passed=False, detail="no decisive A/B answers")
    return Check(
        name=name,
        passed=stats.low > WIN_LOWER_BOUND,
        detail=f"{stats.rate:.3f}, lower bound {stats.low:.3f} (needs > {WIN_LOWER_BOUND})",
    )


def _numeric_check(stats: WinStats | None) -> Check:
    if stats is None or stats.low is None or stats.rate is None:
        return Check(name="numeric/entity non-inferiority", passed=False, detail="no decisive numeric/entity answers")
    return Check(
        name="numeric/entity non-inferiority",
        passed=stats.rate >= NUMERIC_POINT and stats.low > NUMERIC_LOWER_BOUND,
        detail=(
            f"{stats.rate:.3f}, lower bound {stats.low:.3f} (needs >= {NUMERIC_POINT} and > {NUMERIC_LOWER_BOUND})"
        ),
    )


def _gate_checks(control: HypothesisScreen, challenger: HypothesisScreen, language: Language) -> list[Check]:
    c, x = control.quality[language], challenger.quality[language]
    if not x.files or not c.files:
        return [Check(name="quality gates", passed=False, detail="no complete output for this language")]
    return [
        Check(
            name="repetition",
            passed=x.repetition_per_hour <= c.repetition_per_hour + REPETITION_TOLERANCE_PER_HOUR,
            detail=f"{x.repetition_per_hour:.2f}/h vs {c.repetition_per_hour:.2f}/h "
            f"(tolerance +{REPETITION_TOLERANCE_PER_HOUR})",
        ),
        Check(
            name="token cap",
            passed=x.token_cap_per_hour <= c.token_cap_per_hour + TOKEN_CAP_TOLERANCE_PER_HOUR,
            detail=f"{x.token_cap_per_hour:.2f}/h vs {c.token_cap_per_hour:.2f}/h "
            f"(tolerance +{TOKEN_CAP_TOLERANCE_PER_HOUR})",
        ),
        Check(
            name="non-speech output",
            passed=x.nonspeech_share <= c.nonspeech_share + NONSPEECH_TOLERANCE,
            detail=f"{x.nonspeech_share:.2%} vs {c.nonspeech_share:.2%} (tolerance +{NONSPEECH_TOLERANCE:.1%})",
        ),
    ]


def _cer_check(control: HypothesisScreen, challenger: HypothesisScreen, language: Language) -> Check:
    c, x = control.reference_cer.get(language), challenger.reference_cer.get(language)
    if c is None and x is None:
        return Check(name="reference CER", passed=True, detail="no manual reference in this language")
    if c is None or x is None:
        return Check(name="reference CER", passed=False, detail="the reference was scored for only one of the two")
    return Check(
        name="reference CER",
        passed=x.rate <= c.rate + CER_TOLERANCE,
        detail=f"{x.rate:.2%} vs {c.rate:.2%} (tolerance +{CER_TOLERANCE:.0%} point)",
    )


def decide(report: Report) -> Decision:
    ab = report.ab
    if ab is None:
        control = report.screen.recommendation.control
        return Decision(
            control=control,
            challenger=None,
            languages=(),
            engine=dict.fromkeys(LANGUAGES, control),
            summary=f"no A/B session: {control or 'Whisper'} stays",
        )
    if ab.undecided:
        raise ConfigError(
            f"session {ab.session!r} has {ab.undecided} of {ab.spans} spans without a decision (1, 2, b or 0); "
            "finish them with `listen`, then run `report` again"
        )
    control, challenger = report.screen.get(ab.control), report.screen.get(ab.challenger)
    overall = _win_check(ab.overall, "overall win rate (both languages)")
    languages: list[LanguageDecision] = []
    for language in LANGUAGES:
        checks = [
            overall,
            _win_check(ab.by_language.get(language), "language win rate"),
            _numeric_check(ab.numeric_by_language.get(language)),
        ]
        if control is None or challenger is None:
            checks.append(Check(name="screening", passed=False, detail="the pair is missing from the screening"))
        else:
            checks += _gate_checks(control, challenger, language)
            checks.append(_cer_check(control, challenger, language))
        languages.append(
            LanguageDecision(language=language, challenger_wins=all(c.passed for c in checks), checks=tuple(checks))
        )
    engine: dict[Language, str | None] = {
        d.language: ab.challenger if d.challenger_wins else ab.control for d in languages
    }
    winners = [d.language for d in languages if d.challenger_wins]
    if len(winners) == len(LANGUAGES):
        summary = f"{ab.challenger} replaces {ab.control} for every language"
    elif winners:
        summary = f"per-language split: {ab.challenger} for {', '.join(winners)}; {ab.control} stays elsewhere"
    else:
        summary = f"{ab.control} stays: {ab.challenger} did not pass the rule in any language"
    return Decision(
        control=ab.control, challenger=ab.challenger, languages=tuple(languages), engine=engine, summary=summary
    )


def render_decision(decision: Decision) -> str:
    lines = ["## Decision", "", f"**{decision.summary}.**", ""]
    for language in decision.languages:
        verdict = "challenger wins" if language.challenger_wins else "control stays"
        lines += [f"### {language.language}: {verdict}", "", "| check | result | detail |", "|---|---|---|"]
        lines += [f"| {c.name} | {'pass' if c.passed else 'fail'} | {c.detail} |" for c in language.checks]
        lines.append("")
    lines += [
        "Rule: every span decided; Wilson 95% lower bound > 0.5 overall and in the language; "
        "numeric/entity point estimate >= 0.5 with lower bound > 0.40; "
        f"repetition and token-cap rates within +{REPETITION_TOLERANCE_PER_HOUR}/h and non-speech output within "
        f"+{NONSPEECH_TOLERANCE:.1%} of the control; reference CER within +{CER_TOLERANCE:.0%} point. "
        "Otherwise Whisper stays.",
        "",
    ]
    return "\n".join(lines)
