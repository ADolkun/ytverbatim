"""Aggregate report: screening, the absolute error against the one manual reference, the A/B result and speed.

Only aggregates leave this module. Files appear as generic labels (``zh-01``), never by id or title, and no
transcript, reference or context text is copied into the report.
"""

from collections import Counter
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from evals.listen import LoggedDecision, read_decisions
from evals.manifest import Manifest
from evals.metrics import weighted_wilson
from evals.normalize import Language
from evals.screen import LANGUAGES, Quality, Rate, ScreenResult, screen_table
from evals.session import KEY_FILE, KeyEntry, SessionKey, load_key, load_shown
from evals.store import REFERENCE, Layout, rows_by_file
from ytverbatim.errors import ConfigError

SCREEN_FILE = "screen.json"
REPORT_JSON = "report.json"
REPORT_MD = "report.md"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WinStats(_Frozen):
    """Challenger against control on one slice. ``both``/``neither``/``skipped`` never enter the win rate."""

    challenger_wins: int
    control_wins: int
    both: int
    neither: int
    skipped: int
    rate: float | None
    effective_n: float | None
    low: float | None
    high: float | None

    @property
    def decisive(self) -> int:
        return self.challenger_wins + self.control_wins


class ABResult(_Frozen):
    session: str
    control: str
    challenger: str
    seed: int
    spans: int
    answered: int
    # Spans whose latest decision is a skip, or that have none: `decide` refuses until this is zero.
    undecided: int
    overall: WinStats
    by_language: dict[Language, WinStats]
    numeric: WinStats
    numeric_by_language: dict[Language, WinStats]
    by_file: dict[str, WinStats]


class FileRow(_Frozen):
    label: str
    language: Language
    access: str
    minutes: float
    has_reference: bool


class Report(_Frozen):
    generated: str
    files: tuple[FileRow, ...]
    screen: ScreenResult
    ab: ABResult | None


def win_stats(pairs: Sequence[tuple[KeyEntry, LoggedDecision]]) -> WinStats:
    counts = Counter[str]()
    outcomes: list[tuple[bool, float]] = []
    for entry, decision in pairs:
        if decision.choice in ("1", "2"):
            challenger_won = int(decision.choice) == entry.challenger_option
            counts["challenger" if challenger_won else "control"] += 1
            outcomes.append((challenger_won, entry.weight))
        else:
            counts[decision.choice] += 1
    weighted = weighted_wilson(outcomes) if outcomes else None
    return WinStats(
        challenger_wins=counts["challenger"],
        control_wins=counts["control"],
        both=counts["both"],
        neither=counts["neither"],
        skipped=counts["skip"],
        rate=weighted.rate if weighted else None,
        effective_n=weighted.effective_n if weighted else None,
        low=weighted.low if weighted else None,
        high=weighted.high if weighted else None,
    )


def ab_result(layout: Layout, name: str, labels: dict[str, str]) -> ABResult:
    directory = layout.session_dir(name)
    key: SessionKey = load_key(directory)
    shown = load_shown(directory)
    entries = {entry.id: entry for entry in key.entries}
    # A revisited span is appended again; its latest decision is the one that counts.
    decisions = {d.span: d for d in read_decisions(directory)}
    unknown = sorted(set(decisions) - set(entries))
    if unknown:
        raise ConfigError(f"session {name!r} has decisions for unknown spans: {', '.join(unknown[:5])}")
    pairs = [(entries[span], decision) for span, decision in decisions.items()]
    answered = sum(1 for _, decision in pairs if decision.choice != "skip")

    def where(keep: Callable[[KeyEntry], bool]) -> WinStats:
        return win_stats([(e, d) for e, d in pairs if keep(e)])

    return ABResult(
        session=name,
        control=key.control,
        challenger=key.challenger,
        seed=key.seed,
        spans=len(shown),
        answered=answered,
        undecided=len(entries) - answered,
        overall=where(lambda e: True),
        by_language={lang: where(lambda e, lang=lang: e.language == lang) for lang in LANGUAGES},
        numeric=where(lambda e: e.stratum == "numeric"),
        numeric_by_language={
            lang: where(lambda e, lang=lang: e.language == lang and e.stratum == "numeric") for lang in LANGUAGES
        },
        by_file={labels[f]: where(lambda e, f=f: e.file == f) for f in sorted({e.file for e in key.entries})},
    )


def file_rows(layout: Layout, manifest: Manifest) -> tuple[FileRow, ...]:
    labels = manifest.labels()
    references = rows_by_file(layout, REFERENCE)
    return tuple(
        FileRow(
            label=labels[v.id],
            language=v.view_language,
            access=v.access,
            minutes=round(v.audio_duration_s / 60, 1),
            has_reference=v.id in references,
        )
        for v in sorted(manifest.videos, key=lambda v: labels[v.id])
    )


def build_report(
    layout: Layout, manifest: Manifest, screen: ScreenResult, session: str | None, generated: str
) -> Report:
    labels = manifest.labels()
    ab = None
    if session is not None and (layout.session_dir(session) / KEY_FILE).exists():
        ab = ab_result(layout, session, labels)
    return Report(generated=generated, files=file_rows(layout, manifest), screen=screen, ab=ab)


def files_table(files: Sequence[FileRow]) -> list[str]:
    return [
        "| label | language view | access | minutes | manual reference |",
        "|---|---|---|---|---|",
        *(
            f"| {f.label} | {f.language} | {f.access} | {f.minutes} | {'yes' if f.has_reference else 'no'} |"
            for f in files
        ),
    ]


def _ci(stats: WinStats) -> str:
    if stats.rate is None or stats.low is None or stats.high is None:
        return "— (no decisive answers)"
    return f"{stats.rate:.3f} [{stats.low:.3f}, {stats.high:.3f}]"


def _ab_row(name: str, stats: WinStats) -> str:
    n_eff = f"{stats.effective_n:.1f}" if stats.effective_n is not None else "—"
    return (
        f"| {name} | {_ci(stats)} | {stats.challenger_wins} | {stats.control_wins} | {n_eff} | {stats.both} "
        f"| {stats.neither} | {stats.skipped} |"
    )


def _rate(rate: Rate | None) -> str:
    return f"{rate.rate:.2%} ({rate.reference_units} units)" if rate else "—"


def _quality_rows(report: Report) -> list[str]:
    lines = [
        "| hypothesis | language | files | hours | rep/h | token-cap/h | non-speech units | re-decoded chunks "
        "| anomalies |"
    ]
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in report.screen.hypotheses:
        for lang in LANGUAGES:
            q: Quality = row.quality[lang]
            if q.files:
                lines.append(
                    f"| {row.hypothesis} | {lang} | {q.files} | {q.hours:.2f} | {q.repetition_per_hour:.2f} "
                    f"| {q.token_cap_per_hour:.2f} | {q.nonspeech_share:.2%} | {q.redecode_share:.2%} "
                    f"| {q.decoder_anomalies} |"
                )
    return lines


def render_markdown(report: Report) -> str:
    screen = report.screen
    lines = [
        "# ASR selection eval",
        "",
        f"Generated {report.generated}. Aggregates only; files carry generic labels.",
        "",
        "## Files",
        "",
        *files_table(report.files),
        "",
        "## Screening",
        "",
        f"{screen.spans} disputed spans ({screen.format_only_spans} differ only in written form and are excluded). "
        f"Judge: {screen.judge_model or 'not run'} ({screen.judge_prompt_version or '—'}), seed {screen.judge_seed}, "
        f"{screen.judge_answered} of {screen.judge_sample} sampled spans answered. The judge screens only; it does "
        "not decide.",
        "",
        screen_table(screen),
        "",
        f"Recommended control: **{screen.recommendation.control or '—'}** ({screen.recommendation.control_reason}).",
        f"Recommended challenger: **{screen.recommendation.challenger or '—'}** "
        f"({screen.recommendation.challenger_reason}).",
        "",
        "## Absolute error against the manual reference",
        "",
        "One file has human captions; these are the only absolute numbers in this eval.",
        "",
        "| hypothesis | CER (itn) | CER (raw, case kept) | MER (itn) | WER vs YouTube auto captions (en, info only) |",
        "|---|---|---|---|---|",
        *(
            f"| {h.hypothesis} | {_rate(h.reference_cer.get('zh'))} | {_rate(h.reference_cer_raw.get('zh'))} "
            f"| {_rate(h.reference_mer.get('zh'))} | {_rate(h.caption_wer)} |"
            for h in screen.hypotheses
        ),
        "",
        "## Quality gates (inputs)",
        "",
        *_quality_rows(report),
        "",
        "## Speed and memory",
        "",
        "| hypothesis | RTF (decode / file audio) | load s | peak footprint GB | MLX peak GB | max RSS GB |",
        "|---|---|---|---|---|---|",
        *(
            f"| {h.hypothesis} | {_opt(h.speed.rtf, '.3f')} | {_opt(h.speed.load_s, '.1f')} "
            f"| {_opt(_gb(h.speed.peak_footprint_bytes), '.2f')} | {_opt(_gb(h.speed.peak_memory_bytes), '.2f')} "
            f"| {_opt(_gb(h.speed.max_rss_bytes), '.2f')} |"
            for h in screen.hypotheses
        ),
        "",
    ]
    lines += _ab_markdown(report.ab)
    return "\n".join(lines) + "\n"


def _ab_markdown(ab: ABResult | None) -> list[str]:
    if ab is None:
        return ["## Blind A/B", "", "No A/B session yet.", ""]
    header = (
        "| slice | challenger win rate [95% CI] | challenger | control | effective n | both fine | both wrong "
        "| skipped |"
    )
    return [
        "## Blind A/B",
        "",
        f"Session `{ab.session}`: control **{ab.control}** vs challenger **{ab.challenger}**, seed {ab.seed}, "
        f"{ab.answered} of {ab.spans} spans answered. Win rates are weighted back from the stratified sample; "
        "'both fine', 'both wrong' and skips are excluded from them.",
        "",
        header,
        "|---|---|---|---|---|---|---|---|",
        _ab_row("overall", ab.overall),
        *(_ab_row(lang, s) for lang, s in ab.by_language.items() if s.decisive or s.both or s.neither),
        _ab_row("numeric/entity", ab.numeric),
        *(_ab_row(f"numeric/entity {lang}", s) for lang, s in ab.numeric_by_language.items() if s.decisive),
        *(_ab_row(label, s) for label, s in ab.by_file.items()),
        "",
    ]


def _gb(value: int | None) -> float | None:
    return value / 1e9 if value is not None else None


def _opt(value: float | None, spec: str) -> str:
    return format(value, spec) if value is not None else "—"
