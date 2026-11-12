"""Correction-LLM eval runner: ``uv run python -m evals.run_correction <command>``.

Order: ``base --hyp H`` → ``correct --candidate C`` per candidate (then ``cleanup --candidate C`` for a local one) →
``screen`` → ``session`` → ``listen`` → ``report`` → ``decide``. Every command is resumable and writes only under
``$YTVERBATIM_EVAL_DIR`` (``correction/``, and the session under ``ab/``), except ``decide``, which writes the
aggregate ``evals/results/correction-<date>.md``.
"""

import contextlib
from contextlib import AbstractContextManager
from datetime import date
from pathlib import Path

import click

from evals.correction import (
    CANDIDATES,
    SESSION_N,
    SESSION_NAME,
    Candidate,
    CandidateFile,
    build_base,
    cleanup_candidate,
    failure_note,
    first_try_validity,
    load_failure,
    local_server,
    pending_files,
    run_candidate,
    run_dir,
    session_for,
)
from evals.correction_metrics import decide, require_decided
from evals.correction_report import (
    ITEMS_FILE,
    REJECTIONS_FILE,
    REPORT_JSON,
    REPORT_MD,
    SCREEN_FILE,
    Report,
    assess,
    build_report,
    build_screen,
    gather,
    judge_dir,
    load_screen,
    render_decision,
    render_markdown,
    resolve_candidates,
    screen_text,
)
from evals.judge import JUDGE_MODEL, run_judge
from evals.listen import listen as listen_loop
from evals.manifest import load_manifest
from evals.run_asr import RESULTS_DIR, default_hf_cache, layout
from evals.session import write_session
from evals.store import Layout, dump_jsonl, now, write_json
from ytverbatim import atomic, config
from ytverbatim.asr import registry
from ytverbatim.asr.runner import memory_cap_bytes
from ytverbatim.errors import ConfigError, YtverbatimError
from ytverbatim.llm import ClaudeCLIClient, apply_spec, claude_cli, client_from_options
from ytverbatim.llm.protocol import LLMClient, Readiness

DEFAULT_PUNCTUATION_LLM = "claude-cli:claude-sonnet-5-5"
JUDGE_SAMPLE = 1500
_LOCAL = sorted(c.id for c in CANDIDATES.values() if c.local)


def _files(value: str | None) -> list[str]:
    return [part for part in (value or "").split(",") if part]


def _require_ready(readiness: Readiness) -> None:
    if not readiness.ready:
        raise ConfigError(readiness.detail)


def punctuation_client(spec: str) -> LLMClient:
    options = apply_spec(config.LLMOptions(), spec)
    client = client_from_options(options)
    if options.provider == "claude-cli":
        _require_ready(claude_cli.readiness())
    return client


def judge_client() -> LLMClient:
    return ClaudeCLIClient(model=JUDGE_MODEL)


def open_client(store: Layout, candidate: Candidate, cache: Path) -> AbstractContextManager[LLMClient]:
    if candidate.local:
        click.echo(f"{candidate.id}: starting the local server (waits for the MLX worker lock if another one is held)")
        cap = memory_cap_bytes(config.load_options().asr.memory_cap_gib)
        return local_server(candidate, cache, run_dir(store, candidate), memory_cap=cap)
    _require_ready(claude_cli.readiness())
    return contextlib.nullcontext(ClaudeCLIClient(model=candidate.model))


@click.group()
def main() -> None:
    """Correction-LLM eval (data under $YTVERBATIM_EVAL_DIR/correction)."""


_files_option = click.option("--files", help="Comma-separated video ids (default: every file in the manifest).")
_cache_option = click.option(
    "--hf-cache",
    type=click.Path(path_type=Path),
    default=None,
    help="Candidate weight cache (default: $YTVERBATIM_HOME/hf-candidates). Never the default HF cache.",
)


@main.command()
@click.option("--hyp", "hypothesis", required=True, type=click.Choice(sorted(registry.HYPOTHESES)))
@click.option(
    "--punct-llm",
    default=DEFAULT_PUNCTUATION_LLM,
    show_default=True,
    help="PROVIDER:MODEL that punctuates the base, once.",
)
@_files_option
def base(hypothesis: str, punct_llm: str, files: str | None) -> None:
    """Normalize and punctuate a complete hypothesis into the base every candidate corrects."""
    store, manifest = layout(), load_manifest()
    outcome = build_base(store, manifest, hypothesis, _files(files), punctuation_client(punct_llm))
    calls = sum(u.calls for built in outcome.built for u in built.usage)
    click.echo(
        f"base from {hypothesis}: {len(outcome.built)} files built ({calls} punctuation calls), "
        f"{len(outcome.reused)} already done"
    )


@main.command()
@click.option("--candidate", "candidate_id", required=True, type=click.Choice(sorted(CANDIDATES)))
@_files_option
@_cache_option
def correct(candidate_id: str, files: str | None, hf_cache: Path | None) -> None:
    """Run the production correction over the base with one candidate (resumes per batch)."""
    store, manifest = layout(), load_manifest()
    candidate = CANDIDATES[candidate_id]
    pending = pending_files(store, manifest, candidate, _files(files))
    if not pending:
        click.echo(f"{candidate_id}: complete for all selected files")
        return
    with open_client(store, candidate, hf_cache or default_hf_cache()) as client:
        outcome = run_candidate(store, manifest, candidate, pending, client)
    click.echo(_summary(candidate, outcome.corrected))
    if load_failure(store, candidate) is not None:
        click.echo(f"note: {failure_note(candidate, str(run_dir(store, candidate)))}")


def _summary(candidate: Candidate, corrected: tuple[CandidateFile, ...]) -> str:
    validity = first_try_validity(corrected)
    return (
        f"{candidate.id}: {len(corrected)} files corrected, {sum(f.calls for f in corrected)} calls over "
        f"{sum(f.batches for f in corrected)} batches, first-try valid JSON "
        f"{'n/a' if validity is None else f'{validity:.1%}'}, "
        f"{sum(len(e) for f in corrected for e in f.edits)} edits applied, "
        f"{sum(len(f.rejections) for f in corrected)} proposals rejected"
    )


@main.command()
@click.option("--candidate", "candidate_id", required=True, type=click.Choice(_LOCAL))
@_cache_option
def cleanup(candidate_id: str, hf_cache: Path | None) -> None:
    """Verify a local candidate is complete, then delete its weights from the candidate cache."""
    candidate = CANDIDATES[candidate_id]
    freed = cleanup_candidate(layout(), load_manifest(), candidate, hf_cache or default_hf_cache())
    click.echo(f"deleted {candidate.model}: {freed / 1e9:.2f} GB freed")


@main.command()
@click.option("--candidates", help="Comma-separated candidate ids to compare (default: every finished run).")
@click.option("--judge-n", default=JUDGE_SAMPLE, show_default=True, help="Edit disputes sampled for the judge.")
@click.option("--seed", default=0, show_default=True)
@click.option("--max-calls", type=int, default=None, help="Stop after this many judge calls (quota).")
@click.option("--no-judge", is_flag=True, help="Use the cached verdicts only; make no judge call.")
def screen(candidates: str | None, judge_n: int, seed: int, max_calls: int | None, no_judge: bool) -> None:
    """Find every candidate's edits, have the blind judge sample them, and pick the candidate to test."""
    store, manifest = layout(), load_manifest()
    screened, unfinished = resolve_candidates(store, manifest, _files(candidates))
    gathered = gather(store, manifest, screened, unfinished, seed=seed, sample=judge_n)
    dump_jsonl(store.correction / ITEMS_FILE, gathered.spans)
    dump_jsonl(store.correction / REJECTIONS_FILE, gathered.rejections)
    if not no_judge:
        run_judge(
            judge_client(),
            gathered.to_judge,
            judge_dir(store),
            model=JUDGE_MODEL,
            max_calls=max_calls,
            progress=click.echo,
        )
    result = build_screen(gathered, assess(store, manifest, gathered), now())
    write_json(store.correction / SCREEN_FILE, result)
    click.echo(screen_text(result))


@main.command()
@click.option(
    "--candidate",
    "candidate_id",
    type=click.Choice(sorted(CANDIDATES)),
    help="The candidate whose audible edits are heard (default: the judge's pick in screen.json).",
)
@click.option("--n", "count", default=SESSION_N, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--name", default=SESSION_NAME, show_default=True)
def session(candidate_id: str | None, count: int, seed: int, name: str) -> None:
    """Build the blind A/B session of the picked candidate's audible edits against the base."""
    store, manifest = layout(), load_manifest()
    screening = load_screen(store)
    judge = screening.judge
    if not judge.complete:
        raise ConfigError(
            f"the judge sample is incomplete ({judge.answered} of {judge.sample} answered); rerun `screen`, which "
            "resumes from its cache, then run `session`"
        )
    target = candidate_id or screening.pick.candidate
    if target is None:
        raise ConfigError(
            f"no candidate passed the screening ({screening.pick.reason}), so there is nothing to test and "
            "correction stays default-off; see `screen` for why each was ineligible"
        )
    # The judge's options, and so every span id, depend on exactly the candidates `screen` compared.
    screened = [CANDIDATES[c] for c in screening.screened]
    shown, key = session_for(store, manifest, screened, CANDIDATES[target], n=count, seed=seed)
    created = write_session(store, name, shown, key)
    numeric = sum(e.stratum == "numeric" for e in key.entries)
    by_language = {lang: sum(e.language == lang for e in key.entries) for lang in ("zh", "en")}
    click.echo(
        f"session {name!r} {'built' if created else 'already exists'}: {len(shown)} spans "
        f"({numeric} numeric/entity; zh {by_language['zh']}, en {by_language['en']}) "
        f"from {sum(key.population.values())} audible edits of {target}"
    )


@main.command()
@click.option("--name", default=SESSION_NAME, show_default=True)
def listen(name: str) -> None:
    """Blind A/B in the terminal: one keypress per clip (the same loop as `run_asr listen`)."""
    listen_loop(layout(), load_manifest(), name, getchar=click.getchar, echo=click.echo)


@main.command()
@click.option("--name", default=SESSION_NAME, show_default=True, help="The session to read the owner's labels from.")
def report(name: str) -> None:
    """Aggregate JSON and markdown report under $YTVERBATIM_EVAL_DIR/correction."""
    store, manifest = layout(), load_manifest()
    result = build_report(store, manifest, load_screen(store), name, now())
    if result.correction.human is not None:
        require_decided(result.correction.human)
    write_json(store.correction / REPORT_JSON, result)
    atomic.write_text(store.correction / REPORT_MD, render_markdown(result))
    click.echo(f"wrote {store.correction / REPORT_MD}")


@main.command(name="decide")
@click.option("--date", "day", default=None, help="Date for the results file name (default: today).")
def decide_command(day: str | None) -> None:
    """Apply the default-on rule: print the decision as JSON and write evals/results/correction-<date>.md."""
    path = layout().correction / REPORT_JSON
    if not path.exists():
        raise ConfigError("no report; run `report` first")
    result = Report.model_validate_json(path.read_text(encoding="utf-8"))
    decision = decide(result.correction)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written = RESULTS_DIR / f"correction-{day or date.today().isoformat()}.md"
    atomic.write_text(written, render_markdown(result) + "\n" + render_decision(decision))
    click.echo(decision.model_dump_json(indent=2))
    click.echo(f"wrote {written}", err=True)


def run() -> None:
    try:
        main(standalone_mode=False)
    except YtverbatimError as exc:
        raise SystemExit(f"error: {exc}") from None
    except click.exceptions.Abort:
        raise SystemExit(130) from None
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from None


if __name__ == "__main__":
    run()
