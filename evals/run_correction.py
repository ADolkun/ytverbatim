"""Correction-LLM eval runner: ``uv run python -m evals.run_correction <command>``.

Order: ``base --hyp H`` → ``correct --candidate C`` per candidate (then ``cleanup --candidate C`` for a local one) →
``session --candidate C`` → ``listen``. Every command is resumable and writes only under ``$YTVERBATIM_EVAL_DIR``
(``correction/``, and the session under ``ab/``).
"""

import contextlib
from contextlib import AbstractContextManager
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
    complete_candidates,
    first_try_validity,
    load_failure,
    local_server,
    pending_files,
    run_candidate,
    run_dir,
    session_for,
)
from evals.listen import listen as listen_loop
from evals.manifest import load_manifest
from evals.run_asr import default_hf_cache, layout
from evals.session import write_session
from evals.store import Layout
from ytverbatim import config
from ytverbatim.asr import registry
from ytverbatim.asr.runner import memory_cap_bytes
from ytverbatim.errors import ConfigError, YtverbatimError
from ytverbatim.llm import ClaudeCLIClient, apply_spec, claude_cli, client_from_options
from ytverbatim.llm.protocol import LLMClient, Readiness

DEFAULT_PUNCTUATION_LLM = "claude-cli:claude-sonnet-5-5"
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
        click.echo(
            f"note: {run_dir(store, candidate) / 'failure.json'} records an earlier invalid-JSON failure; the "
            "candidate stays ineligible until that file is deleted"
        )


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
@click.option(
    "--candidate",
    "candidate_id",
    required=True,
    type=click.Choice(sorted(CANDIDATES)),
    help="The candidate whose audible edits are heard.",
)
@click.option("--n", "count", default=SESSION_N, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--name", default=SESSION_NAME, show_default=True)
def session(candidate_id: str, count: int, seed: int, name: str) -> None:
    """Build the blind A/B session of one candidate's audible edits against the base."""
    store, manifest = layout(), load_manifest()
    candidates = complete_candidates(store, manifest)
    if not candidates:
        raise ConfigError("no candidate has finished `correct` for every file; run it first")
    shown, key = session_for(store, manifest, candidates, CANDIDATES[candidate_id], n=count, seed=seed)
    created = write_session(store, name, shown, key)
    numeric = sum(e.stratum == "numeric" for e in key.entries)
    by_language = {lang: sum(e.language == lang for e in key.entries) for lang in ("zh", "en")}
    click.echo(
        f"session {name!r} {'built' if created else 'already exists'}: {len(shown)} spans "
        f"({numeric} numeric/entity; zh {by_language['zh']}, en {by_language['en']}) "
        f"from {sum(key.population.values())} audible edits of {candidate_id}"
    )


@main.command()
@click.option("--name", default=SESSION_NAME, show_default=True)
def listen(name: str) -> None:
    """Blind A/B in the terminal: one keypress per clip (the same loop as `run_asr listen`)."""
    listen_loop(layout(), load_manifest(), name, getchar=click.getchar, echo=click.echo)


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
