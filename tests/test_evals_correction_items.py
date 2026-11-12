from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from click.testing import CliRunner
from evals_fixture import CHUNKS, EN, ZH, chunk_rows, make_eval, whole_file_rows

from evals import correction, listen, run_correction
from evals.correction import (
    BASE,
    CANDIDATES,
    CandidateFile,
    RejectionSpan,
    audible_edits,
    build_base,
    complete_candidates,
    edit_spans,
    load_base,
    rejection_spans,
    session_for,
)
from evals.judge import SYSTEM, render_batch
from evals.manifest import Manifest
from evals.screen import Span, judge_items
from evals.session import KEY_FILE, SPANS_FILE, load_key, load_shown, write_session
from evals.store import Layout, write_json, write_rows
from ytverbatim.asr.models import ChunkHypothesis
from ytverbatim.contract.models import Word
from ytverbatim.errors import ConfigError
from ytverbatim.llm.protocol import LLMRequest, LLMResponse
from ytverbatim.text.correct import Rejection

HYPOTHESIS = "w-prod"
ZH_TEXTS = ("我们看这个公司的适应率。", "然后卖出了一百股。", "看NVDA的财报。", "大家好。")
EN_TEXTS = ("We bought the stock.", "Then we sold it.", "The cat sat down.", "Good night.")
SONNET = CANDIDATES["c-sonnet"]
HAIKU = CANDIDATES["c-haiku"]

SONNET_EDITS = {
    ZH: {0: "我们看这个公司的市盈率。", 1: "然后卖出了100股。"},
    EN: {1: "Then we sold them.", 3: "Good night!"},
}
HAIKU_EDITS = {
    ZH: {0: "我们看这个公司的适应力。", 2: "看AMD的财报。"},
    EN: {2: "The cap sat down."},
}


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / "state"))
    monkeypatch.setenv("YTVERBATIM_HOME", str(home / "share"))


class NoLLM:
    """Every fixture text is already punctuated, so building the base must not call a model."""

    id = "fake:punct"

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("the base needed punctuation")


def write_run(
    layout: Layout,
    candidate: correction.Candidate,
    file: str,
    replacements: dict[int, str],
    rejections: Sequence[Rejection] = (),
) -> CandidateFile:
    base = load_base(layout, file)
    texts = tuple(replacements.get(i, s.text) for i, s in enumerate(base.stretches))
    done = CandidateFile(
        file=file,
        candidate=candidate.id,
        client=candidate.client_id,
        base_digest=base.text_digest,
        versions=correction.candidate_versions(),
        texts=texts,
        edits=((),) * len(texts),
        flags=((),) * len(texts),
        rejections=tuple(rejections),
        usage=(),
        batches=0,
    )
    write_json(correction.candidate_path(layout, candidate, file), done)
    return done


def world(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Layout, Manifest]:
    layout, manifest = make_eval(root, monkeypatch)
    write_rows(
        layout,
        HYPOTHESIS,
        [*whole_file_rows(HYPOTHESIS, ZH, ZH_TEXTS), *whole_file_rows(HYPOTHESIS, EN, EN_TEXTS)],
    )
    build_base(layout, manifest, HYPOTHESIS, [], NoLLM())
    return layout, manifest


def with_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Layout, Manifest]:
    layout, manifest = world(tmp_path / "eval", monkeypatch)
    for candidate, edits in ((SONNET, SONNET_EDITS), (HAIKU, HAIKU_EDITS)):
        for file in (ZH, EN):
            write_run(layout, candidate, file, edits.get(file, {}))
    return layout, manifest


def by_id(spans: Sequence[Span]) -> dict[str, Span]:
    return {s.id: s for s in spans}


# --- edit disputes ---


def test_unchanged_stretches_give_no_disputes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = world(tmp_path / "eval", monkeypatch)
    for file in (ZH, EN):
        write_run(layout, SONNET, file, {})
    assert edit_spans(layout, manifest, [SONNET]) == []


def test_a_punctuation_only_change_gives_no_dispute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = world(tmp_path / "eval", monkeypatch)
    write_run(layout, SONNET, ZH, {3: "大家好！"})
    write_run(layout, SONNET, EN, {3: "Good night!"})
    assert edit_spans(layout, manifest, [SONNET]) == []


def test_every_dispute_reads_the_base_and_every_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = with_runs(tmp_path, monkeypatch)
    spans = by_id(edit_spans(layout, manifest, [SONNET, HAIKU]))
    assert set(spans) == {f"{ZH}#0#0", f"{ZH}#1#0", f"{ZH}#2#0", f"{EN}#1#0", f"{EN}#2#0"}
    for span in spans.values():
        assert sorted(r.hypothesis for r in span.readings) == [BASE, "c-haiku", "c-sonnet"]
    shared = spans[f"{ZH}#0#0"]
    assert [shared.reading(h).text for h in (BASE, "c-sonnet", "c-haiku")] == ["适应率", "市盈率", "适应力"]
    assert shared.language == "zh"
    english = spans[f"{EN}#1#0"]
    assert [english.reading(h).text for h in (BASE, "c-sonnet", "c-haiku")] == ["it", "them", "it"]
    assert english.left == "Then we sold"
    assert english.language == "en"


def test_a_written_form_only_edit_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = with_runs(tmp_path, monkeypatch)
    spans = by_id(edit_spans(layout, manifest, [SONNET, HAIKU]))
    assert spans[f"{ZH}#1#0"].format_only
    assert spans[f"{ZH}#1#0"].reading("c-sonnet").text == "100"
    assert not any(s.format_only for i, s in spans.items() if i != f"{ZH}#1#0")


def test_span_ids_are_stable_and_ignore_the_candidate_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = with_runs(tmp_path, monkeypatch)
    first = edit_spans(layout, manifest, [SONNET, HAIKU])
    assert edit_spans(layout, manifest, [SONNET, HAIKU]) == first
    assert [s.id for s in edit_spans(layout, manifest, [HAIKU, SONNET])] == [s.id for s in first]
    assert len({s.id for s in first}) == len(first)


def test_audible_edits_keep_only_the_candidates_own_real_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manifest = with_runs(tmp_path, monkeypatch)
    spans = edit_spans(layout, manifest, [SONNET, HAIKU])
    assert {s.id for s in audible_edits(spans, "c-sonnet")} == {f"{ZH}#0#0", f"{EN}#1#0"}
    assert {s.id for s in audible_edits(spans, "c-haiku")} == {f"{ZH}#0#0", f"{ZH}#2#0", f"{EN}#2#0"}


def test_a_candidate_without_a_run_is_refused_with_its_fix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = world(tmp_path / "eval", monkeypatch)
    with pytest.raises(ConfigError, match=r"c-sonnet has no output for zhfile00001.*correct --candidate c-sonnet"):
        edit_spans(layout, manifest, [SONNET])
    with pytest.raises(ConfigError, match="no candidate"):
        edit_spans(layout, manifest, [])


def test_word_times_come_from_the_input_hypothesis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = with_runs(tmp_path, monkeypatch)
    span = by_id(edit_spans(layout, manifest, [SONNET]))[f"{ZH}#0#0"]
    # 适应 and 市盈 are chars 8-9 of 12 words spread evenly over the 4 s chunk.
    assert (span.reading(BASE).start, span.reading(BASE).end) == pytest.approx((8 / 3, 10 / 3), abs=1e-3)
    assert (span.reading("c-sonnet").start, span.reading("c-sonnet").end) == pytest.approx((8 / 3, 10 / 3), abs=1e-3)


# --- the timing donor ---


def untimed_world(root: Path, monkeypatch: pytest.MonkeyPatch, *, donor: bool) -> tuple[Layout, Manifest]:
    layout, manifest = make_eval(root, monkeypatch)
    write_rows(
        layout,
        "q-base",
        [*chunk_rows("q-base", ZH, ZH_TEXTS), *chunk_rows("q-base", EN, EN_TEXTS)],
    )
    if donor:
        # Words packed into the first 0.1 s per character, nothing like an even spread over the chunk.
        words = [
            Word(text=ch, start=c.start + i * 0.1, end=c.start + (i + 1) * 0.1)
            for c, text in zip(CHUNKS, ZH_TEXTS, strict=True)
            for i, ch in enumerate(text.rstrip("。"))
        ]
        row = ChunkHypothesis(
            hypothesis=HYPOTHESIS, file=ZH, chunk=0, start=0.0, end=20.0, text="".join(ZH_TEXTS), words=tuple(words)
        )
        write_rows(layout, HYPOTHESIS, [row])
    build_base(layout, manifest, "q-base", [], NoLLM())
    return layout, manifest


def test_an_untimed_hypothesis_borrows_the_donors_word_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = untimed_world(tmp_path / "eval", monkeypatch, donor=True)
    assert not any(s.word_timed for s in load_base(layout, ZH).stretches)
    write_run(layout, SONNET, ZH, {0: "我们看这个公司的市盈率。"})
    write_run(layout, SONNET, EN, {})
    (span,) = edit_spans(layout, manifest, [SONNET])
    assert span.reading(BASE).text == "适应"
    assert (span.reading(BASE).start, span.reading(BASE).end) == pytest.approx((0.8, 1.0))
    assert (span.reading("c-sonnet").start, span.reading("c-sonnet").end) == pytest.approx((0.8, 1.0))


def test_an_untimed_hypothesis_without_the_donor_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = untimed_world(tmp_path / "eval", monkeypatch, donor=False)
    write_run(layout, SONNET, ZH, {0: "我们看这个公司的市盈率。"})
    write_run(layout, SONNET, EN, {})
    with pytest.raises(ConfigError, match=r"no word timings.*run_asr asr --hyp w-prod"):
        edit_spans(layout, manifest, [SONNET])


def test_a_file_without_edits_needs_no_donor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = untimed_world(tmp_path / "eval", monkeypatch, donor=False)
    for file in (ZH, EN):
        write_run(layout, SONNET, file, {})
    assert edit_spans(layout, manifest, [SONNET]) == []


# --- judge items ---


def test_the_judge_never_sees_a_candidate_a_base_or_a_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = with_runs(tmp_path, monkeypatch)
    spans = edit_spans(layout, manifest, [SONNET, HAIKU])
    items = judge_items(spans, 100, 0)
    assert len(items) == len(spans) - 1  # the written-form-only dispute is not judged
    prompt = (SYSTEM + render_batch(items)).lower()
    for hidden in ("base", "c-sonnet", "c-haiku", "sonnet", "haiku", "claude", "mlx", "qwen", "opus"):
        assert hidden not in prompt
    options = {item.span_id: set(item.options) for item in items}
    shared = by_id(spans)[f"{ZH}#0#0"]
    assert options[shared.id] == {"适应率", "市盈率", "适应力"}
    assert options[f"{EN}#2#0"] == {"cat", "cap"}
    assert all(len(item.options) == len(set(item.options)) for item in items)


# --- rejected proposals ---


def rejected(segment: int, before: str, proposed: str, reason: str) -> Rejection:
    return Rejection(segment, "first", before, proposed, reason)  # type: ignore[arg-type]


def with_rejections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Layout, Manifest]:
    layout, manifest = world(tmp_path / "eval", monkeypatch)
    digits = rejected(0, ZH_TEXTS[0], "我们看这个公司的市盈率。", "digits")
    negation = rejected(1, ZH_TEXTS[1], "然后卖出了两百股。", "negation")
    symbols = rejected(2, ZH_TEXTS[2], "看AMD的财报。", "symbols")
    unaudible = rejected(3, ZH_TEXTS[3], "大家好！", "empty")
    bad_index = Rejection(9, "first", "", "x", "index")
    write_run(layout, SONNET, ZH, {}, [*[digits] * 40, *[negation] * 5, symbols, unaudible, bad_index])
    write_run(layout, SONNET, EN, {})
    write_run(layout, HAIKU, ZH, {}, [*[digits] * 10, *[symbols] * 2])
    write_run(layout, HAIKU, EN, {})
    return layout, manifest


def proposals_by_guard(spans: Sequence[RejectionSpan]) -> dict[str, set[tuple[str, int]]]:
    out: dict[str, set[tuple[str, int]]] = {}
    for s in spans:
        out.setdefault(s.guard, set()).add((s.candidate, s.rejection))
    return out


def test_the_rejection_sample_is_capped_per_guard_and_skips_what_cannot_be_judged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manifest = with_rejections(tmp_path, monkeypatch)
    spans = rejection_spans(layout, manifest, [SONNET, HAIKU])
    sizes = {guard: len(found) for guard, found in proposals_by_guard(spans).items()}
    assert sizes == {"digits": 30, "negation": 5, "symbols": 3}
    assert len({s.span.id for s in spans}) == len(spans)
    digits = next(s for s in spans if s.guard == "digits")
    assert [r.hypothesis for r in digits.span.readings] == [correction.BEFORE, correction.PROPOSED]
    assert [r.text for r in digits.span.readings] == ["适应", "市盈"]
    assert digits.span.id.startswith(f"{digits.candidate}:{ZH}#")
    assert not any(s.span.format_only for s in spans)


def test_the_rejection_sample_is_seeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = with_rejections(tmp_path, monkeypatch)
    pair = [SONNET, HAIKU]
    first = rejection_spans(layout, manifest, pair, seed=3)
    assert rejection_spans(layout, manifest, pair, seed=3) == first
    other = rejection_spans(layout, manifest, pair, seed=4)
    assert proposals_by_guard(other)["digits"] != proposals_by_guard(first)["digits"]
    assert proposals_by_guard(other)["negation"] == proposals_by_guard(first)["negation"]  # fewer than the cap: all


def test_the_total_cap_shrinks_the_per_guard_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = with_rejections(tmp_path, monkeypatch)
    spans = rejection_spans(layout, manifest, [SONNET, HAIKU], total=20)
    sizes = {guard: len(found) for guard, found in proposals_by_guard(spans).items()}
    assert sizes == {"digits": 12, "negation": 5, "symbols": 3}
    assert sum(sizes.values()) <= 20
    assert len(rejection_spans(layout, manifest, [SONNET], per_guard=2)) == 2 + 2 + 1


def test_no_rejections_gives_an_empty_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = with_runs(tmp_path, monkeypatch)
    assert rejection_spans(layout, manifest, [SONNET, HAIKU]) == []


# --- the session ---


def test_the_session_holds_only_the_picked_candidates_audible_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manifest = with_runs(tmp_path, monkeypatch)
    shown, key = session_for(layout, manifest, [SONNET, HAIKU], SONNET, seed=5)
    assert (key.control, key.challenger, key.seed, key.n) == (BASE, "c-sonnet", 5, correction.SESSION_N)
    assert {e.source for e in key.entries} == {f"{ZH}#0#0", f"{EN}#1#0"}
    assert sum(key.population.values()) == 2
    for span, entry in zip(shown, key.entries, strict=True):
        readings = {f"{ZH}#0#0": ("适应率", "市盈率"), f"{EN}#1#0": ("it", "them")}[entry.source]
        other = 2 if entry.challenger_option == 1 else 1
        assert span.options[entry.challenger_option - 1] == readings[1]
        assert span.options[other - 1] == readings[0]
    assert all(span.clip_end > span.clip_start for span in shown)
    for needle in ("c-haiku", "c-sonnet", BASE):
        assert needle not in "".join(s.model_dump_json() for s in shown)


def test_the_sides_are_seeded_and_both_occur(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = with_runs(tmp_path, monkeypatch)
    again = session_for(layout, manifest, [SONNET, HAIKU], SONNET, seed=5)
    same = session_for(layout, manifest, [SONNET, HAIKU], SONNET, seed=5)
    assert again[0] == same[0]
    assert [e.challenger_option for e in again[1].entries] == [e.challenger_option for e in same[1].entries]
    sides = {
        e.challenger_option
        for seed in range(12)
        for e in session_for(layout, manifest, [SONNET, HAIKU], SONNET, seed=seed)[1].entries
    }
    assert sides == {1, 2}


def test_a_rewrite_is_idempotent_and_a_different_session_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manifest = with_runs(tmp_path, monkeypatch)
    pair = [SONNET, HAIKU]
    assert write_session(layout, "correction", *session_for(layout, manifest, pair, SONNET, seed=1))
    assert not write_session(layout, "correction", *session_for(layout, manifest, pair, SONNET, seed=1))
    with pytest.raises(ConfigError, match="already exists with different spans"):
        write_session(layout, "correction", *session_for(layout, manifest, pair, HAIKU, seed=1))


def test_the_target_must_be_one_of_the_compared_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = with_runs(tmp_path, monkeypatch)
    with pytest.raises(ConfigError, match="not among the candidates"):
        session_for(layout, manifest, [SONNET], HAIKU)


def test_a_candidate_with_no_audible_edit_has_no_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = world(tmp_path / "eval", monkeypatch)
    for file in (ZH, EN):
        write_run(layout, SONNET, file, {})
        write_run(layout, HAIKU, file, HAIKU_EDITS.get(file, {}))
    with pytest.raises(ConfigError, match="no audible disputes between base and c-sonnet"):
        session_for(layout, manifest, [SONNET, HAIKU], SONNET)


# --- the commands ---


def test_complete_candidates_lists_only_finished_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = world(tmp_path / "eval", monkeypatch)
    assert complete_candidates(layout, manifest) == []
    write_run(layout, SONNET, ZH, {})
    assert complete_candidates(layout, manifest) == []
    write_run(layout, SONNET, EN, {})
    assert complete_candidates(layout, manifest) == [SONNET]


def test_the_session_command_builds_the_session_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, _ = with_runs(tmp_path, monkeypatch)
    runner = CliRunner()
    built = runner.invoke(run_correction.main, ["session", "--candidate", "c-sonnet", "--seed", "2"])
    assert built.exit_code == 0, built.output
    assert "session 'correction' built: 2 spans" in built.output
    assert "from 2 audible edits of c-sonnet" in built.output
    directory = layout.session_dir("correction")
    assert (directory / SPANS_FILE).exists()
    assert load_key(directory).challenger == "c-sonnet"
    assert len(load_shown(directory)) == 2

    again = runner.invoke(run_correction.main, ["session", "--candidate", "c-sonnet", "--seed", "2"])
    assert "already exists" in again.output
    other = runner.invoke(run_correction.main, ["session", "--candidate", "c-haiku", "--seed", "2"])
    assert other.exit_code != 0
    assert isinstance(other.exception, ConfigError)
    assert "different spans" in str(other.exception)
    named = runner.invoke(run_correction.main, ["session", "--candidate", "c-haiku", "--name", "second"])
    assert named.exit_code == 0, named.output
    assert (layout.session_dir("second") / KEY_FILE).exists()


def test_the_session_command_needs_a_finished_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world(tmp_path / "eval", monkeypatch)
    result = CliRunner().invoke(run_correction.main, ["session", "--candidate", "c-sonnet"])
    assert isinstance(result.exception, ConfigError)
    assert "run it first" in str(result.exception)


class Keys:
    def __init__(self, keys: str) -> None:
        self._keys: Iterator[str] = iter(keys)
        self.played: list[Path] = []
        self.cut: list[tuple[float, float]] = []
        self.output: list[str] = []

    def getchar(self) -> str:
        return next(self._keys)

    def player(self, clip: Path) -> None:
        assert clip.exists()
        self.played.append(clip)

    def cutter(self, wav: Path, start: float, end: float, out: Path) -> None:
        assert wav.exists()
        self.cut.append((start, end))
        out.write_bytes(b"RIFF")


def test_the_listen_loop_runs_on_the_correction_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = with_runs(tmp_path, monkeypatch)
    write_session(layout, "correction", *session_for(layout, manifest, [SONNET, HAIKU], SONNET, seed=1))
    keys = Keys("1r2")
    result = listen.listen(
        layout,
        manifest,
        correction.SESSION_NAME,
        getchar=keys.getchar,
        echo=keys.output.append,
        player=keys.player,
        cutter=keys.cutter,
    )
    assert (result.decided, result.remaining, result.quit) == (2, 0, False)
    assert len(keys.played) == 3  # two clips and a replay
    assert [d.choice for d in listen.read_decisions(layout.session_dir("correction"))] == ["1", "2"]
    assert "⟪ ? ⟫" in keys.output[1]
    assert "session 'correction' complete: 2 spans" in keys.output[-1]


def test_the_listen_command_defaults_to_the_correction_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = world(tmp_path / "eval", monkeypatch)
    seen: list[tuple[Layout, Manifest, str]] = []

    def fake(store: Layout, loaded: Manifest, name: str, **_: object) -> None:
        seen.append((store, loaded, name))

    monkeypatch.setattr(run_correction, "listen_loop", fake)
    runner = CliRunner()
    assert runner.invoke(run_correction.main, ["listen"]).exit_code == 0
    assert runner.invoke(run_correction.main, ["listen", "--name", "second"]).exit_code == 0
    assert [(s.root, m, n) for s, m, n in seen] == [
        (layout.root, manifest, "correction"),
        (layout.root, manifest, "second"),
    ]
