"""The correction eval end to end through its CLI with fake LLMs, from `base` to `decide`."""

import contextlib
import json
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result
from evals_fixture import EN, ZH, FakeJudge, chunk_rows, make_eval, whole_file_rows

from evals import correction, listen, run_correction
from evals.correction import CANDIDATES, Failure
from evals.correction_report import SCREEN_FILE, CorrectionScreen
from evals.session import ShownSpan, load_shown
from evals.store import REFERENCE, Layout, write_json, write_rows
from ytverbatim.errors import ConfigError
from ytverbatim.llm import claude_cli
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Readiness, Usage

HYPOTHESIS = "w-prod"
ZH_TEXTS = ("我们看这个公司的适应率", "然后卖出了一百股", "看NVDA的财报", "大家好")
EN_TEXTS = ("we bought the stock", "then we sold it", "the cat sat down", "good night")
# What the judge prefers and what the owner hears: the readings that were actually said.
SAID = ("市盈率", "市盈", "them", "cat")
PRIVATE = (
    ZH,
    EN,
    "Example title",
    "Example NVDA talk",
    "Example Channel",
    "适应率",
    "市盈率",
    "适应力",
    "一百股",
    "二百股",
    "we bought",
    "then we sold",
    "the cat sat",
    "good night",
    "NVDA",
)
CHECKS = {
    "eligible candidate",
    "every span decided",
    "over-correction",
    "errors per hour",
    "number preservation",
    "JSON after the retry",
}


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / "state"))
    monkeypatch.setenv("YTVERBATIM_HOME", str(home / "share"))


def segments(request: LLMRequest) -> list[dict[str, Any]]:
    items, _ = json.JSONDecoder().raw_decode(request.user.split("SEGMENTS:\n", 1)[1])
    return items


@dataclass
class Punctuator:
    """Closes every segment with a full stop."""

    id: str = "fake:punct"
    requests: list[LLMRequest] = field(default_factory=list[LLMRequest])

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        items = segments(request)
        stop = "。" if any("一" <= ch <= "鿿" for item in items for ch in item["text"]) else "."
        reply = {"items": [{"i": i["i"], "text": i["text"] + stop} for i in items]}
        return LLMResponse(text=json.dumps(reply, ensure_ascii=False), usage=Usage(50, 10), elapsed_s=0.1)


@dataclass
class Corrector:
    """Proofreads by replacing ``fixes`` wherever they occur, and returns only the segments it changed."""

    id: str
    fixes: Mapping[str, str]
    requests: list[LLMRequest] = field(default_factory=list[LLMRequest])

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        fixed = []
        for item in segments(request):
            text = item["text"]
            for old, new in self.fixes.items():
                text = text.replace(old, new)
            if text != item["text"]:
                fixed.append({"i": item["i"], "text": text})
        return LLMResponse(text=json.dumps({"fixed": fixed}, ensure_ascii=False), usage=Usage(100, 20), elapsed_s=0.25)


@dataclass
class Rig:
    layout: Layout
    results: Path
    punct: Punctuator
    correctors: dict[str, Corrector]
    judge: FakeJudge
    runner: CliRunner = field(default_factory=CliRunner)
    quit_listening: bool = False

    @property
    def llm_calls(self) -> int:
        clients = [self.punct, self.judge, *self.correctors.values()]
        return sum(len(c.requests) for c in clients)

    def ok(self, *args: str) -> Result:
        result = self.runner.invoke(run_correction.main, list(args))
        assert result.exit_code == 0, result.output
        return result

    def refused(self, *args: str) -> str:
        result = self.runner.invoke(run_correction.main, list(args))
        assert isinstance(result.exception, ConfigError), result.output
        return str(result.exception)

    def correct_all(self) -> None:
        for candidate in ("c-sonnet", "c-haiku", "l-qwen35-9b"):
            self.ok("correct", "--candidate", candidate)

    def presses(self, shown: list[ShownSpan]) -> str:
        if self.quit_listening:
            return "q"
        return "".join("1" if s.options[0] in SAID else "2" if s.options[1] in SAID else "b" for s in shown)


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    layout, _ = make_eval(tmp_path / "eval", monkeypatch)
    write_rows(
        layout,
        HYPOTHESIS,
        [*whole_file_rows(HYPOTHESIS, ZH, ZH_TEXTS), *whole_file_rows(HYPOTHESIS, EN, EN_TEXTS)],
    )
    manual = ("我们看这个公司的市盈率", "然后卖出了一百股", "看NVDA的财报", "大家好")
    write_rows(layout, REFERENCE, chunk_rows(REFERENCE, ZH, manual))
    correctors = {
        "c-sonnet": Corrector("claude-cli:claude-sonnet-5-5", {"适应率": "市盈率", "sold it": "sold them"}),
        "c-haiku": Corrector("claude-cli:claude-haiku-5-5", {"适应率": "适应力", "cat sat": "cap sat"}),
        # The second edit changes a number, so the production guards refuse it.
        "l-qwen35-9b": Corrector("mlx:fake@0123456789ab", {"适应率": "市盈率", "一百股": "二百股"}),
    }
    made = Rig(layout, tmp_path / "results", Punctuator(), correctors, FakeJudge(prefer=SAID))
    by_model = {c.model: made.correctors[c.id] for c in CANDIDATES.values() if c.id in made.correctors}

    @contextlib.contextmanager
    def server(candidate: correction.Candidate, *_: object, **__: object) -> Generator[Corrector]:
        yield made.correctors[candidate.id]

    def fake_listen(store: Layout, manifest: Any, name: str, *, getchar: Callable[[], str], echo: Any) -> None:
        pressed = iter(made.presses(load_shown(store.session_dir(name))))
        listen.listen(
            store,
            manifest,
            name,
            getchar=lambda: next(pressed),
            echo=echo,
            player=lambda clip: None,
            cutter=lambda wav, start, end, out: out.write_bytes(b"RIFF"),
        )

    monkeypatch.setattr(run_correction, "punctuation_client", lambda spec: made.punct)
    monkeypatch.setattr(claude_cli, "readiness", lambda: Readiness(True, "fake"))
    monkeypatch.setattr(run_correction, "ClaudeCLIClient", lambda *, model: by_model[model])
    monkeypatch.setattr(run_correction, "local_server", server)
    monkeypatch.setattr(run_correction, "judge_client", lambda: made.judge)
    monkeypatch.setattr(run_correction, "listen_loop", fake_listen)
    monkeypatch.setattr(run_correction, "RESULTS_DIR", made.results)
    return made


def screen_json(layout: Layout) -> CorrectionScreen:
    return CorrectionScreen.model_validate_json((layout.correction / SCREEN_FILE).read_text(encoding="utf-8"))


def test_the_whole_flow_picks_decides_and_publishes_only_generic_labels(rig: Rig) -> None:
    rig.ok("base", "--hyp", HYPOTHESIS)
    rig.correct_all()

    for stopped in (("--max-calls", "0"), ("--no-judge",)):
        rig.ok("screen", *stopped)
        assert not screen_json(rig.layout).judge.complete
        assert "judge sample is incomplete" in rig.refused("session")
        assert "judge sample is incomplete" in rig.refused("report")
        assert rig.judge.requests == []

    screened = rig.ok("screen").output
    assert "Judge pick: **c-sonnet**" in screened
    assert "c-haiku is ineligible" in screened
    assert "c-haiku" in screened
    screen = screen_json(rig.layout)
    assert screen.judge.complete
    assert screen.screened == ("c-sonnet", "c-haiku", "l-qwen35-9b")
    assert screen.pick.candidate == "c-sonnet"
    assert {g.guard for g in screen.guards} == {"cjk_numerals"}
    assert all(g.informational for g in screen.guards)

    built = rig.ok("session").output
    assert "from 2 audible edits of c-sonnet" in built
    rig.ok("listen")
    rig.ok("report")
    decided = rig.ok("decide", "--date", "2026-11-12")
    decision = json.loads(decided.stdout)
    assert decision["default_on"] is True
    assert decision["candidate"] == "c-sonnet"
    assert {c["name"] for c in decision["checks"]} == CHECKS
    assert all(c["passed"] for c in decision["checks"])

    reference = {
        r["candidate"]: r for r in json.loads((rig.layout.correction / "report.json").read_text())["reference"]
    }
    assert [reference["c-sonnet"]["cer"][k] for k in ("before", "after")] == [2, 0]
    assert reference["c-haiku"]["cer"]["after"] == 3

    published = (rig.results / "correction-2026-11-12.md").read_text(encoding="utf-8")
    assert "correction ships default-on with c-sonnet" in published
    assert "zh-01" in published
    assert "en-01" in published
    for check in CHECKS:
        assert check in published
    for private in (*PRIVATE, str(rig.layout.root), "/Users/"):
        assert private not in published
    for name in ("report.json", "report.md"):
        assert (rig.layout.correction / name).exists()

    calls = rig.llm_calls
    assert calls > 0
    for args in (
        ("base", "--hyp", HYPOTHESIS),
        ("correct", "--candidate", "c-sonnet"),
        ("correct", "--candidate", "c-haiku"),
        ("correct", "--candidate", "l-qwen35-9b"),
        ("screen",),
        ("session",),
        ("listen",),
        ("report",),
        ("decide", "--date", "2026-11-12"),
    ):
        rig.ok(*args)
    assert rig.llm_calls == calls


def test_a_session_waits_for_the_judge_and_a_pick(rig: Rig) -> None:
    rig.ok("base", "--hyp", HYPOTHESIS)
    rig.correct_all()
    assert "run `screen` first" in rig.refused("session")
    rig.ok("screen", "--candidates", "c-haiku")
    assert screen_json(rig.layout).screened == ("c-haiku",)
    assert "no candidate passed the screening" in rig.refused("session")


def test_the_session_uses_exactly_the_candidates_screen_compared(rig: Rig) -> None:
    rig.ok("base", "--hyp", HYPOTHESIS)
    rig.correct_all()
    rig.ok("screen", "--candidates", "c-sonnet,l-qwen35-9b")
    assert rig.ok("session").output.count("c-sonnet") == 1
    shown = load_shown(rig.layout.session_dir("correction"))
    # Without haiku's 适应力 the disputed stretch is narrower, so these span ids and options exist only for this set.
    assert {o for s in shown for o in s.options} == {"市盈", "适应", "them", "it"}
    assert (
        rig.refused("session", "--candidate", "c-haiku")
        == "c-haiku is not among the candidates whose edits were compared"
    )


def test_a_stale_failure_is_named_with_its_fix(rig: Rig) -> None:
    rig.ok("base", "--hyp", HYPOTHESIS)
    rig.correct_all()
    haiku = CANDIDATES["c-haiku"]
    write_json(
        correction.run_dir(rig.layout, haiku) / correction.FAILURE_FILE,
        Failure(file=ZH, error="invalid JSON twice", at="2026-11-12T00:00:00+00:00"),
    )
    output = rig.ok("screen").output
    assert "c-haiku is ineligible: a correction call returned invalid JSON twice" in output
    assert "move correction/runs/c-haiku" in output
    assert "from scratch" in output
    rig.ok("session")
    rig.ok("listen")
    published = rig.ok("report")
    assert "from scratch" in (rig.layout.correction / "report.md").read_text(encoding="utf-8")
    assert ZH not in (rig.layout.correction / "report.md").read_text(encoding="utf-8")
    assert published.exit_code == 0


def test_report_and_decide_refuse_while_a_span_is_undecided(rig: Rig) -> None:
    rig.ok("base", "--hyp", HYPOTHESIS)
    rig.correct_all()
    rig.ok("screen")
    rig.ok("session")
    rig.quit_listening = True
    rig.ok("listen")
    assert "without a decision" in rig.refused("report")
    assert "no report" in rig.refused("decide")


def test_without_a_session_the_decision_is_default_off(rig: Rig) -> None:
    rig.ok("base", "--hyp", HYPOTHESIS)
    rig.correct_all()
    rig.ok("screen")
    rig.ok("report")
    decision = json.loads(rig.ok("decide", "--date", "2026-11-12").stdout)
    assert decision["default_on"] is False
    assert [c["name"] for c in decision["checks"] if not c["passed"]] == ["human session"]
