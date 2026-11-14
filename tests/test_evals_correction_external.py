import importlib
import json
import sys
import textwrap
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from evals_fixture import EN, ZH, make_eval, whole_file_rows

from evals import correction, run_correction
from evals.correction import CANDIDATES, build_base, load_base, load_candidate_file
from evals.correction_report import resolve_candidates
from evals.manifest import Manifest
from evals.run_correction import CLIENT_FACTORY_ENV, external_client
from evals.store import Layout, write_rows
from ytverbatim.errors import ConfigError
from ytverbatim.llm import ClaudeCLIClient
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Usage

HYPOTHESIS = "w-prod"
MEDIUM = CANDIDATES["x-gpt61-medium"]
HIGH = CANDIDATES["x-gpt61-high"]

FACTORY_MODULE = """
import json

calls = []


class Client:
    id = "fake-transport:whatever"

    def complete(self, request):
        from ytverbatim.llm.protocol import LLMResponse, Usage

        body = request.user.split("SEGMENTS:\\n", 1)[1]
        items, _ = json.JSONDecoder().raw_decode(body)
        if "proofreader" in request.system:
            text = json.dumps({"fixed": []})
        else:
            text = json.dumps({"items": [{"i": i["i"], "text": i["text"] + "."} for i in items]})
        return LLMResponse(text=text, usage=Usage(10, 5), elapsed_s=0.1)


def make(*, model, effort):
    calls.append((model, effort))
    return Client()


def not_a_client(*, model, effort):
    return object()


def no_id(*, model, effort):
    class Half:
        def complete(self, request):
            raise AssertionError

    return Half()


not_callable = 3
"""


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / "state"))
    monkeypatch.setenv("YTVERBATIM_HOME", str(home / "share"))
    monkeypatch.delenv(CLIENT_FACTORY_ENV, raising=False)


@pytest.fixture
def factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Any]:
    modules = tmp_path / "factory-modules"
    modules.mkdir()
    (modules / "fake_eval_factory.py").write_text(textwrap.dedent(FACTORY_MODULE), encoding="utf-8")
    monkeypatch.syspath_prepend(str(modules))
    module = importlib.import_module("fake_eval_factory")
    yield module
    sys.modules.pop("fake_eval_factory", None)


def eval_with_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Layout, Manifest]:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    write_rows(
        layout,
        HYPOTHESIS,
        [
            *whole_file_rows(HYPOTHESIS, ZH, ("我们看这个公司", "大家好")),
            *whole_file_rows(HYPOTHESIS, EN, ("we bought the stock", "good night")),
        ],
    )

    class Punctuator:
        id = "fake:punct"

        def complete(self, request: LLMRequest) -> LLMResponse:
            items, _ = json.JSONDecoder().raw_decode(request.user.split("SEGMENTS:\n", 1)[1])
            reply = {"items": [{"i": i["i"], "text": i["text"] + "."} for i in items]}
            return LLMResponse(text=json.dumps(reply), usage=Usage(1, 1), elapsed_s=0.0)

    build_base(layout, manifest, HYPOTHESIS, [], Punctuator())
    return layout, manifest


def test_the_factory_is_called_with_the_candidates_model_and_effort(
    factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CLIENT_FACTORY_ENV, "fake_eval_factory:make")
    client = external_client(HIGH)
    assert client.id == "fake-transport:whatever"
    external_client(MEDIUM)
    assert factory.calls == [("gpt-6.1-sol", "high"), ("gpt-6.1-sol", "medium")]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "is not set; set YTVERBATIM_EVAL_CLIENT_FACTORY=package.module:function"),
        ("", "is not set"),
        ("fake_eval_factory", "is not package.module:function"),
        ("fake_eval_factory:", "is not package.module:function"),
        (":make", "is not package.module:function"),
        ("no_such_module_anywhere:make", "cannot import 'no_such_module_anywhere'"),
        ("fake_eval_factory:missing", "no callable 'missing'"),
        ("fake_eval_factory:not_callable", "no callable 'not_callable'"),
    ],
)
@pytest.mark.usefixtures("factory")
def test_a_missing_or_unusable_variable_fails_loud_with_how_to_set_it(
    monkeypatch: pytest.MonkeyPatch, value: str | None, message: str
) -> None:
    if value is not None:
        monkeypatch.setenv(CLIENT_FACTORY_ENV, value)
    with pytest.raises(ConfigError, match=message) as raised:
        external_client(MEDIUM)
    assert "package.module:function" in str(raised.value)


@pytest.mark.parametrize("name", ["not_a_client", "no_id"])
@pytest.mark.usefixtures("factory")
def test_an_object_that_is_not_an_llm_client_is_refused(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(CLIENT_FACTORY_ENV, f"fake_eval_factory:{name}")
    with pytest.raises(ConfigError, match="not an LLMClient"):
        external_client(MEDIUM)


def test_an_external_candidate_never_falls_back_to_the_claude_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _ = eval_with_base(tmp_path, monkeypatch)
    monkeypatch.setattr(run_correction, "ClaudeCLIClient", lambda **_: pytest.fail("fell back to the Claude CLI"))
    result = CliRunner().invoke(run_correction.main, ["correct", "--candidate", "x-gpt61-medium"])
    assert isinstance(result.exception, ConfigError)
    assert "YTVERBATIM_EVAL_CLIENT_FACTORY" in str(result.exception)
    assert not (correction.run_dir(layout, MEDIUM) / "calls.jsonl").exists()


def test_a_run_records_the_model_and_effort_and_each_effort_is_its_own_run(
    factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manifest = eval_with_base(tmp_path, monkeypatch)
    monkeypatch.setenv(CLIENT_FACTORY_ENV, "fake_eval_factory:make")
    runner = CliRunner()
    for candidate in ("x-gpt61-medium", "x-gpt61-high"):
        result = runner.invoke(run_correction.main, ["correct", "--candidate", candidate])
        assert result.exit_code == 0, result.output
        assert f"{candidate}: 2 files corrected" in result.output

    done = load_candidate_file(layout, MEDIUM, ZH)
    assert done is not None
    assert done.client == "external:gpt-6.1-sol@medium"
    assert {u.client for u in done.usage} == {"external:gpt-6.1-sol@medium"}
    other = load_candidate_file(layout, HIGH, ZH)
    assert other is not None
    assert other.client == "external:gpt-6.1-sol@high"
    assert correction.run_dir(layout, MEDIUM) != correction.run_dir(layout, HIGH)
    assert [m.id for m in manifest.videos] == [ZH, EN]
    assert factory.calls == [("gpt-6.1-sol", "medium"), ("gpt-6.1-sol", "high")]

    again = runner.invoke(run_correction.main, ["correct", "--candidate", "x-gpt61-high"])
    assert "complete for all selected files" in again.output
    assert len(factory.calls) == 2

    # Moving the label of a finished run to another effort refuses instead of mixing.
    stored = correction.candidate_path(layout, MEDIUM, ZH)
    payload = json.loads(stored.read_text(encoding="utf-8"))
    payload["client"] = "external:gpt-6.1-sol@high"
    stored.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"made with client 'external:gpt-6\.1-sol@high'"):
        load_candidate_file(layout, MEDIUM, ZH)


@pytest.mark.usefixtures("factory")
def test_screen_by_default_includes_every_finished_external_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manifest = eval_with_base(tmp_path, monkeypatch)
    monkeypatch.setenv(CLIENT_FACTORY_ENV, "fake_eval_factory:make")
    assert CliRunner().invoke(run_correction.main, ["correct", "--candidate", "x-gpt61-high"]).exit_code == 0
    screened, unfinished = resolve_candidates(layout, manifest, [])
    assert screened == [HIGH]
    assert unfinished == []
    assert load_base(layout, ZH).versions == correction.base_versions()


def test_external_candidates_are_not_local_and_have_no_weights_to_clean_up() -> None:
    for candidate in (MEDIUM, HIGH):
        assert candidate.external
        assert not candidate.local
        assert candidate.model == "gpt-6.1-sol"
        with pytest.raises(ConfigError, match="no local weights"):
            _ = candidate.weights
    assert (MEDIUM.effort, HIGH.effort) == ("medium", "high")
    assert MEDIUM.client_id != HIGH.client_id


def test_existing_candidates_keep_their_run_identity() -> None:
    sonnet, haiku, qwen35, qwen3 = (CANDIDATES[c] for c in ("c-sonnet", "c-haiku", "l-qwen35-9b", "l-qwen3-8b"))
    assert sonnet.client_id == "claude-cli:claude-sonnet-5-5" == ClaudeCLIClient(model=sonnet.model).id
    assert haiku.client_id == "claude-cli:claude-haiku-5-5"
    assert qwen35.client_id == "mlx:mlx-community/Qwen3.5-9B-8bit@16daa4818c54"
    assert qwen3.client_id == "mlx:mlx-community/Qwen3-8B-8bit@48a0b75b1ae7"
    assert not any(c.external for c in (sonnet, haiku, qwen35, qwen3))
    assert [c.local for c in (sonnet, haiku, qwen35, qwen3)] == [False, False, True, True]
    assert correction.candidate_versions() == {
        "stages": correction.stages.VERSION,
        "clean": correction.clean.VERSION,
        "punctuate": correction.punctuate_module.PROMPT_VERSION,
        "correct": correction.correct_module.PROMPT_VERSION,
    }
