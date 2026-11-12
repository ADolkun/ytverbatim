import json
import re
from pathlib import Path

import pytest
from evals_fixture import ZH, chunk_rows, make_eval

from evals import judge
from evals.consensus import UNCERTAIN
from evals.screen import collect_spans, judge_items, judge_shares
from evals.store import write_rows
from ytverbatim.asr import registry
from ytverbatim.errors import LLMOutputError
from ytverbatim.llm import LLMRequest, LLMResponse

HYPOTHESES = ("w-vad", "q-base", "f-aed")
TEXTS = {
    # Chunk 0 has two disputes four agreed units apart; each one's context must hide the other's wording.
    "w-vad": ("我们今天买了这只股票然后卖出", "大家好", "", ""),
    "q-base": ("我们明天买了这只股票然后卖掉", "大家好", "", ""),
    "f-aed": ("我们今天买了这只股票然后卖掉", "大家好", "", ""),
}


class FakeClient:
    def __init__(self, answer: str = "first") -> None:
        self.requests: list[LLMRequest] = []
        self.answer = answer

    @property
    def id(self) -> str:
        return "fake:judge"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        labels = re.findall(r"^\[(s\d+)\]", request.user, flags=re.MULTILINE)
        choice: object = 1 if self.answer == "first" else judge.NEEDS_AUDIO
        decisions = [{"span": label, "choice": choice, "reason": "fits"} for label in labels]
        return LLMResponse(json.dumps({"decisions": decisions}), None, 0.1)


@pytest.fixture
def spans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    for name, texts in TEXTS.items():
        write_rows(layout, name, chunk_rows(name, ZH, texts))
    return collect_spans(layout, manifest, HYPOTHESES)


def test_the_prompt_hides_engines_and_agreement_counts(spans: list, tmp_path: Path) -> None:
    assert len(spans) == 2
    items = judge_items(spans, 10, seed=3)
    client = FakeClient()
    judge.run_judge(client, items, tmp_path / "judge", model="m")
    (request,) = client.requests
    rendered = request.system + "\n" + request.user
    names = {*registry.HYPOTHESES, "yt-auto", "whisper", "qwen", "firered", "mimo"}
    assert not any(name.casefold() in rendered.casefold() for name in names)
    assert not re.search(r"\b\d+\s*(?:of|/)\s*\d+\b|\b(?:engines?|votes?|agree)", request.user)
    # Two engines wrote 今 and two wrote 掉, yet each reading is listed once, so frequency is invisible.
    assert len(re.findall(r"^  \d\. 今$", request.user, flags=re.MULTILINE)) == 1
    assert len(re.findall(r"^  \d\. 掉$", request.user, flags=re.MULTILINE)) == 1
    for item in items:
        assert len(item.options) == len(set(item.options)) == 2


def test_the_context_holds_only_agreed_units_with_placeholders(spans: list) -> None:
    items = {frozenset(i.options): i for i in judge_items(spans, 10, seed=0)}
    first = items[frozenset({"今", "明"})]
    assert first.left == "我们"
    assert first.right == f"天买了这只股票然后卖{UNCERTAIN}"
    second = items[frozenset({"出", "掉"})]
    assert second.left == f"我们{UNCERTAIN}天买了这只股票然后卖"
    rendered = judge.render_batch([first, second])
    assert f"[s1] 我们 ⟪?⟫ 天买了这只股票然后卖{UNCERTAIN}" in rendered
    assert f"[s2] 我们{UNCERTAIN}天买了这只股票然后卖 ⟪?⟫" in rendered


def test_reading_order_is_seeded_per_span() -> None:
    a = judge.make_item("x#0#0", "", "", ["甲", "乙", "丙", "丁"], seed=1)
    assert a == judge.make_item("x#0#0", "", "", ["丁", "丙", "乙", "甲"], seed=1)
    orders = {judge.make_item("x#0#0", "", "", ["甲", "乙", "丙", "丁"], seed=s).options for s in range(20)}
    assert len(orders) > 1


def test_verdicts_are_cached_on_disk_and_reused(spans: list, tmp_path: Path) -> None:
    items = judge_items(spans, 10, seed=0)
    client = FakeClient()
    directory = tmp_path / "judge"
    first = judge.run_judge(client, items, directory, model="m", batch_size=1)
    assert len(client.requests) == 2
    assert {v.choice for v in first.values()} == {i.options[0] for i in items}
    assert len(list((directory / "raw").iterdir())) == 2
    again = judge.run_judge(FakeClient(), items, directory, model="m")
    assert again == first
    # Another model or prompt version is a different cache entry.
    other = FakeClient()
    judge.run_judge(other, items, directory, model="other")
    assert len(other.requests) == 1


def test_a_quota_stop_resumes_where_it_left_off(spans: list, tmp_path: Path) -> None:
    items = judge_items(spans, 10, seed=0)
    directory = tmp_path / "judge"
    partial = judge.run_judge(FakeClient(), items, directory, model="m", batch_size=1, max_calls=1)
    assert len(partial) == 1
    client = FakeClient()
    rest = judge.run_judge(client, items, directory, model="m", batch_size=1)
    assert len(client.requests) == 1
    assert len(rest) == 2


def test_needs_audio_counts_for_nobody(spans: list, tmp_path: Path) -> None:
    items = judge_items(spans, 10, seed=0)
    verdicts = judge.run_judge(FakeClient("audio"), items, tmp_path / "judge", model="m")
    shares = judge_shares(spans, verdicts, HYPOTHESES)
    assert shares == {}


def test_judge_shares_credit_every_engine_with_the_chosen_reading(spans: list, tmp_path: Path) -> None:
    items = judge_items(spans, 10, seed=0)
    verdicts = judge.run_judge(FakeClient(), items, tmp_path / "judge", model="m")
    shares = judge_shares(spans, verdicts, HYPOTHESES)
    chosen = {span.id: verdicts[span.id].choice for span in spans}
    for name in HYPOTHESES:
        wins = sum(span.reading(name).text == chosen[span.id] for span in spans)
        assert shares[name].wins == wins
        assert shares[name].decided == 2


def test_an_answer_for_the_wrong_spans_fails_loud() -> None:
    item = judge.make_item("x", "", "", ["甲", "乙"], 0)
    with pytest.raises(LLMOutputError, match="instead of"):
        judge._verdicts([item], [judge.Decision(span="s2", choice=1, reason="")], "m")
    with pytest.raises(LLMOutputError, match="chose 3"):
        judge._verdicts([item], [judge.Decision(span="s1", choice=3, reason="")], "m")
