import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ytverbatim.contract.models import LLMUsage
from ytverbatim.errors import LLMOutputError
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Usage
from ytverbatim.text import punctuate as punctuate_module
from ytverbatim.text.punctuate import PunctuationResult, guard_violation, needs_punctuation, punctuate

CLIENT_ID = "fake:punctuator"
ZH_BARE = "大家好欢迎回来今天我们聊聊个股的估值问题" * 5
EN_BARE = "so today we are going to talk about how the market reacts when the fed changes its mind about rates"
EN_FINE = "The Fed met on Tuesday. Rates were left alone. Markets rose. Nobody was surprised by that outcome at all."


def reply(*items: tuple[int, str]) -> str:
    return json.dumps({"items": [{"i": i, "text": text} for i, text in items]}, ensure_ascii=False)


def asked(request: LLMRequest) -> list[dict[str, Any]]:
    payload = request.user.split("SEGMENTS:\n", 1)[1]
    items: list[dict[str, Any]] = json.loads(payload)
    return items


@dataclass
class FakeClient:
    """Replies in order, or through ``respond`` when it is set."""

    replies: list[str] = field(default_factory=list)
    respond: Callable[[LLMRequest], str] | None = None
    requests: list[LLMRequest] = field(default_factory=list)

    @property
    def id(self) -> str:
        return CLIENT_ID

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        text = self.respond(request) if self.respond is not None else self.replies.pop(0)
        return LLMResponse(text=text, usage=Usage(100, 20), elapsed_s=0.01)


def run(texts: list[str], client: FakeClient, tmp_path: Path, language: str = "zh") -> PunctuationResult:
    return punctuate(texts, language=language, client=client, checkpoint_dir=tmp_path / "checkpoints")


def test_density_flags_zh_without_marks_and_not_punctuated_zh() -> None:
    assert needs_punctuation(ZH_BARE, "zh")
    assert needs_punctuation("大家好 欢迎回来", "zh")
    assert not needs_punctuation("大家好，欢迎回来。今天我们聊聊个股的估值问题。", "zh")


def test_density_flags_a_lowercase_english_intro_but_not_the_punctuated_rest() -> None:
    assert needs_punctuation(EN_BARE, "en")
    assert not needs_punctuation(EN_FINE, "en")


def test_a_lowercase_english_stretch_that_already_ends_in_periods_needs_recasing() -> None:
    assert needs_punctuation("hey everyone welcome back to the channel. today we cover the q4 2026 results.", "en")


def test_cased_english_and_short_lowercase_fragments_are_left_alone() -> None:
    assert not needs_punctuation("Hey everyone, welcome back to the channel. Today we cover the q4 2026 results.", "en")
    assert not needs_punctuation("so we go.", "en")
    assert not needs_punctuation("1 2 3 4 5 6 7 8 9 10.", "en")


def test_zh_text_with_marks_is_unaffected_by_the_case_check() -> None:
    assert not needs_punctuation("大家好，欢迎回来。今天我们聊聊个股的估值问题。", "zh")


def test_a_lowercase_punctuated_english_stretch_is_sent_to_the_llm(tmp_path: Path) -> None:
    text = "hey everyone welcome back to the channel. today we cover the q4 2026 results."
    fixed = "Hey everyone welcome back to the channel. Today we cover the Q4 2026 results."
    client = FakeClient([reply((0, fixed))])
    result = run([text, EN_FINE], client, tmp_path, "en")
    assert [item["i"] for item in asked(client.requests[0])] == [0]
    assert result.texts == (fixed, EN_FINE)
    assert "even when the text already ends in a period" in client.requests[0].system


def test_density_ignores_blank_text() -> None:
    assert not needs_punctuation("   ", "en")
    assert not needs_punctuation("", "zh")


def test_only_the_unpunctuated_stretches_go_to_the_llm(tmp_path: Path) -> None:
    punctuated = "so today we talk about the fed. " + EN_FINE
    client = FakeClient(
        [
            reply(
                (
                    0,
                    "So today we are going to talk about how the market reacts when the Fed changes its mind about "
                    "rates.",
                )
            )
        ]
    )
    result = run([EN_BARE, EN_FINE, punctuated], client, tmp_path, "en")
    assert [item["text"] for item in asked(client.requests[0])] == [EN_BARE]
    assert result.texts[0].startswith("So today")
    assert result.texts[1:] == (EN_FINE, punctuated)
    assert result.failed == ()


def test_already_punctuated_text_makes_no_llm_call(tmp_path: Path) -> None:
    client = FakeClient()
    result = run(["大家好，欢迎回来。今天我们聊聊个股的估值问题。"], client, tmp_path)
    assert client.requests == []
    assert result.texts == ("大家好，欢迎回来。今天我们聊聊个股的估值问题。",)
    assert result.edits == ((),)
    assert result.usage == ()


def test_zh_pause_spaces_become_full_width_punctuation_and_are_recorded(tmp_path: Path) -> None:
    client = FakeClient([reply((0, "大家好，欢迎回来。"))])
    result = run(["大家好 欢迎回来"], client, tmp_path)
    assert result.texts == ("大家好，欢迎回来。",)
    edit = result.edits[0][0]
    assert (edit.stage, edit.before, edit.after, edit.llm) == (
        "punctuation",
        "大家好 欢迎回来",
        "大家好，欢迎回来。",
        CLIENT_ID,
    )
    assert "，。？！；：、" in client.requests[0].system
    assert [(u.stage, u.client, u.calls, u.input_tokens) for u in result.usage] == [("punctuation", CLIENT_ID, 1, 100)]


def test_zh_tagged_text_without_cjk_gets_the_english_style(tmp_path: Path) -> None:
    client = FakeClient([reply((0, "Hello everyone, welcome back."))])
    run(["hello everyone welcome back"], client, tmp_path, "zh")
    assert "ASCII punctuation" in client.requests[0].system


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("大家好 欢迎回来", "大家好，欢迎回来。"),
        ("so the fed raised rates", "So, the Fed raised rates."),
        ("buy sp 500 at 3.5%", "Buy SP 500 at 3.5%."),
        ("we hold nvda and amd", "We hold NVDA, and AMD."),
        ("S&P 500 rose 3.5%", "S&P 500 rose 3.5%."),
        ("全角，标点。", "全角,标点."),
        ("prices rose 12 today", "Prices rose 12. Today."),
        ("in 2018 in midterm years", "In 2018, in midterm years."),
        ("上涨了12 今天收盘", "上涨了12，今天收盘。"),
        ("rose 1,000 points", "Rose 1,000 points."),
        ("a 10-K filing", "A 10-K filing."),
    ],
)
def test_guard_accepts_punctuation_spacing_and_case_changes(before: str, after: str) -> None:
    assert guard_violation(before, after) is None


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("大家好欢迎回来", "大家好，欢迎你回来。"),
        ("今天聊个股估值", "今天聊各股估值。"),
        ("the fed raised rates again", "The Fed raised rates."),
        ("rates rose 3.5 percent", "Rates rose 3.6 percent."),
        ("rates rose 3.5 percent", "Rates rose 35 percent."),
        ("rates rose 3 5 percent", "Rates rose 3,5 percent."),
        ("S&P 500 rose", "SP 500 rose."),
        ("up 3.5%", "up 3.5."),
        ("今天上涨三个点", "今天上涨3个点。"),
    ],
)
def test_guard_rejects_changed_letters_characters_and_digits(before: str, after: str) -> None:
    assert guard_violation(before, after) is not None


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("rose 1,000 points", "rose 1，000 points."),
        ("rates at 3.5 now", "rates at 3．5 now."),
        ("fell -5% today", "fell 5% today."),
        ("about 1/2 of it", "about 12 of it."),
        ("at 3:30 pm", "at 330 pm."),
        ("the 10-K filing", "the 10K filing."),
        ("the 1000 shares", "the 1,000 shares."),
        ("we hold NVDA", "We hold nvda."),
        ("the Fed met", "the fed met."),
    ],
)
def test_guard_rejects_number_punctuation_and_lowercasing(before: str, after: str) -> None:
    assert guard_violation(before, after) is not None


def test_guard_names_a_lowercased_letter() -> None:
    assert guard_violation("we hold NVDA", "We hold nvda.") == 'lowercased "NVDA"'


def test_the_prompt_forbids_touching_punctuation_at_numbers(tmp_path: Path) -> None:
    client = FakeClient([reply((0, "Hello, everyone."))])
    run(["hello everyone"], client, tmp_path, "en")
    assert "next to a number" in client.requests[0].system and "Never lowercase" in client.requests[0].system


def test_guard_diff_names_the_change() -> None:
    assert guard_violation("今天聊个股估值", "今天聊各股估值。") == 'changed "个" to "各" after "今天聊"'
    assert guard_violation("the fed cut rates", "The Fed rates.") == 'dropped "cut" after "the fed"'


def test_guard_failure_retries_once_with_the_diff_and_takes_the_fixed_reply(tmp_path: Path) -> None:
    client = FakeClient([reply((0, "今天聊各股估值。")), reply((0, "今天聊个股估值。"))])
    result = run(["今天聊个股估值"], client, tmp_path)
    assert result.texts == ("今天聊个股估值。",)
    assert result.failed == ()
    retry = asked(client.requests[1])[0]
    assert retry["text"] == "今天聊个股估值"
    assert retry["rejected_reply"] == "今天聊各股估值。"
    assert 'changed "个" to "各"' in retry["problem"]


def test_second_guard_failure_keeps_the_original_and_reports_the_index(tmp_path: Path) -> None:
    texts = ["今天聊个股估值", "大家好 欢迎回来"]
    client = FakeClient([reply((0, "今天聊各股估值。"), (1, "大家好，欢迎回来。")), reply((0, "今天聊各股估值！"))])
    result = run(texts, client, tmp_path)
    assert result.texts == ("今天聊个股估值", "大家好，欢迎回来。")
    assert result.failed == (0,)
    assert result.edits[0] == ()
    assert len(client.requests) == 2
    assert [item["text"] for item in asked(client.requests[1])] == ["今天聊个股估值"]


def test_a_batch_with_a_kept_original_is_not_checkpointed(tmp_path: Path) -> None:
    bad = reply((0, "今天聊各股估值。"))
    run(["今天聊个股估值"], FakeClient([bad, bad]), tmp_path)
    assert not (tmp_path / "checkpoints").exists()


def test_checkpoint_hit_skips_the_llm(tmp_path: Path) -> None:
    first = FakeClient([reply((0, "大家好，欢迎回来。"))])
    run(["大家好 欢迎回来"], first, tmp_path)
    second = FakeClient()
    result = run(["大家好 欢迎回来"], second, tmp_path)
    assert second.requests == []
    assert result.texts == ("大家好，欢迎回来。",)
    assert len(result.edits[0]) == 1
    # The stored batch still counts: usage is what the result cost, not what this call spent.
    assert result.usage == (
        LLMUsage(stage="punctuation", client=CLIENT_ID, calls=1, input_tokens=100, output_tokens=20),
    )


def test_checkpoint_is_keyed_by_client_and_prompt_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run(["大家好 欢迎回来"], FakeClient([reply((0, "大家好，欢迎回来。"))]), tmp_path)
    monkeypatch.setattr(punctuate_module, "PROMPT_VERSION", punctuate_module.PROMPT_VERSION + "-bumped")
    other = FakeClient([reply((0, "大家好，欢迎回来。"))])
    run(["大家好 欢迎回来"], other, tmp_path)
    assert len(other.requests) == 1
    assert len(list((tmp_path / "checkpoints").glob("*.json"))) == 2


def test_checkpoint_that_no_longer_passes_the_guard_is_redone(tmp_path: Path) -> None:
    run(["大家好 欢迎回来"], FakeClient([reply((0, "大家好，欢迎回来。"))]), tmp_path)
    (checkpoint,) = (tmp_path / "checkpoints").glob("*.json")
    checkpoint.write_text(reply((0, "大家好，欢迎你回来。")), encoding="utf-8")
    client = FakeClient([reply((0, "大家好，欢迎回来！"))])
    result = run(["大家好 欢迎回来"], client, tmp_path)
    assert len(client.requests) == 1
    assert result.texts == ("大家好，欢迎回来！",)


def test_unreadable_checkpoint_is_redone(tmp_path: Path) -> None:
    run(["大家好 欢迎回来"], FakeClient([reply((0, "大家好，欢迎回来。"))]), tmp_path)
    (checkpoint,) = (tmp_path / "checkpoints").glob("*.json")
    checkpoint.write_text("{", encoding="utf-8")
    client = FakeClient([reply((0, "大家好，欢迎回来。"))])
    run(["大家好 欢迎回来"], client, tmp_path)
    assert len(client.requests) == 1


def test_malformed_output_twice_raises(tmp_path: Path) -> None:
    client = FakeClient(["not json", "still not json"])
    with pytest.raises(LLMOutputError):
        run(["大家好 欢迎回来"], client, tmp_path)


def test_wrong_indices_twice_raises(tmp_path: Path) -> None:
    client = FakeClient([reply((5, "大家好，欢迎回来。")), reply((0, "大家好，欢迎回来。"), (0, "大家好，欢迎回来。"))])
    with pytest.raises(LLMOutputError, match="indices"):
        run(["大家好 欢迎回来"], client, tmp_path)
    assert "[5]" in client.requests[1].user


def test_wrong_indices_once_is_retried(tmp_path: Path) -> None:
    client = FakeClient([reply((3, "x")), reply((0, "大家好，欢迎回来。"))])
    assert run(["大家好 欢迎回来"], client, tmp_path).texts == ("大家好，欢迎回来。",)


def test_batches_hold_about_1500_characters_and_use_batch_local_indices(tmp_path: Path) -> None:
    texts = [("大家好欢迎回来" * 100) for _ in range(5)]
    seen: list[list[int]] = []

    def respond(request: LLMRequest) -> str:
        items = asked(request)
        seen.append([item["i"] for item in items])
        return reply(*((int(item["i"]), item["text"] + "。") for item in items))

    result = run(texts, FakeClient(respond=respond), tmp_path)
    assert seen == [[0, 1, 2], [0, 1]]
    assert all(text.endswith("。") for text in result.texts)
