import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ytverbatim.contract.models import Edit, Flag, LLMUsage
from ytverbatim.errors import LLMOutputError
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Usage
from ytverbatim.text import correct
from ytverbatim.text.correct import CorrectionContext, CorrectionResult, Rejection, correct_segments

CLIENT_ID = "fake:proofreader"


def fixed(*items: tuple[int, str]) -> str:
    return json.dumps({"fixed": [{"i": i, "text": text} for i, text in items]}, ensure_ascii=False)


@dataclass
class FakeClient:
    """Replies in order, or through ``respond`` when it is set."""

    replies: list[str] = field(default_factory=list)
    respond: Callable[[LLMRequest], str] | None = None
    requests: list[LLMRequest] = field(default_factory=list)
    name: str = CLIENT_ID

    @property
    def id(self) -> str:
        return self.name

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        text = self.respond(request) if self.respond is not None else self.replies.pop(0)
        return LLMResponse(text=text, usage=Usage(100, 20), elapsed_s=0.01)


def run(
    texts: list[str],
    client: FakeClient,
    tmp_path: Path,
    *,
    context: CorrectionContext | None = None,
    confidence: list[list[float] | None] | None = None,
) -> CorrectionResult:
    return correct_segments(
        texts,
        client=client,
        language="zh",
        context=context or CorrectionContext(),
        checkpoint_dir=tmp_path / "checkpoints",
        confidence=confidence,
    )


def test_applies_a_fix_and_preserves_structure(tmp_path: Path) -> None:
    result = run(["动态适应率", "没问题"], FakeClient([fixed((0, "动态市盈率"))]), tmp_path)
    assert result.texts == ("动态市盈率", "没问题")
    assert result.edits == (
        (Edit(stage="correction", before="动态适应率", after="动态市盈率", llm=CLIENT_ID),),
        (),
    )
    assert result.flags == ((), ())
    assert result.rejections == ()
    assert result.usage == (
        LLMUsage(stage="correction", client=CLIENT_ID, calls=1, input_tokens=100, output_tokens=20),
    )


def test_rejected_fixes_are_reported_with_their_guard(tmp_path: Path) -> None:
    texts = ["这是一个完整的中文句子用于测试", "目标价 $1000", "我们要买入这只股票"]
    client = FakeClient([fixed((0, "   "), (1, "目标价 $1500"), (2, "我们要卖出这只股票"))])
    result = run(texts, client, tmp_path)
    assert result.texts == tuple(texts)
    assert result.edits == ((), (), ())
    assert [(r.segment, r.reason) for r in result.rejections] == [(0, "empty"), (1, "digits"), (2, "direction")]


def test_bad_and_repeated_indices_are_skipped(tmp_path: Path) -> None:
    client = FakeClient([fixed((5, "越界索引"), (0, "动态市盈率"), (0, "动态适应律"))])
    result = run(["动态适应率"], client, tmp_path)
    assert result.texts == ("动态市盈率",)
    assert [(r.segment, r.reason) for r in result.rejections] == [(5, "index"), (0, "index")]


def test_an_echoed_segment_is_neither_an_edit_nor_a_rejection(tmp_path: Path) -> None:
    result = run(["没问题"], FakeClient([fixed((0, " 没问题 "))]), tmp_path)
    assert result.edits == ((),)
    assert result.rejections == ()


def test_acronym_fixes_are_flagged_at_their_offsets_after_a_length_change(tmp_path: Path) -> None:
    client = FakeClient([fixed((0, "according to GAAP and MSCI"))])
    result = run(["accordng to GAP and SMCI"], client, tmp_path)
    (text,) = result.texts
    assert text == "according to GAAP and MSCI"
    assert result.flags == (
        (
            Flag(kind="entity_corrected", char_start=13, char_end=17, note="GAP → GAAP"),
            Flag(kind="entity_corrected", char_start=22, char_end=26, note="SMCI → MSCI"),
        ),
    )
    assert [text[f.char_start : f.char_end] for f in result.flags[0]] == ["GAAP", "MSCI"]


@pytest.mark.parametrize(
    ("original", "proposed"),
    [
        ("买入 NVDA 的理由", "买入 的理由"),
        ("今年的适应率", "今年的NVDA适应率"),
        ("目标价位是一百元", "目标价位是一千元"),
        ("revenue was $1.2M this year", "revenue was $12M this year"),
        ("we do not buy here", "we do buy here"),
        ("我们要买入这只股票", "我们要卖出这只股票"),
    ],
)
def test_unsafe_proposals_leave_the_text_alone(tmp_path: Path, original: str, proposed: str) -> None:
    result = run([original], FakeClient([fixed((0, proposed))]), tmp_path)
    assert result.texts == (original,)
    assert result.edits == ((),)
    assert len(result.rejections) == 1


def test_word_fix_around_a_numeral_is_accepted(tmp_path: Path) -> None:
    result = run(["他一致在说这个问题"], FakeClient([fixed((0, "他一直在说这个问题"))]), tmp_path)
    assert result.texts == ("他一直在说这个问题",)


def test_malformed_reply_is_retried_once(tmp_path: Path) -> None:
    client = FakeClient(['{"oops": true}', fixed((0, "动态市盈率"))])
    result = run(["动态适应率"], client, tmp_path)
    assert result.texts == ("动态市盈率",)
    assert len(client.requests) == 2
    assert result.usage[0].calls == 2


def test_malformed_reply_twice_raises(tmp_path: Path) -> None:
    client = FakeClient(['{"result": "wrong shape"}', '{"fixed": [{"i": "0", "text": "x"}]}'])
    with pytest.raises(LLMOutputError):
        run(["原文"], client, tmp_path)
    assert len(client.requests) == 2
    assert not (tmp_path / "checkpoints").exists() or not any((tmp_path / "checkpoints").iterdir())


def test_llm_failure_propagates(tmp_path: Path) -> None:
    def boom(request: LLMRequest) -> str:
        raise RuntimeError("llm down")

    with pytest.raises(RuntimeError):
        run(["原文"], FakeClient(respond=boom), tmp_path)


def test_a_checkpoint_hit_skips_the_llm(tmp_path: Path) -> None:
    first = run(["动态适应率", "没问题"], FakeClient([fixed((0, "动态市盈率"))]), tmp_path)
    again = FakeClient()
    second = run(["动态适应率", "没问题"], again, tmp_path)
    assert again.requests == []
    assert second.texts == first.texts
    assert second.edits == first.edits
    assert second.usage == ()


def test_checkpoints_are_keyed_by_client_and_batch_text(tmp_path: Path) -> None:
    run(["动态适应率"], FakeClient([fixed()]), tmp_path)
    other_client = FakeClient([fixed()], name="fake:other")
    run(["动态适应率"], other_client, tmp_path)
    other_text = FakeClient([fixed()])
    run(["前南很好"], other_text, tmp_path)
    assert len(other_client.requests) == 1
    assert len(other_text.requests) == 1
    assert len(list((tmp_path / "checkpoints").iterdir())) == 3


def test_unreadable_checkpoint_is_redone(tmp_path: Path) -> None:
    run(["动态适应率"], FakeClient([fixed()]), tmp_path)
    (checkpoint,) = (tmp_path / "checkpoints").iterdir()
    checkpoint.write_text("not json")
    client = FakeClient([fixed((0, "动态市盈率"))])
    assert run(["动态适应率"], client, tmp_path).texts == ("动态市盈率",)
    assert len(client.requests) == 1


def test_batches_split_by_character_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(correct, "BATCH_CHARS", 6)
    client = FakeClient(respond=lambda request: fixed())
    run(["一二三四", "五六七八", "甲乙", "丙丁"], client, tmp_path)
    batches = [json.loads(r.user.split("SEGMENTS:\n", 1)[1]) for r in client.requests]
    assert [[item["i"] for item in batch] for batch in batches] == [[0, 1], [2, 3]]


def test_prompt_carries_language_glossary_and_capped_creator_note(tmp_path: Path) -> None:
    client = FakeClient([fixed()])
    context = CorrectionContext(glossary=("市盈率", "GAAP"), creator_note="x" * 700)
    run(["原文"], client, tmp_path, context=context)
    (request,) = client.requests
    assert "The transcript language is zh." in request.user
    assert "Prefer these domain terms when relevant: 市盈率, GAAP." in request.user
    assert "PINNED" in request.user
    assert "x" * 600 + "…" in request.user
    assert "x" * 601 not in request.user
    assert "GAP→GAAP" in request.system


class TestSecondPass:
    TEXTS = ("动态适应率", "没问题")

    @staticmethod
    def second_pass_only(request: LLMRequest) -> str:
        if "LOW-CONFIDENCE" in request.system:
            return fixed((0, "动态市盈率"))
        return fixed()

    def test_fixes_low_confidence_segments_only(self, tmp_path: Path) -> None:
        client = FakeClient(respond=self.second_pass_only)
        result = run(list(self.TEXTS), client, tmp_path, confidence=[[0.2, 0.3], [0.95]])
        assert result.texts == ("动态市盈率", "没问题")
        second = [r for r in client.requests if "LOW-CONFIDENCE" in r.system]
        assert len(second) == 1
        assert '"i": 1' not in second[0].user
        assert [u.stage for u in result.usage] == ["correction", "correction_low_confidence"]

    def test_does_not_run_without_confidence(self, tmp_path: Path) -> None:
        client = FakeClient(respond=self.second_pass_only)
        result = run(list(self.TEXTS), client, tmp_path)
        assert result.texts == self.TEXTS
        assert all("LOW-CONFIDENCE" not in r.system for r in client.requests)

    def test_skips_segments_without_probabilities(self, tmp_path: Path) -> None:
        client = FakeClient(respond=self.second_pass_only)
        result = run(list(self.TEXTS), client, tmp_path, confidence=[None, []])
        assert result.texts == self.TEXTS
        assert len(client.requests) == 1

    def test_guards_also_apply_to_the_second_pass(self, tmp_path: Path) -> None:
        def unsafe(request: LLMRequest) -> str:
            return fixed((0, "我们要卖出")) if "LOW-CONFIDENCE" in request.system else fixed()

        result = run(["我们要买入"], FakeClient(respond=unsafe), tmp_path, confidence=[[0.1]])
        assert result.texts == ("我们要买入",)
        assert result.rejections == (Rejection(0, "low_confidence", "我们要买入", "我们要卖出", "direction"),)

    def test_both_passes_record_an_edit_and_flags_point_at_the_final_text(self, tmp_path: Path) -> None:
        def respond(request: LLMRequest) -> str:
            if "LOW-CONFIDENCE" in request.system:
                return fixed((0, "according to GAAP rules"))
            return fixed((0, "acording to GAAP rules"))

        result = run(["acordng to GAP rules"], FakeClient(respond=respond), tmp_path, confidence=[[0.1]])
        assert result.texts == ("according to GAAP rules",)
        assert [(e.before, e.after) for e in result.edits[0]] == [
            ("acordng to GAP rules", "acording to GAAP rules"),
            ("acording to GAAP rules", "according to GAAP rules"),
        ]
        assert result.flags == ((Flag(kind="entity_corrected", char_start=13, char_end=17, note="GAP → GAAP"),),)

    def test_confidence_must_cover_every_segment(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="confidence"):
            run(list(self.TEXTS), FakeClient(), tmp_path, confidence=[[0.1]])
