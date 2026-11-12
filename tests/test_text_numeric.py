import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ytverbatim.acquire.captions import CaptionKind, CaptionSegment, CaptionTrack, parse_json3
from ytverbatim.asr.chunks import Chunk
from ytverbatim.asr.models import ChunkHypothesis
from ytverbatim.asr.protocol import EngineDef, EngineSpec
from ytverbatim.contract.models import Edit, Flag, LLMUsage, Reading, Word
from ytverbatim.errors import LLMUnavailableError
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Usage
from ytverbatim.text.numeric import (
    Evidence,
    NumericResult,
    NumericStats,
    SecondOpinion,
    TimedText,
    second_opinion_request,
    verify_numbers,
)

CLIENT_ID = "fake:auditor"
PRIMARY = "asr:fake-engine"


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


def verdicts(*items: tuple[int, str | None, str]) -> str:
    return json.dumps(
        {"verdicts": [{"id": i, "value": value, "confidence": conf, "reason": "context"} for i, value, conf in items]},
        ensure_ascii=False,
    )


def manual(*cues: tuple[float, float, str]) -> CaptionTrack:
    return CaptionTrack(
        language="zh",
        kind=CaptionKind.MANUAL,
        source_format="json3",
        segments=tuple(CaptionSegment(start=s, end=e, text=t) for s, e, t in cues),
    )


def run(
    segments: list[TimedText],
    evidence: Evidence,
    client: FakeClient,
    tmp_path: Path,
    *,
    language: str = "zh",
) -> NumericResult:
    return verify_numbers(
        segments, language=language, evidence=evidence, client=client, checkpoint_dir=tmp_path / "checkpoints"
    )


def span_text(result: NumericResult, segment: int, flag: Flag) -> str:
    return result.texts[segment][flag.char_start : flag.char_end]


DISPUTED = [TimedText("这只股票上周涨了十五个百分点，非常强劲", 0.0, 4.0)]
DISPUTED_CAPTIONS = manual((0.0, 4.0, "这只股票上周涨了五十个百分点非常强劲"))
DISPUTED_READINGS = (
    Reading(source=PRIMARY, text="十五个百分点"),
    Reading(source="caption_manual", text="五十个百分点"),
)


def test_agreeing_readings_need_no_flag_and_no_llm(tmp_path: Path) -> None:
    client = FakeClient()
    segments = [TimedText("营收增长了20%，利润也不错", 0.0, 3.0)]
    result = run(
        segments, Evidence(PRIMARY, captions=(manual((0.0, 3.0, "营收增长了20%利润也不错")),)), client, tmp_path
    )
    assert result.texts == ("营收增长了20%，利润也不错",)
    assert result.flags == ((),)
    assert result.edits == ((),)
    assert result.stats == NumericStats(agreed=1)
    assert client.requests == []
    assert result.usage == ()


@pytest.mark.parametrize(
    ("asr", "caption"),
    [("一共是一百五十期的数据", "一共是150期的数据"), ("规模达到两万亿美元左右", "规模达到2万亿美元左右")],
)
def test_readings_that_differ_only_in_surface_form_agree(asr: str, caption: str, tmp_path: Path) -> None:
    client = FakeClient()
    result = run(
        [TimedText(asr, 0.0, 3.0)], Evidence(PRIMARY, captions=(manual((0.0, 3.0, caption)),)), client, tmp_path
    )
    assert result.stats == NumericStats(agreed=1)
    assert result.flags == ((),)
    assert client.requests == []


def test_dispute_resolved_by_choosing_a_listed_reading(tmp_path: Path) -> None:
    client = FakeClient([verdicts((0, "五十个百分点", "high"))])
    result = run(DISPUTED, Evidence(PRIMARY, captions=(DISPUTED_CAPTIONS,)), client, tmp_path)
    assert result.texts == ("这只股票上周涨了五十个百分点，非常强劲",)
    assert result.edits == (
        (
            Edit(
                stage="numeric",
                before="这只股票上周涨了十五个百分点，非常强劲",
                after="这只股票上周涨了五十个百分点，非常强劲",
                llm=CLIENT_ID,
            ),
        ),
    )
    (flag,) = result.flags[0]
    assert flag.kind == "numeric_fixed"
    assert flag.readings == DISPUTED_READINGS
    assert span_text(result, 0, flag) == "五十个百分点"
    assert flag.note.startswith("十五个百分点 → 五十个百分点")
    assert result.stats == NumericStats(disputed=1, fixed=1)
    assert result.usage == (LLMUsage(stage="numeric", client=CLIENT_ID, calls=1, input_tokens=100, output_tokens=20),)
    user = client.requests[0].user
    assert "⟦十五个百分点⟧" in user
    assert '"readings": ["十五个百分点", "五十个百分点"]' in user


def test_a_value_no_source_heard_is_rejected(tmp_path: Path) -> None:
    client = FakeClient([verdicts((0, "五个百分点", "high"))])
    result = run(DISPUTED, Evidence(PRIMARY, captions=(DISPUTED_CAPTIONS,)), client, tmp_path)
    assert result.texts == (DISPUTED[0].text,)
    assert result.edits == ((),)
    (flag,) = result.flags[0]
    assert flag.kind == "numeric_disputed"
    assert flag.readings == DISPUTED_READINGS
    assert "no source heard" in flag.note
    assert result.stats == NumericStats(disputed=1)


@pytest.mark.parametrize(
    "reply",
    [verdicts((0, "五十个百分点", "low")), verdicts((0, None, "low")), verdicts((0, "十五个百分点", "high"))],
    ids=["low-confidence", "uncertain", "kept"],
)
def test_unconfident_or_unchanged_verdicts_stay_disputed(reply: str, tmp_path: Path) -> None:
    result = run(DISPUTED, Evidence(PRIMARY, captions=(DISPUTED_CAPTIONS,)), FakeClient([reply]), tmp_path)
    assert result.texts == (DISPUTED[0].text,)
    assert [f.kind for f in result.flags[0]] == ["numeric_disputed"]
    assert span_text(result, 0, result.flags[0][0]) == "十五个百分点"


def test_malformed_output_degrades_to_disputed_and_is_not_checkpointed(tmp_path: Path) -> None:
    client = FakeClient(["not json", '{"verdicts": "nope"}'])
    result = run(DISPUTED, Evidence(PRIMARY, captions=(DISPUTED_CAPTIONS,)), client, tmp_path)
    (flag,) = result.flags[0]
    assert flag.kind == "numeric_disputed"
    assert "malformed" in flag.note
    assert len(client.requests) == 2
    assert not (tmp_path / "checkpoints").exists()


def test_a_missing_verdict_leaves_the_span_disputed(tmp_path: Path) -> None:
    result = run(DISPUTED, Evidence(PRIMARY, captions=(DISPUTED_CAPTIONS,)), FakeClient([verdicts()]), tmp_path)
    assert [f.kind for f in result.flags[0]] == ["numeric_disputed"]


def test_transport_failures_propagate(tmp_path: Path) -> None:
    def fail(_: LLMRequest) -> str:
        raise LLMUnavailableError("down")

    with pytest.raises(LLMUnavailableError):
        run(DISPUTED, Evidence(PRIMARY, captions=(DISPUTED_CAPTIONS,)), FakeClient(respond=fail), tmp_path)


def test_a_checkpoint_hit_skips_the_llm(tmp_path: Path) -> None:
    evidence = Evidence(PRIMARY, captions=(DISPUTED_CAPTIONS,))
    first = run(DISPUTED, evidence, FakeClient([verdicts((0, "五十个百分点", "high"))]), tmp_path)
    again = FakeClient()
    second = run(DISPUTED, evidence, again, tmp_path)
    assert again.requests == []
    assert second.texts == first.texts
    assert second.usage == ()


def test_fix_that_would_change_prose_is_rejected(tmp_path: Path) -> None:
    segments = [TimedText("他说一共是一百五十期的数据", 0.0, 3.0)]
    captions = manual((0.0, 3.0, "他说一共是150天的数据"))
    result = run(
        segments, Evidence(PRIMARY, captions=(captions,)), FakeClient([verdicts((0, "150天", "high"))]), tmp_path
    )
    assert result.texts == (segments[0].text,)
    (flag,) = result.flags[0]
    assert flag.kind == "numeric_disputed"
    assert "more than the number" in flag.note


def words_for(text: str, start: float, end: float, conf: dict[str, float]) -> tuple[Word, ...]:
    """One word per CJK character or digit run, with ``conf`` overriding the default 0.9 per word."""
    pieces = ["这只", "股票", "的", "目标价", "是", "120", "块"]
    assert "".join(pieces) == text
    step = (end - start) / len(pieces)
    return tuple(
        Word(text=p, start=start + k * step, end=start + (k + 1) * step, conf=conf.get(p, 0.9))
        for k, p in enumerate(pieces)
    )


@pytest.mark.parametrize(
    ("conf", "kinds"), [(0.2, ["numeric_single_source", "low_confidence"]), (0.8, ["numeric_single_source"])]
)
def test_single_source_spans_flag_low_confidence_only_below_the_threshold(
    conf: float, kinds: list[str], tmp_path: Path
) -> None:
    text = "这只股票的目标价是120块"
    words = words_for(text, 0.0, 3.0, {"120": conf})
    client = FakeClient()
    result = run([TimedText(text, 0.0, 3.0)], Evidence(PRIMARY, words=words), client, tmp_path)
    assert [f.kind for f in result.flags[0]] == kinds
    assert all(span_text(result, 0, f) == "120块" for f in result.flags[0])
    assert result.flags[0][0].readings == (Reading(source=PRIMARY, text="120块"),)
    assert f"{conf:.2f}" in result.flags[0][0].note
    assert result.stats == NumericStats(single_source=1)
    assert client.requests == []


def test_an_unalignable_span_becomes_single_source(tmp_path: Path) -> None:
    segments = [TimedText("我们看到营收增长了三十个百分点", 10.0, 13.0)]
    unrelated = manual((10.0, 13.0, "完全无关的另外一段内容在这里25"), (40.0, 43.0, "我们看到营收增长了三十个百分点"))
    result = run(segments, Evidence(PRIMARY, captions=(unrelated,)), FakeClient(), tmp_path)
    assert [f.kind for f in result.flags[0]] == ["numeric_single_source"]
    assert result.stats == NumericStats(single_source=1)


def test_numeral_words_are_never_spans(tmp_path: Path) -> None:
    segments = [TimedText("我们一直觉得两家公司一样好", 0.0, 3.0)]
    captions = manual((0.0, 3.0, "我们一致觉得两家公司依样好"))
    client = FakeClient()
    result = run(segments, Evidence(PRIMARY, captions=(captions,)), client, tmp_path)
    assert result.flags == ((),)
    assert result.stats == NumericStats()
    assert client.requests == []


def test_no_independent_evidence_flags_nothing(tmp_path: Path) -> None:
    segments = [TimedText("目标价是120块，涨了20%", 0.0, 3.0)]
    captions = manual((0.0, 3.0, "目标价是120块涨了20%"))
    evidence = Evidence("caption_manual", captions=(captions,))
    result = run(segments, evidence, FakeClient(), tmp_path)
    assert result.flags == ((),)
    assert result.stats == NumericStats(single_source=2)


def json3_track() -> CaptionTrack:
    events = [
        {"tStartMs": 0, "dDurationMs": 3000, "segs": [{"utf8": "revenue"}, {"utf8": " grew", "tOffsetMs": 600}]},
        {
            "tStartMs": 3000,
            "dDurationMs": 3000,
            "segs": [
                {"utf8": "to"},
                {"utf8": " $15", "tOffsetMs": 400},
                {"utf8": " billion", "tOffsetMs": 900},
                {"utf8": " this", "tOffsetMs": 1500},
                {"utf8": " year", "tOffsetMs": 2000},
            ],
        },
        {
            "tStartMs": 6000,
            "dDurationMs": 3000,
            "segs": [
                {"utf8": "and"},
                {"utf8": " margins", "tOffsetMs": 500},
                {"utf8": " hit", "tOffsetMs": 1100},
                {"utf8": " 40%", "tOffsetMs": 1500},
            ],
        },
    ]
    segments = parse_json3(json.dumps({"events": events}), kind=CaptionKind.AUTO)
    return CaptionTrack(language="en", kind=CaptionKind.AUTO, source_format="json3", segments=segments)


def test_caption_word_timings_align_to_each_segment(tmp_path: Path) -> None:
    segments = [
        TimedText("Revenue grew to $50 billion this year.", 0.0, 5.5),
        TimedText("And margins hit 40%.", 6.0, 9.0),
    ]
    client = FakeClient([verdicts((0, "$15", "low"))])
    result = run(segments, Evidence(PRIMARY, captions=(json3_track(),)), client, tmp_path, language="en")
    (disputed,) = result.flags[0]
    assert disputed.kind == "numeric_disputed"
    assert disputed.readings == (Reading(source=PRIMARY, text="$50"), Reading(source="caption_auto", text="$15"))
    assert result.flags[1] == ()
    assert result.stats == NumericStats(agreed=1, disputed=1)


def test_second_opinion_readings_count_as_evidence(tmp_path: Path) -> None:
    rows = [
        ChunkHypothesis(
            hypothesis="other-engine",
            file="f",
            chunk=0,
            start=0.0,
            end=6.0,
            text="这只股票的目标价是一百二十块，市盈率二十倍",
        ),
    ]
    segments = [TimedText("这只股票的目标价是120块", 0.0, 3.0), TimedText("市盈率三十倍", 3.0, 6.0)]
    client = FakeClient([verdicts((0, "二十倍", "high"))])
    evidence = Evidence(PRIMARY, second_opinion=SecondOpinion.from_hypotheses(rows))
    result = run(segments, evidence, client, tmp_path)
    assert result.flags[0] == ()
    assert result.texts[1] == "市盈率二十倍"
    (flag,) = result.flags[1]
    assert flag.kind == "numeric_fixed"
    assert flag.readings == (Reading(source=PRIMARY, text="三十倍"), Reading(source="asr2:other-engine", text="二十倍"))
    assert result.stats == NumericStats(agreed=1, disputed=1, fixed=1)


def test_offsets_follow_a_length_changing_fix(tmp_path: Path) -> None:
    text = "一共是一百五十期，另外还有20%的增长和3个百分点"
    captions = manual((0.0, 4.0, "一共是160期另外还有20%的增长和3个百分点"))
    client = FakeClient([verdicts((0, "160期", "high"))])
    result = run([TimedText(text, 0.0, 4.0)], Evidence(PRIMARY, captions=(captions,)), client, tmp_path)
    assert result.texts == ("一共是160期，另外还有20%的增长和3个百分点",)
    (fixed,) = result.flags[0]
    assert span_text(result, 0, fixed) == "160期"
    assert result.stats == NumericStats(agreed=2, disputed=1, fixed=1)

    single = manual((0.0, 4.0, "一共是160期另外还有很多增长"))
    client = FakeClient([verdicts((0, "160期", "high"))])
    result = run([TimedText(text, 0.0, 4.0)], Evidence(PRIMARY, captions=(single,)), client, tmp_path / "b")
    assert result.texts == ("一共是160期，另外还有20%的增长和3个百分点",)
    assert [(f.kind, span_text(result, 0, f)) for f in result.flags[0]] == [
        ("numeric_fixed", "160期"),
        ("numeric_single_source", "20%"),
        ("numeric_single_source", "3个百分点"),
    ]


def test_spans_in_many_segments_are_batched(tmp_path: Path) -> None:
    segments = [TimedText(f"第{k}段说涨了十五个百分点", k * 4.0, k * 4.0 + 4.0) for k in range(10)]
    captions = manual(*((k * 4.0, k * 4.0 + 4.0, f"第{k}段说涨了五十个百分点") for k in range(10)))

    def respond(request: LLMRequest) -> str:
        items = json.loads(request.user.split("ITEMS:\n", 1)[1])
        return verdicts(*((item["id"], "五十个百分点", "high") for item in items))

    client = FakeClient(respond=respond)
    result = run(segments, Evidence(PRIMARY, captions=(captions,)), client, tmp_path)
    assert len(client.requests) == 2
    assert all(text.endswith("五十个百分点") for text in result.texts)
    assert result.stats.fixed == 10


def chunked_engine(max_chunk_s: float | None = 30.0) -> EngineDef:
    spec = EngineSpec(
        id="other-engine",
        family="fake",
        repo="example/fake",
        revision="0",
        max_chunk_s=max_chunk_s,
        native_timestamps=False,
        emits_punctuation=True,
        supports_context=False,
        sample_rate=16000,
    )
    return EngineDef(spec=spec, factory="fake", variant={"language": "pass"})


def test_second_opinion_request_covers_only_segments_with_quantities(tmp_path: Path) -> None:
    segments = [
        TimedText("开场白没有数字", 0.0, 5.0),
        TimedText("营收是三十亿", 5.0, 8.0),
        TimedText("利润是五亿", 8.0, 11.0),
        TimedText("中间闲聊", 11.0, 60.0),
        TimedText("最后说20%", 60.0, 63.0),
    ]
    request = second_opinion_request(
        segments, engine=chunked_engine(), file_id="f", wav=tmp_path / "a.wav", out_dir=tmp_path / "out", language="zh"
    )
    assert request is not None
    (job,) = request.files
    assert job.chunks == (Chunk(4.5, 11.5), Chunk(59.5, 63.5))
    assert job.language == "zh"


def test_second_opinion_request_splits_long_spans_and_skips_number_free_input(tmp_path: Path) -> None:
    common = {"file_id": "f", "wav": tmp_path / "a.wav", "out_dir": tmp_path / "out", "language": None}
    request = second_opinion_request([TimedText("长句里有20%", 0.0, 49.0)], engine=chunked_engine(), **common)
    assert request is not None
    assert request.files[0].chunks == (Chunk(0.0, 24.75, forced_cut=True), Chunk(24.75, 49.5))
    assert second_opinion_request([TimedText("没有数字", 0.0, 3.0)], engine=chunked_engine(), **common) is None
    with pytest.raises(ValueError, match="whole files"):
        second_opinion_request([TimedText("20%", 0.0, 3.0)], engine=chunked_engine(None), **common)
