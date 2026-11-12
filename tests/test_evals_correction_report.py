from pathlib import Path

import pytest

from evals.correction import BEFORE, PROPOSED, RejectionSpan
from evals.correction_report import ErrorChange, guard_rates, judged
from evals.judge import JUDGE_MODEL, PROMPT_VERSION, JudgeItem, Verdict, make_item
from evals.screen import Span, SpanReading
from evals.store import Layout


def span(id: str, before: str, proposed: str) -> Span:
    readings = {BEFORE: before, PROPOSED: proposed}
    return Span(
        id=id, file="f", language="zh", window=0,
        readings=tuple(
            SpanReading(hypothesis=h, text=t, start=0.0, end=1.0, before=None, after=None) for h, t in readings.items()
        ),
        distinct=(before, proposed), left="", right="", start=0.0, end=1.0,
        touches_quantity=False, touches_latin=False, format_only=False,
    )  # fmt: skip


def rejected(candidate: str, ordinal: int, guard: str, before: str, proposed: str, part: int = 0) -> RejectionSpan:
    return RejectionSpan(
        candidate=candidate,
        guard=guard,
        rejection=ordinal,
        span=span(f"{candidate}:{ordinal}:{part}", before, proposed),
    )


def verdict(choice: str | None) -> Verdict:
    return Verdict(key="k", choice=choice, needs_audio=choice is None, reason="", model="m", prompt_version="v")


def test_guard_rates_count_a_proposal_only_when_every_dispute_is_judged() -> None:
    rejections = [
        rejected("c1", 0, "balloon", "a", "A"),
        rejected("c1", 1, "balloon", "b", "B"),
        rejected("c1", 2, "balloon", "c", "C"),
        rejected("c2", 0, "balloon", "d", "D", part=0),
        rejected("c2", 0, "balloon", "e", "E", part=1),
        rejected("c1", 3, "digits", "f", "F"),
    ]
    verdicts = {
        "c1:0:0": verdict("A"),
        "c1:1:0": verdict("b"),
        "c1:2:0": verdict(None),
        "c2:0:0": verdict("D"),
        "c2:0:1": verdict("e"),
    }
    rates = {g.guard: g for g in guard_rates(rejections, verdicts)}
    balloon = rates["balloon"]
    # c1:2 needs audio, and c2:0 is not a false reject because one of its two disputes kept the original.
    assert (balloon.sampled, balloon.judged, balloon.false_rejects) == (4, 3, 1)
    assert balloon.rate is not None
    assert balloon.rate.value == pytest.approx(1 / 3)
    assert not balloon.informational
    digits = rates["digits"]
    assert (digits.sampled, digits.judged, digits.false_rejects, digits.rate) == (1, 0, 0, None)
    assert digits.informational


def test_verdicts_map_from_the_cache_key_back_to_span_ids(tmp_path: Path) -> None:
    layout = Layout(tmp_path)
    items: list[JudgeItem] = [make_item("s1", "L", "R", ["a", "b"], 0), make_item("s2", "L", "R", ["c", "d"], 0)]
    directory = layout.correction / "judge"
    directory.mkdir(parents=True)
    done = Verdict(
        key=items[1].key(JUDGE_MODEL),
        choice="c",
        needs_audio=False,
        reason="",
        model=JUDGE_MODEL,
        prompt_version=PROMPT_VERSION,
    )
    (directory / "verdicts.jsonl").write_text(done.model_dump_json() + "\n", encoding="utf-8")
    assert judged(layout, items) == {"s2": done}


def test_error_change_reports_the_share_removed() -> None:
    assert ErrorChange(before=4, after=1, units=40).removed_share == pytest.approx(0.75)
    assert ErrorChange(before=0, after=0, units=40).removed_share is None
    total = ErrorChange(before=2, after=1, units=10).plus(ErrorChange(before=3, after=0, units=20))
    assert (total.before, total.after, total.units) == (5, 1, 30)
