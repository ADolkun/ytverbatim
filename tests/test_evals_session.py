import random
from collections import Counter
from pathlib import Path

import pytest
from evals_fixture import EN, ZH, make_eval, write_pair

from evals.normalize import Language
from evals.screen import Span, SpanReading, collect_spans
from evals.session import (
    KEY_FILE,
    SPANS_FILE,
    build_session,
    clip_window,
    load_key,
    load_shown,
    sample_spans,
    session_from_spans,
    write_session,
)
from evals.store import Layout
from ytverbatim.errors import ConfigError


def span(index: int, file: str, language: Language, numeric: bool) -> Span:
    readings = (
        SpanReading(hypothesis="c", text=f"a{index}", start=1.0, end=2.0, before=0.5, after=2.5),
        SpanReading(hypothesis="x", text=f"b{index}", start=1.0, end=2.0, before=0.5, after=2.5),
    )
    return Span(id=f"{file}#0#{index}", file=file, language=language, window=0, readings=readings,
                distinct=(f"a{index}", f"b{index}"), left="", right="", start=1.0, end=2.0,
                touches_quantity=numeric, touches_latin=False, format_only=False)  # fmt: skip


def population() -> list[Span]:
    spans: list[Span] = []
    for f, (language, count, numeric_every) in {
        "zh-a": ("zh", 300, 10), "zh-b": ("zh", 100, 5), "en-a": ("en", 80, 4), "en-b": ("en", 40, 2),
    }.items():  # fmt: skip
        spans += [span(i, f, language, i % numeric_every == 0) for i in range(count)]  # type: ignore[arg-type]
    return spans


def test_sampling_splits_languages_and_oversamples_numeric_spans() -> None:
    picked = sample_spans(population(), 150, random.Random(0))
    assert len(picked) == 150
    by_language = Counter(s.language for s in picked)
    assert by_language == {"zh": 75, "en": 75}
    for language in ("zh", "en"):
        numeric = sum(s.numeric for s in picked if s.language == language)
        assert numeric == 30  # 40% of 75, though only 12-50% of the population is numeric
    files = Counter(s.file for s in picked if s.language == "zh" and not s.numeric)
    assert abs(files["zh-a"] - files["zh-b"]) <= 1
    assert len({s.id for s in picked}) == 150


def test_a_short_stratum_or_language_hands_its_quota_to_the_rest() -> None:
    spans = [span(i, "zh-a", "zh", i < 5) for i in range(100)] + [span(i, "en-a", "en", False) for i in range(10)]
    picked = sample_spans(spans, 60, random.Random(1))
    assert len(picked) == 60
    assert sum(s.language == "en" for s in picked) == 10
    assert sum(s.numeric for s in picked) == 5


def test_sampling_is_seeded() -> None:
    a = [s.id for s in sample_spans(population(), 50, random.Random(7))]
    assert a == [s.id for s in sample_spans(population(), 50, random.Random(7))]
    assert a != [s.id for s in sample_spans(population(), 50, random.Random(8))]


def reading(
    start: float | None, end: float | None, before: float | None = None, after: float | None = None
) -> SpanReading:
    return SpanReading(hypothesis="h", text="甲" if start is not None else "", start=start, end=end, before=before,
                       after=after)  # fmt: skip


def test_clip_spans_the_union_of_both_readings_padded() -> None:
    s = span(0, "f", "zh", False)
    assert clip_window((reading(10.0, 14.0), reading(12.0, 17.0)), s, 100.0) == (9.0, 18.0)
    assert clip_window((reading(12.0, 17.0), reading(10.0, 14.0)), s, 100.0) == (9.0, 18.0)


def test_an_empty_reading_uses_its_neighbouring_units() -> None:
    s = span(0, "f", "zh", False)
    assert clip_window((reading(None, None, 9.0, 15.0), reading(10.0, 14.0)), s, 100.0) == (8.0, 16.0)


def test_a_short_clip_is_widened_to_five_seconds_around_its_centre() -> None:
    s = span(0, "f", "zh", False)
    assert clip_window((reading(10.0, 11.0), reading(10.2, 11.5)), s, 100.0) == (8.25, 13.25)


def test_the_minimum_clip_is_shifted_inside_the_file() -> None:
    s = span(0, "f", "zh", False)
    assert clip_window((reading(None, None, None, 0.4), reading(0.1, 0.4)), s, 100.0) == (0.0, 5.0)
    assert clip_window((reading(99.5, 99.9), reading(99.6, 99.9)), s, 100.0) == (95.0, 100.0)
    assert clip_window((reading(1.0, 2.0), reading(1.0, 2.0)), s, 3.0) == (0.0, 3.0)


@pytest.fixture
def pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Layout, object]:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    write_pair(layout)
    return layout, manifest


def test_session_refuses_a_hypothesis_that_is_not_complete_on_every_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    write_pair(layout)
    rows = layout.rows_path("q-base", EN)
    lines = rows.read_text(encoding="utf-8").splitlines(keepends=True)
    rows.write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"q-base is not complete for 1 of 2 files"):
        build_session(layout, manifest, "w-prod", "q-base")
    rows.unlink()
    with pytest.raises(ConfigError, match=r"q-base is not complete"):
        build_session(layout, manifest, "w-prod", "q-base")


def test_session_hides_engines_and_keeps_the_key_apart(pair: tuple[Layout, object]) -> None:
    layout, manifest = pair
    shown, key = build_session(layout, manifest, "w-prod", "q-base", n=20, seed=4)  # type: ignore[arg-type]
    assert write_session(layout, "main", shown, key)
    text = (layout.session_dir("main") / SPANS_FILE).read_text(encoding="utf-8")
    assert "w-prod" not in text and "q-base" not in text and "challenger" not in text
    assert load_key(layout.session_dir("main")).control == "w-prod"
    for item, entry in zip(shown, key.entries, strict=True):
        assert item.id == entry.id
        challenger_text = item.options[entry.challenger_option - 1]
        control_text = item.options[2 - entry.challenger_option]
        assert challenger_text != control_text
    assert {e.file for e in key.entries} == {ZH, EN}
    assert sum(e.stratum == "numeric" for e in key.entries) >= 2
    assert all(e.weight >= 1.0 for e in key.entries)


def test_side_randomisation_is_seeded_and_mixes_sides(pair: tuple[Layout, object]) -> None:
    layout, manifest = pair
    sides = {
        seed: tuple(
            e.challenger_option for e in build_session(layout, manifest, "w-prod", "q-base", seed=seed)[1].entries
        )  # type: ignore[arg-type]
        for seed in range(12)
    }
    assert sides[3] == tuple(
        e.challenger_option
        for e in build_session(layout, manifest, "w-prod", "q-base", seed=3)[1].entries  # type: ignore[arg-type]
    )
    assert len(set(sides.values())) > 1
    assert {side for values in sides.values() for side in values} == {1, 2}


def test_clips_follow_the_word_timestamps_with_a_five_second_minimum(pair: tuple[Layout, object]) -> None:
    layout, manifest = pair
    shown, key = build_session(layout, manifest, "w-prod", "q-base", seed=0)  # type: ignore[arg-type]
    by_option = {}
    for item, entry in zip(shown, key.entries, strict=True):
        by_option[item.options[2 - entry.challenger_option]] = item
    # 一 sits on the 6th of 8 characters of the 5-9 s chunk in the control's word timing (7.5-8.0 s); padded that is
    # 2.5 s, so the clip widens to 5 s around its centre.
    one = by_option["一"]
    assert (one.clip_start, one.clip_end) == pytest.approx((7.75 - 2.5, 7.75 + 2.5))


def test_an_existing_session_is_kept_or_refused(pair: tuple[Layout, object]) -> None:
    layout, manifest = pair
    shown, key = build_session(layout, manifest, "w-prod", "q-base", seed=1)  # type: ignore[arg-type]
    assert write_session(layout, "main", shown, key)
    assert not write_session(layout, "main", *build_session(layout, manifest, "w-prod", "q-base", seed=1))  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="already exists"):
        write_session(layout, "main", *build_session(layout, manifest, "w-prod", "q-base", seed=2))  # type: ignore[arg-type]
    assert load_shown(layout.session_dir("main")) == shown
    assert (layout.session_dir("main") / KEY_FILE).exists()


def test_session_needs_two_different_complete_runs(pair: tuple[Layout, object]) -> None:
    layout, manifest = pair
    with pytest.raises(ConfigError, match="must differ"):
        build_session(layout, manifest, "w-prod", "w-prod")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="f-aed is not complete for 2 of 2 files"):
        build_session(layout, manifest, "w-prod", "f-aed")  # type: ignore[arg-type]


def test_session_from_spans_equals_build_session_on_the_same_spans(pair: tuple[Layout, object]) -> None:
    layout, manifest = pair
    spans = collect_spans(layout, manifest, ["w-prod", "q-base"], pivot="w-prod")  # type: ignore[arg-type]
    built, built_key = build_session(layout, manifest, "w-prod", "q-base", n=20, seed=4)  # type: ignore[arg-type]
    shown, key = session_from_spans(spans, manifest, "w-prod", "q-base", n=20, seed=4)  # type: ignore[arg-type]
    assert shown == built
    assert key.model_copy(update={"created": built_key.created}) == built_key


def test_session_from_spans_refuses_when_no_audible_span_is_left(pair: tuple[Layout, object]) -> None:
    layout, manifest = pair
    spans = collect_spans(layout, manifest, ["w-prod", "q-base"], pivot="w-prod")  # type: ignore[arg-type]
    written_form = [s.model_copy(update={"format_only": True}) for s in spans]
    with pytest.raises(ConfigError, match="no audible disputes"):
        session_from_spans(written_form, manifest, "w-prod", "q-base")  # type: ignore[arg-type]
