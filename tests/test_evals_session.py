import random
from collections import Counter
from pathlib import Path

import pytest
from evals_fixture import EN, ZH, make_eval, write_pair

from evals.normalize import Language
from evals.screen import Span, SpanReading
from evals.session import (
    KEY_FILE,
    SPANS_FILE,
    build_session,
    clip_window,
    load_key,
    load_shown,
    sample_spans,
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


def test_clip_uses_the_control_timing_padded() -> None:
    s = span(0, "f", "zh", False)
    control = SpanReading(hypothesis="c", text="甲", start=10.0, end=11.5, before=9.0, after=12.0)
    assert clip_window(control, s, 100.0) == (9.0, 12.5)
    empty = SpanReading(hypothesis="c", text="", start=None, end=None, before=9.0, after=12.0)
    assert clip_window(empty, s, 100.0) == (8.0, 13.0)
    at_start = SpanReading(hypothesis="c", text="", start=None, end=None, before=None, after=0.4)
    assert clip_window(at_start, s, 100.0) == (0.0, 1.4)
    at_end = SpanReading(hypothesis="c", text="甲", start=99.5, end=99.9, before=None, after=None)
    assert clip_window(at_end, s, 100.0) == (98.5, 100.0)


@pytest.fixture
def pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Layout, object]:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    write_pair(layout)
    return layout, manifest


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


def test_clips_follow_the_control_word_timestamps(pair: tuple[Layout, object]) -> None:
    layout, manifest = pair
    shown, key = build_session(layout, manifest, "w-prod", "q-base", seed=0)  # type: ignore[arg-type]
    by_option = {}
    for item, entry in zip(shown, key.entries, strict=True):
        by_option[item.options[2 - entry.challenger_option]] = item
    # 一 sits on the 6th of 8 characters of the 5-9 s chunk in the control's word timing.
    one = by_option["一"]
    assert (one.clip_start, one.clip_end) == pytest.approx((5.0 + 5 * 0.5 - 1.0, 5.0 + 6 * 0.5 + 1.0))


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
    with pytest.raises(ConfigError, match="no disputes"):
        build_session(layout, manifest, "w-prod", "f-aed")  # type: ignore[arg-type]
