import json
from pathlib import Path

import numpy as np
import pytest
from fake_engine import RATE, write_silence
from numpy.typing import NDArray
from scripted_engine import LOOP_TEXT, ScriptedEngine, looping, sound

from ytverbatim.asr import worker
from ytverbatim.asr.chunks import Chunk
from ytverbatim.asr.guard import MIN_REDECODE_S, QualityGuard, Trigger, triggers
from ytverbatim.asr.models import HYPOTHESES_FILE, ChunkHypothesis, FileJob, WorkerRequest
from ytverbatim.asr.protocol import AsrContext, ChunkInput, EngineDef, EngineSpec, RawChunk
from ytverbatim.contract.models import Word

DURATION = 10.0


def tone(seconds: float, *, quiet: tuple[float, float] | None = None) -> NDArray[np.float32]:
    t = np.arange(round(seconds * RATE)) / RATE
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    if quiet is not None:
        audio[round(quiet[0] * RATE) : round(quiet[1] * RATE)] = 0.0
    return audio


def chunk_input(
    audio: NDArray[np.float32], *, language: str | None = "en", context: AsrContext | None = None
) -> ChunkInput:
    return ChunkInput(audio, 100.0, 100.0 + len(audio) / RATE, language, context)


def collapsed(count: int = 12) -> tuple[Word, ...]:
    return tuple(Word(text=f"unique{i}", start=1.0, end=1.0) for i in range(count))


@pytest.mark.parametrize(
    ("raw", "duration", "expected"),
    [
        (sound(), DURATION, ()),
        (RawChunk("fine", generated_tokens=64, max_tokens=64), DURATION, ("token_cap",)),
        (RawChunk("fine", generated_tokens=64), DURATION, ()),
        (RawChunk("fine", max_tokens=64), DURATION, ()),
        (RawChunk(LOOP_TEXT), DURATION, ("repetition_loop",)),
        (RawChunk("风险限制" * 8), DURATION, ("repetition_loop",)),
        (RawChunk("。。。"), DURATION, ("nonlexical_output",)),
        (RawChunk(""), DURATION, ("nonlexical_output",)),
        (RawChunk("。。。"), MIN_REDECODE_S - 0.1, ()),
        (RawChunk("fine", words=collapsed()), DURATION, ("word_timing_collapse",)),
        (RawChunk("fine", words=collapsed(9)), DURATION, ()),
        (RawChunk("damaged � text"), DURATION, ("replacement_character",)),
        (
            RawChunk(LOOP_TEXT, generated_tokens=64, max_tokens=64, words=collapsed()),
            DURATION,
            ("token_cap", "repetition_loop", "word_timing_collapse"),
        ),
    ],
)
def test_trigger_matrix(raw: RawChunk, duration: float, expected: tuple[Trigger, ...]) -> None:
    assert triggers(raw, duration=duration) == expected


def test_dense_punctuation_words_trigger_even_when_the_text_has_letters() -> None:
    words = tuple(Word(text="!", start=1 + i * 0.5, end=1 + i * 0.5 + 0.01) for i in range(10))
    words = (*words[:-1], Word(text="!", start=5.5, end=6.0))
    assert triggers(RawChunk("real text !!!!!!!!!!", words=words), duration=DURATION) == ("nonlexical_output",)


def test_a_sound_decode_is_returned_untouched_without_decoding_again() -> None:
    engine = ScriptedEngine([sound("never asked")])
    raw = sound()
    guarded = QualityGuard()(engine, chunk_input(tone(DURATION)), raw)
    assert guarded.raw is raw
    assert guarded.flags == ()
    assert engine.heard == []


def test_a_passing_redecode_replaces_the_whole_chunk() -> None:
    first = RawChunk(
        "你好",
        language="zh",
        generated_tokens=4,
        max_tokens=64,
        confidence=0.9,
        words=(Word(text="你好", start=0, end=1),),
    )
    second = RawChunk(
        "世界", generated_tokens=6, max_tokens=64, confidence=0.5, words=(Word(text="世界", start=0.2, end=0.8),)
    )
    audio = tone(DURATION, quiet=(4.9, 5.1))
    engine = ScriptedEngine([first, second])
    guarded = QualityGuard()(engine, chunk_input(audio, language="zh"), looping())
    assert guarded.flags == ()
    merged = guarded.raw
    assert merged.text == "你好世界"
    assert merged.language == "zh"
    assert (merged.generated_tokens, merged.max_tokens) == (10, 128)
    cut = len(engine.heard[0]) / RATE
    assert merged.confidence == pytest.approx((0.9 * cut + 0.5 * (DURATION - cut)) / DURATION)
    assert [(w.text, w.start, w.end) for w in merged.words] == [
        ("你好", 0, 1),
        ("世界", pytest.approx(cut + 0.2), pytest.approx(cut + 0.8)),
    ]


def test_english_halves_are_joined_with_a_space() -> None:
    engine = ScriptedEngine([sound("first half."), sound(" second half")])
    guarded = QualityGuard()(engine, chunk_input(tone(DURATION)), looping())
    assert guarded.raw.text == "first half. second half"


def test_halves_come_from_the_chunks_own_audio_cut_at_the_quietest_point_near_the_middle() -> None:
    audio = tone(DURATION, quiet=(4.2, 4.5))
    engine = ScriptedEngine([sound()])
    QualityGuard()(engine, chunk_input(audio), looping())
    first, second = engine.heard
    assert np.array_equal(np.concatenate([first, second]), audio)
    assert np.shares_memory(first, audio)
    assert np.shares_memory(second, audio)
    assert 4.2 <= len(first) / RATE <= 4.5


def test_energy_ties_cut_nearest_the_middle_and_the_cut_stays_in_the_middle_window() -> None:
    engine = ScriptedEngine([sound()])
    QualityGuard()(engine, chunk_input(np.zeros(RATE * 10, dtype=np.float32)), looping())
    assert len(engine.heard[0]) == pytest.approx(5 * RATE, abs=RATE // 10)
    loud_middle = tone(DURATION, quiet=(0.0, 1.0))
    engine = ScriptedEngine([sound()])
    QualityGuard()(engine, chunk_input(loud_middle), looping())
    assert 3.0 <= len(engine.heard[0]) / RATE <= 7.0


def test_the_redecode_reuses_the_language_and_context() -> None:
    seen: list[tuple[str | None, AsrContext | None]] = []

    class Spy(ScriptedEngine):
        def transcribe(
            self, audio: NDArray[np.float32], *, language: str | None, context: AsrContext | None
        ) -> RawChunk:
            seen.append((language, context))
            return super().transcribe(audio, language=language, context=context)

    context = AsrContext(system_prompt="finance")
    QualityGuard()(Spy([sound()]), chunk_input(tone(DURATION), language="zh", context=context), looping())
    assert seen == [("zh", context)] * 2


def test_a_still_failing_redecode_is_the_only_one_and_the_chunk_is_flagged() -> None:
    engine = ScriptedEngine([looping()])
    guarded = QualityGuard()(engine, chunk_input(tone(DURATION)), looping())
    assert len(engine.heard) == 2
    assert guarded.flags == ("decoder_anomaly", "token_cap_hit")


def test_the_attempt_with_fewer_failures_is_kept() -> None:
    worse = RawChunk(LOOP_TEXT, generated_tokens=64, max_tokens=64)
    engine = ScriptedEngine([RawChunk(LOOP_TEXT, generated_tokens=5, max_tokens=64), sound()])
    guarded = QualityGuard()(engine, chunk_input(tone(DURATION)), worse)
    assert guarded.flags == ("decoder_anomaly",)
    assert LOOP_TEXT in guarded.raw.text
    assert guarded.raw.text.endswith("a sound decode")


def test_text_beats_silence_when_both_attempts_fail() -> None:
    original = looping()
    guarded = QualityGuard()(ScriptedEngine([RawChunk(""), RawChunk("")]), chunk_input(tone(DURATION)), original)
    assert guarded.raw is original
    assert guarded.flags == ("decoder_anomaly", "token_cap_hit")


def test_a_loop_across_the_cut_is_caught_in_the_merged_text() -> None:
    half = " ".join(["repeat risk"] * 5)
    guarded = QualityGuard()(ScriptedEngine([sound(half)]), chunk_input(tone(DURATION)), looping())
    assert guarded.flags == ("decoder_anomaly",)


def test_a_chunk_too_short_to_split_is_flagged_without_decoding_again() -> None:
    engine = ScriptedEngine([sound()])
    guarded = QualityGuard()(engine, chunk_input(tone(MIN_REDECODE_S - 0.1)), RawChunk(LOOP_TEXT))
    assert engine.heard == []
    assert guarded.flags == ("decoder_anomaly",)


def request(tmp_path: Path, wav: Path, *, always: bool = False, guard: bool = True) -> WorkerRequest:
    spec = EngineSpec("scripted", "fake", "example/fake", "0" * 40, 30.0, False, False, False, RATE)
    definition = EngineDef(spec, "scripted_engine.create", {"always": always})
    job = FileJob(id="f1", wav=str(wav), language="en", chunks=(Chunk(0.0, 6.0), Chunk(7.0, 9.0)))
    return WorkerRequest(engine=definition, out_dir=str(tmp_path / "out"), files=(job,), quality_guard=guard)


def rows(tmp_path: Path) -> list[ChunkHypothesis]:
    lines = (tmp_path / "out" / HYPOTHESES_FILE).read_text(encoding="utf-8").splitlines()
    return [ChunkHypothesis.model_validate_json(line) for line in lines]


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    return write_silence(tmp_path / "in.wav", 10.0)


def test_run_request_redecodes_a_chunk_whose_first_decode_loops(tmp_path: Path, wav: Path) -> None:
    req = request(tmp_path, wav)
    engine = ScriptedEngine([looping(), sound("left"), sound("right"), sound("untouched")])
    metrics = worker.run_request(req, engine=engine, guard=QualityGuard())
    first, second = rows(tmp_path)
    assert (first.text, first.flags) == ("left right", ())
    assert (first.start, first.end) == (0.0, 6.0)
    assert (second.text, second.flags) == ("untouched", ())
    assert len(engine.heard) == 4
    assert metrics.chunks_decoded == 2


def test_run_request_flags_a_chunk_that_keeps_looping(tmp_path: Path, wav: Path) -> None:
    engine = ScriptedEngine([looping()])
    worker.run_request(request(tmp_path, wav), engine=engine, guard=QualityGuard())
    assert [r.flags for r in rows(tmp_path)] == [("token_cap_hit", "decoder_anomaly")] * 2
    # Per chunk: the first decode plus one re-decode of two halves, however badly the halves fare.
    assert len(engine.heard) == 2 * (1 + 2)


def test_main_guards_by_default_and_the_request_can_opt_out(tmp_path: Path, wav: Path) -> None:
    for guard in (True, False):
        base = tmp_path / str(guard)
        base.mkdir()
        req = request(base, wav, guard=guard)
        path = base / "request.json"
        path.write_text(req.model_dump_json(), encoding="utf-8")
        assert worker.main([str(path)]) == 0
        first, second = rows(base)
        assert (first.text == LOOP_TEXT) is (not guard)
        assert first.flags == (() if guard else ("token_cap_hit",))
        assert second.flags == ()
    assert json.loads(path.read_text(encoding="utf-8"))["quality_guard"] is False
