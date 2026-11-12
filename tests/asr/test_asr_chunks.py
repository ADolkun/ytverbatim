import itertools

import pytest

from ytverbatim.asr.chunks import Chunk, Region, alignment_windows, plan_chunks


def regions(*pairs: tuple[float, float]) -> list[Region]:
    return [Region(a, b) for a, b in pairs]


def test_everything_that_fits_is_one_chunk() -> None:
    assert plan_chunks(regions((1.0, 5.0), (5.5, 12.0)), 20.0) == [Chunk(1.0, 12.0)]


def test_no_speech_gives_no_chunks() -> None:
    assert plan_chunks([], 60.0) == []
    assert plan_chunks(regions((5.0, 5.0)), 60.0) == []


def test_cut_is_in_the_longest_pause_of_the_tail() -> None:
    # Window is 0-30 with its tail at 18-30. Pauses: 0.5 s at 10 (outside the tail), 0.3 s at 20, 0.9 s at 26.
    speech = regions((0.0, 10.0), (10.5, 20.0), (20.3, 26.0), (26.9, 40.0))
    chunks = plan_chunks(speech, 60.0)
    assert chunks == [Chunk(0.0, 26.0), Chunk(26.9, 40.0)]


def test_tie_between_pauses_prefers_the_later_one() -> None:
    speech = regions((0.0, 20.0), (20.5, 25.0), (25.5, 40.0))
    assert plan_chunks(speech, 60.0)[0] == Chunk(0.0, 25.0)


def test_pause_outside_the_tail_is_used_before_cutting_speech() -> None:
    speech = regions((0.0, 5.0), (6.0, 50.0))
    chunks = plan_chunks(speech, 60.0)
    assert chunks[0] == Chunk(0.0, 5.0)
    assert chunks[1].forced_cut


def test_chunks_never_exceed_the_cap_and_never_overlap() -> None:
    speech = regions(*[(i * 7.0, i * 7.0 + 6.0) for i in range(40)])
    chunks = plan_chunks(speech, 300.0)
    assert all(chunk.duration <= 30.0 for chunk in chunks)
    assert all(a.end <= b.start for a, b in itertools.pairwise(chunks))
    assert not any(chunk.forced_cut for chunk in chunks)


def test_long_region_is_force_cut_at_the_quietest_tail_frame() -> None:
    energy = [1.0] * 700
    energy[250] = 0.01  # 25.0-25.1 s, inside the 18-30 s tail
    energy[50] = 0.0  # quieter but outside the tail
    chunks = plan_chunks(regions((0.0, 70.0)), 70.0, frame_energy=energy)
    assert chunks[0] == Chunk(0.0, pytest.approx(25.05), forced_cut=True)
    assert all(chunk.duration <= 30.0 for chunk in chunks)
    assert [chunk.forced_cut for chunk in chunks] == [True, True, False]
    assert all(a.end == b.start for a, b in itertools.pairwise(chunks))


def test_forced_cut_ties_go_to_the_later_frame() -> None:
    chunks = plan_chunks(regions((0.0, 45.0)), 45.0, frame_energy=[0.5] * 450)
    assert chunks[0].end == pytest.approx(29.95)


def test_forced_cut_without_energy_lands_at_the_window_end() -> None:
    chunks = plan_chunks(regions((2.0, 50.0)), 50.0)
    assert chunks == [Chunk(2.0, 32.0, forced_cut=True), Chunk(32.0, 50.0)]


def test_sub_minimum_pauses_are_not_cut_points() -> None:
    speech = regions((0.0, 20.0), (20.05, 40.0))
    chunks = plan_chunks(speech, 40.0)
    assert chunks[0].forced_cut


def test_regions_are_sorted_clipped_and_merged() -> None:
    speech = regions((10.0, 12.0), (-3.0, 4.0), (3.0, 6.0), (11.0, 99.0))
    assert plan_chunks(speech, 20.0) == [Chunk(0.0, 20.0)]


def test_plan_is_deterministic() -> None:
    speech = regions(*[(i * 3.3, i * 3.3 + 2.7) for i in range(200)])
    energy = [((i * 7919) % 101) / 100 for i in range(700)]
    assert plan_chunks(speech, 700.0, frame_energy=energy) == plan_chunks(list(speech), 700.0, frame_energy=energy)


def test_custom_cap() -> None:
    chunks = plan_chunks(regions((0.0, 4.0), (4.5, 8.0), (8.4, 12.0)), 12.0, max_chunk_s=9.0)
    assert [(c.start, c.end) for c in chunks] == [(0.0, 8.0), (8.4, 12.0)]


@pytest.mark.parametrize(("cap", "tail"), [(0.0, 0.4), (30.0, 0.0), (30.0, 1.5)])
def test_invalid_parameters(cap: float, tail: float) -> None:
    with pytest.raises(ValueError, match="max_chunk_s"):
        plan_chunks(regions((0.0, 1.0)), 1.0, max_chunk_s=cap, tail_fraction=tail)


def chunk_run(*pairs: tuple[float, float], forced: frozenset[int] = frozenset()) -> list[Chunk]:
    return [Chunk(a, b, i in forced) for i, (a, b) in enumerate(pairs)]


def test_windows_split_at_long_pauses() -> None:
    chunks = chunk_run((0, 20), (20.3, 40), (41.0, 60), (60.5, 80), (85.0, 90))
    windows = alignment_windows(chunks)
    assert [(w.start, w.end) for w in windows] == [(0, 40), (41.0, 80), (85.0, 90)]
    assert [len(w.chunks) for w in windows] == [2, 2, 1]
    assert windows[0].duration == 40


def test_windows_cap_cuts_at_the_latest_boundary_that_fits() -> None:
    chunks = chunk_run(*[(i * 30.5, i * 30.5 + 30.0) for i in range(12)])
    windows = alignment_windows(chunks, max_window_s=100.0)
    assert [len(w.chunks) for w in windows] == [3, 3, 3, 3]
    assert all(w.duration <= 100.0 for w in windows)
    assert [c for w in windows for c in w.chunks] == chunks


def test_windows_never_cut_at_a_forced_cut() -> None:
    chunks = chunk_run((0, 30), (30, 60), (60, 90), (90.5, 120), (120.5, 150), forced=frozenset({0, 1, 2}))
    windows = alignment_windows(chunks, max_window_s=100.0)
    assert [len(w.chunks) for w in windows] == [4, 1]


def test_windows_exceed_the_cap_when_no_boundary_is_safe() -> None:
    chunks = chunk_run((0, 30), (30, 60), (60, 90), (90, 120), forced=frozenset({0, 1, 2}))
    assert [len(w.chunks) for w in alignment_windows(chunks, max_window_s=50.0)] == [4]


def test_windows_of_nothing() -> None:
    assert alignment_windows([]) == []
