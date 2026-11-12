import math
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_MAX_CHUNK_S = 30.0
DEFAULT_TAIL_FRACTION = 0.4
DEFAULT_MIN_PAUSE_S = 0.1
DEFAULT_WINDOW_PAUSE_S = 0.8
DEFAULT_MAX_WINDOW_S = 300.0
FRAME_S = 0.1


@dataclass(frozen=True, slots=True)
class Region:
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Chunk:
    """One contiguous slice of audio. `forced_cut` means its end fell inside continuous speech."""

    start: float
    end: float
    forced_cut: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Window:
    chunks: tuple[Chunk, ...]

    @property
    def start(self) -> float:
        return self.chunks[0].start

    @property
    def end(self) -> float:
        return self.chunks[-1].end

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_chunks(
    regions: Sequence[Region],
    duration: float,
    *,
    max_chunk_s: float = DEFAULT_MAX_CHUNK_S,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    min_pause_s: float = DEFAULT_MIN_PAUSE_S,
    frame_energy: Sequence[float] | None = None,
    frame_s: float = FRAME_S,
) -> list[Chunk]:
    """Cut speech into chunks of at most `max_chunk_s`, each a single slice of the audio.

    Regions are expected to carry their own padding (the VAD's `speech_pad_ms`), so chunk edges sit on region
    edges. A chunk ends in the longest pause of the last `tail_fraction` of its window, else in the longest pause
    anywhere in the window. Only a speech run with no pause of at least `min_pause_s` inside the window is cut
    mid-speech, at the lowest-energy frame of that tail; `frame_energy` holds one value per `frame_s`, and
    without it the cut lands at the window end.
    """
    if max_chunk_s <= 0 or not 0 < tail_fraction <= 1:
        raise ValueError("max_chunk_s must be positive and tail_fraction within (0, 1]")
    speech = _normalize(regions, duration)
    chunks: list[Chunk] = []
    i = 0
    start = 0.0
    while i < len(speech):
        start = max(start, speech[i].start)
        limit = start + max_chunk_s
        if speech[-1].end <= limit:
            chunks.append(Chunk(start, speech[-1].end))
            break
        tail_start = limit - tail_fraction * max_chunk_s
        last = _pause_cut(speech, i, limit, tail_start, min_pause_s)
        if last is not None:
            chunks.append(Chunk(start, speech[last].end))
            i = last + 1
            continue
        end = _forced_end(start, limit, tail_start, frame_energy, frame_s)
        chunks.append(Chunk(start, end, forced_cut=True))
        while speech[i].end <= end:
            i += 1
        start = end
    return chunks


def alignment_windows(
    chunks: Sequence[Chunk],
    *,
    min_pause_s: float = DEFAULT_WINDOW_PAUSE_S,
    max_window_s: float = DEFAULT_MAX_WINDOW_S,
) -> list[Window]:
    """Group consecutive chunks into windows no word can straddle.

    A window ends at a pause of at least `min_pause_s`. A group longer than `max_window_s` is cut at its latest
    boundary that is not a forced cut; with none available the window exceeds the cap rather than split a word.
    """
    windows: list[Window] = []
    first = 0
    fallback: int | None = None
    j = 1
    while j < len(chunks):
        prev, cur = chunks[j - 1], chunks[j]
        if not prev.forced_cut:
            if cur.start - prev.end >= min_pause_s:
                windows.append(Window(tuple(chunks[first:j])))
                first, fallback = j, None
                j += 1
                continue
            fallback = j
        if fallback is not None and cur.end - chunks[first].start > max_window_s:
            windows.append(Window(tuple(chunks[first:fallback])))
            first, fallback = fallback, None
            j = first + 1
            continue
        j += 1
    if first < len(chunks):
        windows.append(Window(tuple(chunks[first:])))
    return windows


def _normalize(regions: Sequence[Region], duration: float) -> list[Region]:
    clipped = [Region(max(r.start, 0.0), min(r.end, duration)) for r in regions]
    merged: list[Region] = []
    for region in sorted((r for r in clipped if r.end > r.start), key=lambda r: (r.start, r.end)):
        if merged and region.start <= merged[-1].end:
            merged[-1] = Region(merged[-1].start, max(merged[-1].end, region.end))
        else:
            merged.append(region)
    return merged


def _pause_cut(speech: Sequence[Region], i: int, limit: float, tail_start: float, min_pause_s: float) -> int | None:
    """Index of the last region of a chunk that starts at `speech[i]`, or None when no pause fits the window."""
    in_tail: list[tuple[float, float, int]] = []
    anywhere: list[tuple[float, float, int]] = []
    k = i
    while k + 1 < len(speech) and speech[k].end <= limit:
        gap = speech[k + 1].start - speech[k].end
        if gap >= min_pause_s:
            # Longest pause wins; on a tie the later one, which makes the chunk longer.
            candidate = (gap, speech[k].end, k)
            anywhere.append(candidate)
            if speech[k + 1].start > tail_start:
                in_tail.append(candidate)
        k += 1
    pool = in_tail or anywhere
    return max(pool)[2] if pool else None


def _forced_end(
    start: float, limit: float, tail_start: float, frame_energy: Sequence[float] | None, frame_s: float
) -> float:
    if frame_energy is None:
        return limit
    first = max(math.floor(tail_start / frame_s), 0)
    best: tuple[float, int] | None = None
    for k in range(first, min(math.floor(limit / frame_s) + 1, len(frame_energy))):
        middle = (k + 0.5) * frame_s
        if start < middle <= limit and tail_start <= middle:
            key = (frame_energy[k], -k)
            if best is None or key < best:
                best = key
    return limit if best is None else (-best[1] + 0.5) * frame_s
