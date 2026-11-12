"""Build the blind two-option A/B session between the control and the single best challenger.

What the listener sees (``spans.jsonl``) and which side is which engine (``key.json``) are separate files; the
listening loop never opens the key.
"""

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evals.manifest import Manifest
from evals.normalize import Language
from evals.screen import Span, SpanReading, collect_spans
from evals.store import Layout, dump_jsonl, incomplete_files, now, write_json
from ytverbatim.asr import registry
from ytverbatim.errors import ConfigError

DEFAULT_N = 150
NUMERIC_SHARE = 0.4
CLIP_PAD_S = 1.0
# A shorter clip leaves the listener no context to place the words in.
CLIP_MIN_S = 5.0
SPANS_FILE = "spans.jsonl"
KEY_FILE = "key.json"
DECISIONS_FILE = "decisions.jsonl"

Stratum = Literal["numeric", "other"]
Side = Literal[1, 2]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ShownSpan(_Frozen):
    """Everything the listening loop needs, and nothing that tells the engines apart."""

    id: str
    file: str
    clip_start: float
    clip_end: float
    left: str
    right: str
    options: tuple[str, str]


class KeyEntry(_Frozen):
    id: str
    source: str
    file: str
    language: Language
    stratum: Stratum
    # Population over sample size of the span's (file, stratum) cell; reweights the oversampled strata back.
    weight: float
    challenger_option: Side


class SessionKey(_Frozen):
    control: str
    challenger: str
    seed: int
    n: int
    numeric_share: float
    created: str
    population: dict[str, int]
    entries: tuple[KeyEntry, ...]
    # Candidates whose disputes the spans were drawn from (correction eval only): span ids and cells depend on the set.
    candidates: tuple[str, ...] = ()


def cell_name(file: str, stratum: Stratum) -> str:
    return f"{file}|{stratum}"


def stratum(span: Span) -> Stratum:
    return "numeric" if span.numeric else "other"


def _bounds(reading: SpanReading) -> tuple[float, float] | None:
    """Return a reading's own timing; an empty reading falls back to its neighbouring units."""
    if reading.start is not None and reading.end is not None:
        return reading.start, reading.end
    if reading.before is not None:
        return reading.before, reading.after if reading.after is not None else reading.before
    if reading.after is not None:
        return reading.after, reading.after
    return None


def clip_window(
    readings: Sequence[SpanReading],
    span: Span,
    duration_s: float,
    pad_s: float = CLIP_PAD_S,
    min_s: float = CLIP_MIN_S,
) -> tuple[float, float]:
    """Clip bounds covering every reading's timing, padded, widened to ``min_s`` around the centre, inside the file.

    Taking the union keeps either engine from losing because the clip cut off words only it placed in time.
    """
    bounds = [b for b in map(_bounds, readings) if b is not None] or [(span.start, span.end)]
    start, end = min(b[0] for b in bounds) - pad_s, max(b[1] for b in bounds) + pad_s
    if end - start < min_s:
        centre = (start + end) / 2
        start, end = centre - min_s / 2, centre + min_s / 2
    if start < 0:
        start, end = 0.0, end - start
    if end > duration_s:
        start, end = start - (end - duration_s), duration_s
    return max(0.0, start), min(duration_s, end)


def _split[K: str](total: int, capacity: Mapping[K, int]) -> dict[K, int]:
    """Spread ``total`` evenly over the keys, moving what a key cannot hold to the others."""
    quota = dict.fromkeys(capacity, 0)
    remaining = min(total, sum(capacity.values()))
    open_keys = sorted(k for k in capacity if capacity[k] > 0)
    while remaining and open_keys:
        share, extra = divmod(remaining, len(open_keys))
        for i, key in enumerate(open_keys):
            take = min(share + (i < extra), capacity[key] - quota[key])
            quota[key] += take
            remaining -= take
        open_keys = [k for k in open_keys if quota[k] < capacity[k]]
    return quota


def _strata_quota(total: int, numeric: int, other: int, share: float) -> tuple[int, int]:
    want = round(share * total)
    take_numeric = min(want, numeric)
    take_other = min(total - take_numeric, other)
    take_numeric = min(numeric, total - take_other)
    return take_numeric, take_other


def _round_robin(by_file: Mapping[str, list[Span]], quota: int, rng: random.Random) -> list[Span]:
    queues = {file: list(spans) for file, spans in sorted(by_file.items())}
    for spans in queues.values():
        rng.shuffle(spans)
    order = list(queues)
    rng.shuffle(order)
    picked: list[Span] = []
    while len(picked) < quota and any(queues.values()):
        for file in order:
            if queues[file] and len(picked) < quota:
                picked.append(queues[file].pop())
    return picked


def sample_spans(spans: Sequence[Span], n: int, rng: random.Random, numeric_share: float = NUMERIC_SHARE) -> list[Span]:
    """Equal shares per language, ``numeric_share`` numeric/entity spans within each, spread evenly over files."""
    cells: dict[Language, dict[Stratum, dict[str, list[Span]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for span in spans:
        cells[span.language][stratum(span)][span.file].append(span)
    capacity = {
        lang: sum(len(v) for by_file in strata.values() for v in by_file.values()) for lang, strata in cells.items()
    }
    picked: list[Span] = []
    for language, quota in sorted(_split(n, capacity).items()):
        strata = cells[language]
        available = {s: sum(len(v) for v in strata[s].values()) for s in ("numeric", "other")}
        numeric, other = _strata_quota(quota, available["numeric"], available["other"], numeric_share)
        picked += _round_robin(strata["numeric"], numeric, rng)
        picked += _round_robin(strata["other"], other, rng)
    return picked


def build_session(
    layout: Layout,
    manifest: Manifest,
    control: str,
    challenger: str,
    *,
    n: int = DEFAULT_N,
    seed: int = 0,
    numeric_share: float = NUMERIC_SHARE,
) -> tuple[list[ShownSpan], SessionKey]:
    if control == challenger:
        raise ConfigError("the control and the challenger must differ")
    # Disputes come only from files both engines finished, so a partial run would silently shrink the sample.
    for hypothesis in (control, challenger):
        missing = incomplete_files(layout, manifest, registry.get(hypothesis))
        if missing:
            raise ConfigError(
                f"{hypothesis} is not complete for {len(missing)} of {len(manifest.videos)} files "
                f"({', '.join(missing[:3])}); finish it with `asr --hyp {hypothesis}` first"
            )
    spans = collect_spans(layout, manifest, [control, challenger], pivot=control)
    return session_from_spans(spans, manifest, control, challenger, n=n, seed=seed, numeric_share=numeric_share)


def session_from_spans(
    spans: Sequence[Span],
    manifest: Manifest,
    control: str,
    challenger: str,
    *,
    n: int = DEFAULT_N,
    seed: int = 0,
    numeric_share: float = NUMERIC_SHARE,
    candidates: Sequence[str] = (),
) -> tuple[list[ShownSpan], SessionKey]:
    """Sample ``spans`` into a session; spans that differ only in written form are set aside first.

    Every span must carry a reading for both ``control`` and ``challenger``.
    """
    spans = [s for s in spans if not s.format_only]
    if not spans:
        raise ConfigError(f"no audible disputes between {control} and {challenger}; check that both runs are complete")
    population: dict[str, int] = defaultdict(int)
    for span in spans:
        population[cell_name(span.file, stratum(span))] += 1
    rng = random.Random(seed)
    picked = sample_spans(spans, n, rng, numeric_share)
    sampled: dict[str, int] = defaultdict(int)
    for span in picked:
        sampled[cell_name(span.file, stratum(span))] += 1
    rng.shuffle(picked)
    durations = {video.id: video.audio_duration_s for video in manifest.videos}
    shown: list[ShownSpan] = []
    entries: list[KeyEntry] = []
    for number, span in enumerate(picked, start=1):
        side: Side = 1 if rng.random() < 0.5 else 2  # noqa: PLR2004 - a fair coin
        control_text, challenger_text = span.reading(control).text, span.reading(challenger).text
        options = (challenger_text, control_text) if side == 1 else (control_text, challenger_text)
        start, end = clip_window((span.reading(control), span.reading(challenger)), span, durations[span.file])
        span_id = f"{number:03d}"
        shown.append(
            ShownSpan(
                id=span_id,
                file=span.file,
                clip_start=round(start, 3),
                clip_end=round(end, 3),
                left=span.left,
                right=span.right,
                options=options,
            )
        )
        cell = cell_name(span.file, stratum(span))
        entries.append(
            KeyEntry(
                id=span_id,
                source=span.id,
                file=span.file,
                language=span.language,
                stratum=stratum(span),
                weight=population[cell] / sampled[cell],
                challenger_option=side,
            )
        )
    key = SessionKey(
        control=control,
        challenger=challenger,
        seed=seed,
        n=n,
        numeric_share=numeric_share,
        created=now(),
        population=dict(sorted(population.items())),
        entries=tuple(entries),
        candidates=tuple(sorted(candidates)),
    )
    return shown, key


def load_shown(directory: Path) -> list[ShownSpan]:
    path = directory / SPANS_FILE
    return [ShownSpan.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_key(directory: Path) -> SessionKey:
    return SessionKey.model_validate_json((directory / KEY_FILE).read_text(encoding="utf-8"))


def write_session(layout: Layout, name: str, shown: Sequence[ShownSpan], key: SessionKey) -> bool:
    """Write a new session; return False when the same session already exists, and refuse a different one."""
    directory = layout.session_dir(name)
    if (directory / SPANS_FILE).exists():
        same = (
            load_shown(directory) == list(shown)
            and load_key(directory).model_copy(update={"created": key.created}) == key
        )
        if same:
            return False
        raise ConfigError(f"session {name!r} already exists with different spans; pick another --name")
    dump_jsonl(directory / SPANS_FILE, shown)
    write_json(directory / KEY_FILE, key)
    return True
