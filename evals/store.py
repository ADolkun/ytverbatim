"""Where every eval artifact lives under ``$YTVERBATIM_EVAL_DIR``, and how hypotheses are read back as timed units.

Nothing here writes into the repository: plans, runs, screening, judge verdicts, A/B sessions and reports all stay
in the private eval directory.
"""

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz.distance import Levenshtein

from evals.consensus import Segment, TimedUnit, timed_units
from evals.manifest import Manifest, Video
from evals.normalize import Language, View, to_units
from ytverbatim import atomic
from ytverbatim.asr.chunks import Chunk
from ytverbatim.asr.models import HYPOTHESES_FILE, ChunkHypothesis, RunMetrics
from ytverbatim.asr.protocol import EngineDef
from ytverbatim.errors import ConfigError

CAPTION_HYPOTHESIS = "yt-auto"
REFERENCE = "ref-manual"
HISTORY_FILE = "runs.jsonl"

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Plan(_Frozen):
    """The shared audio plan of one file: every chunked hypothesis decodes exactly these chunks."""

    file: str
    duration_s: Seconds
    speech: tuple[tuple[float, float], ...]
    chunks: tuple[Chunk, ...]
    # Alignment windows, widened to meet halfway across the pauses between them so every unit of every hypothesis,
    # including text decoded inside a pause, belongs to exactly one window.
    windows: tuple[tuple[float, float], ...]
    max_chunk_s: float
    vad: dict[str, float]
    silero_revision: str


class RunRecord(_Frozen):
    """One `asr` invocation; the worker's own `metrics.json` only describes the latest one."""

    at: str
    files: tuple[str, ...]
    file_seconds: Seconds
    metrics: RunMetrics
    children_max_rss_bytes: int
    elapsed_s: Seconds


@dataclass(frozen=True)
class Layout:
    root: Path

    @property
    def plans(self) -> Path:
        return self.root / "plans"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def screen(self) -> Path:
        return self.root / "screen"

    @property
    def judge(self) -> Path:
        return self.root / "judge"

    @property
    def ab(self) -> Path:
        return self.root / "ab"

    @property
    def report(self) -> Path:
        return self.root / "report"

    def plan_path(self, file_id: str) -> Path:
        return self.plans / f"{file_id}.json"

    def run_dir(self, hypothesis: str) -> Path:
        return self.runs / hypothesis

    def file_run_dir(self, hypothesis: str, file_id: str) -> Path:
        """One worker output directory per file: its fingerprint then covers exactly one job and stays stable."""
        return self.run_dir(hypothesis) / file_id

    def rows_path(self, hypothesis: str, file_id: str) -> Path:
        return self.file_run_dir(hypothesis, file_id) / HYPOTHESES_FILE

    def session_dir(self, name: str) -> Path:
        return self.ab / name

    def wav(self, video: Video) -> Path:
        return self.root / video.audio


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def save_plan(layout: Layout, plan: Plan) -> None:
    layout.plans.mkdir(parents=True, exist_ok=True)
    atomic.write_text(layout.plan_path(plan.file), plan.model_dump_json(indent=2) + "\n")


def load_plan(layout: Layout, file_id: str) -> Plan | None:
    path = layout.plan_path(file_id)
    return Plan.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None


def require_plan(layout: Layout, file_id: str) -> Plan:
    plan = load_plan(layout, file_id)
    if plan is None:
        raise ConfigError(f"no audio plan for {file_id}; run `vad` first")
    return plan


def widened_windows(spans: Sequence[tuple[float, float]], duration_s: float) -> tuple[tuple[float, float], ...]:
    if not spans:
        return ((0.0, duration_s),)
    cuts = [(spans[i][1] + spans[i + 1][0]) / 2 for i in range(len(spans) - 1)]
    edges = [0.0, *cuts, max(duration_s, spans[-1][1])]
    return tuple((edges[i], edges[i + 1]) for i in range(len(spans)))


def read_rows(path: Path) -> list[ChunkHypothesis]:
    """Every complete line; a line still being written by a running worker is ignored."""
    if not path.exists():
        return []
    data = path.read_text(encoding="utf-8")
    return [ChunkHypothesis.model_validate_json(line) for line in data[: data.rfind("\n") + 1].splitlines() if line]


def rows_by_file(layout: Layout, hypothesis: str) -> dict[str, list[ChunkHypothesis]]:
    grouped: dict[str, list[ChunkHypothesis]] = defaultdict(list)
    for path in sorted(layout.run_dir(hypothesis).glob(f"*/{HYPOTHESES_FILE}")):
        for row in read_rows(path):
            grouped[row.file].append(row)
    return {file: sorted(rows, key=lambda r: r.chunk) for file, rows in grouped.items()}


def write_rows(layout: Layout, hypothesis: str, rows: Iterable[ChunkHypothesis]) -> None:
    grouped: dict[str, list[ChunkHypothesis]] = defaultdict(list)
    for row in rows:
        grouped[row.file].append(row)
    for file_id, file_rows in grouped.items():
        layout.file_run_dir(hypothesis, file_id).mkdir(parents=True, exist_ok=True)
        atomic.write_text(layout.rows_path(hypothesis, file_id), "".join(r.model_dump_json() + "\n" for r in file_rows))


def expected_chunks(definition: EngineDef, plan: Plan) -> int:
    return 1 if definition.spec.max_chunk_s is None else len(plan.chunks)


def file_complete(definition: EngineDef, plan: Plan, rows: Sequence[ChunkHypothesis]) -> bool:
    expected = expected_chunks(definition, plan)
    done = {row.chunk for row in rows}
    if done - set(range(expected)):
        raise ConfigError(
            f"{definition.spec.id} output for {plan.file} has chunks outside the current plan; it was decoded with a "
            "different plan, so move its run directory aside"
        )
    return len(done) == expected


def incomplete_files(layout: Layout, manifest: Manifest, definition: EngineDef) -> list[str]:
    """Manifest files with no plan or without this hypothesis's full output."""
    rows = rows_by_file(layout, definition.spec.id)
    missing = []
    for video in manifest.videos:
        plan = load_plan(layout, video.id)
        if plan is None or not file_complete(definition, plan, rows.get(video.id, [])):
            missing.append(video.id)
    return missing


def append_history(layout: Layout, hypothesis: str, record: RunRecord) -> None:
    path = layout.run_dir(hypothesis) / HISTORY_FILE
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")


def read_history(layout: Layout, hypothesis: str) -> list[RunRecord]:
    path = layout.run_dir(hypothesis) / HISTORY_FILE
    if not path.exists():
        return []
    return [RunRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def hypotheses_on_disk(layout: Layout) -> list[str]:
    if not layout.runs.is_dir():
        return []
    return sorted(p.name for p in layout.runs.iterdir() if p.name != REFERENCE and any(p.glob(f"*/{HYPOTHESES_FILE}")))


def row_units(
    row: ChunkHypothesis, language: Language, view: View, speech: Sequence[tuple[float, float]] = ()
) -> list[TimedUnit]:
    """Units of one row with the best timing it offers: word times when present, else spread over the chunk.

    The text is normalized as a whole, never word by word, because standalone fillers and quantities only read
    correctly in context; word times are then carried over by aligning the two unit sequences. Without word times
    the units spread over the VAD ``speech`` inside the chunk.
    """
    if not row.words:
        return timed_units([Segment(row.text, row.start, row.end)], language, view, speech)
    whole = to_units(row.text, language, view)
    timed = timed_units([Segment(w.text, w.start, w.end) for w in row.words], language, view)
    times: list[tuple[float, float] | None] = [None] * len(whole)
    for op in Levenshtein.opcodes([u.text for u in timed], whole):
        if op.tag in ("equal", "replace"):
            for offset in range(min(op.src_end - op.src_start, op.dest_end - op.dest_start)):
                unit = timed[op.src_start + offset]
                times[op.dest_start + offset] = (unit.start_s, unit.end_s)
    out: list[TimedUnit] = []
    cursor = row.start
    for text, span in zip(whole, times, strict=True):
        start, end = span if span is not None else (cursor, cursor)
        out.append(TimedUnit(text, start, end))
        cursor = end
    return out


def hypothesis_units(
    rows: Sequence[ChunkHypothesis], language: Language, view: View, speech: Sequence[tuple[float, float]] = ()
) -> list[TimedUnit]:
    return [unit for row in rows for unit in row_units(row, language, view, speech)]


def row_text(rows: Sequence[ChunkHypothesis]) -> str:
    return "\n".join(row.text for row in rows)


def videos(manifest: Manifest, selected: Sequence[str] = ()) -> list[Video]:
    if not selected:
        return list(manifest.videos)
    known = {video.id: video for video in manifest.videos}
    unknown = [file for file in selected if file not in known]
    if unknown:
        raise ConfigError(f"not in the manifest: {', '.join(unknown)}")
    return [known[file] for file in selected]


def dump_jsonl(path: Path, items: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic.write_text(path, "".join(item.model_dump_json() + "\n" for item in items))


def write_json(path: Path, item: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic.write_text(path, item.model_dump_json(indent=2) + "\n")


def load_json_lines(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
