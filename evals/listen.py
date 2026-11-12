"""The owner's blind A/B loop in the terminal: hear a clip, press one key, repeat.

Every decision is appended and synced before the next clip, so quitting or a crash loses nothing. The loop reads
only ``spans.jsonl``; which option belongs to which engine stays in ``key.json``.
"""

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evals.judge import EMPTY_READING
from evals.manifest import Manifest
from evals.session import DECISIONS_FILE, SPANS_FILE, ShownSpan, load_shown
from evals.store import Layout, now
from ytverbatim import proc
from ytverbatim.errors import ConfigError, MediaValidationError

AFPLAY = "/usr/bin/afplay"
SECONDS_PER_SPAN = 6
_CLIP_TIMEOUT_S = 60.0
_PLAY_TIMEOUT_S = 300.0

Choice = Literal["1", "2", "both", "neither", "skip"]
CHOICE_KEYS: dict[str, Choice] = {"1": "1", "2": "2", "b": "both", "0": "neither", "s": "skip"}

Player = Callable[[Path], None]
Cutter = Callable[[Path, float, float, Path], None]

HELP = """Blind A/B listening: {remaining} of {total} spans left, about {minutes} min at ~{per_span} s each.
Each clip plays once; then press ONE key:
  1 / 2  the reading you heard        b  both fine (equivalent)     0  both wrong
  r      replay                       s  skip this span             q  save and quit
Decisions save immediately; run the same command again to resume.
"""


class LoggedDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    span: str
    choice: Choice
    at: str


@dataclass(frozen=True)
class ListenResult:
    decided: int
    remaining: int
    quit: bool


def ffmpeg_cut(wav: Path, start: float, end: float, out: Path) -> None:
    argv = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-t", f"{max(end - start, 0.1):.3f}", "-i", str(wav),
        "-ac", "1", "-ar", "16000", str(out),
    ]  # fmt: skip
    result = proc.run(argv, timeout_s=_CLIP_TIMEOUT_S)
    if result.returncode != 0 or not out.exists():
        raise MediaValidationError(f"ffmpeg could not cut a clip ({result.returncode}): {result.stderr.strip()[:300]}")


def afplay(clip: Path) -> None:
    proc.run([AFPLAY, str(clip)], timeout_s=_PLAY_TIMEOUT_S)


def read_decisions(directory: Path) -> list[LoggedDecision]:
    path = directory / DECISIONS_FILE
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [LoggedDecision.model_validate_json(line) for line in lines if line.strip()]


def render(span: ShownSpan, position: int, total: int) -> str:
    left = f"…{span.left} " if span.left else ""
    right = f" {span.right}…" if span.right else ""
    one, two = (option or EMPTY_READING for option in span.options)
    return f"\n[{position}/{total}]\n{left}⟪ ? ⟫{right}\n  1  {one}\n  2  {two}"


def _append(path: Path, decision: LoggedDecision) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(decision.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def listen(
    layout: Layout,
    manifest: Manifest,
    name: str,
    *,
    getchar: Callable[[], str],
    echo: Callable[[str], None],
    player: Player = afplay,
    cutter: Cutter = ffmpeg_cut,
) -> ListenResult:
    directory = layout.session_dir(name)
    if not (directory / SPANS_FILE).exists():
        raise ConfigError(f"no A/B session {name!r}; build it with `session` first")
    spans = load_shown(directory)
    done = {d.span for d in read_decisions(directory)}
    pending = [s for s in spans if s.id not in done]
    wavs = {video.id: layout.wav(video) for video in manifest.videos}
    echo(
        HELP.format(
            remaining=len(pending),
            total=len(spans),
            minutes=round(len(pending) * SECONDS_PER_SPAN / 60),
            per_span=SECONDS_PER_SPAN,
        )
    )
    decided = 0
    with tempfile.TemporaryDirectory(prefix="ytverbatim-ab-") as scratch:
        for offset, span in enumerate(pending):
            clip = Path(scratch) / f"{span.id}.wav"
            cutter(wavs[span.file], span.clip_start, span.clip_end, clip)
            echo(render(span, len(done) + offset + 1, len(spans)))
            player(clip)
            while True:
                key = getchar().strip().lower()
                if key == "r":
                    player(clip)
                elif key == "q":
                    echo("saved; run listen again to resume")
                    return ListenResult(decided, len(pending) - offset, quit=True)
                elif key in CHOICE_KEYS:
                    _append(directory / DECISIONS_FILE, LoggedDecision(span=span.id, choice=CHOICE_KEYS[key], at=now()))
                    decided += 1
                    break
                else:
                    echo("keys: 1 2 b 0 r s q")
    echo(f"session {name!r} complete: {len(spans)} spans")
    return ListenResult(decided, 0, quit=False)
