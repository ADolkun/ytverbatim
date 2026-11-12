"""A tiny synthetic eval directory: two silent files, plans, metadata, caption tracks and hypothesis rows."""

import json
import re
import wave
from collections.abc import Sequence
from pathlib import Path

import pytest

from evals.manifest import EVAL_DIR_ENV, Manifest, load_manifest
from evals.store import Layout, Plan, save_plan, widened_windows, write_rows
from ytverbatim.asr.chunks import Chunk
from ytverbatim.asr.models import ChunkHypothesis
from ytverbatim.contract.models import Word
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Usage

ZH = "zhfile00001"
EN = "enfile00001"
SECONDS = 20.0
CHUNKS = (Chunk(0.0, 4.0), Chunk(5.0, 9.0), Chunk(11.0, 15.0), Chunk(16.0, 19.0))


def _wav(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\x00\x00" * round(seconds * 16_000))


def _json3(events: Sequence[tuple[int, int, Sequence[tuple[str, int]]]]) -> str:
    return json.dumps(
        {
            "events": [
                {"tStartMs": start, "dDurationMs": duration, "segs": [{"utf8": t, "tOffsetMs": o} for t, o in segs]}
                for start, duration, segs in events
            ]
        }
    )


def _video(video_id: str, language: str, captions: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "channel": "Example Channel",
        "title": "Example title",
        "duration_s": int(SECONDS),
        "language": language,
        "access": "public",
        "audio": f"audio/{video_id}.wav",
        "audio_sha256": "0" * 64,
        "audio_format": "wav pcm_s16le 16000 Hz mono",
        "audio_duration_s": SECONDS,
        "upload_date": "20250101",
        "chapters": "",
        "chapter_count": 0,
        "captions": {"listed": [], "listed_note": "synthetic", "obtained": captions},
        "source": "synthetic",
        "notes": "",
        "meta": f"meta/{video_id}.info.json",
    }


def make_eval(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Layout, Manifest]:
    monkeypatch.setenv(EVAL_DIR_ENV, str(root))
    zh_captions: list[dict[str, object]] = [
        {"kind": "manual", "lang": "zh", "format": "json3", "path": f"captions/{ZH}.manual.zh.json3", "bytes": 1}
    ]
    en_captions: list[dict[str, object]] = [
        {"kind": "auto", "lang": "en-orig", "format": "json3", "path": f"captions/{EN}.auto.en-orig.json3", "bytes": 1}
    ]
    manifest = {
        "created": "2025-01-01",
        "audio_format": "wav",
        "videos": [_video(ZH, "zh+en", zh_captions), _video(EN, "en", en_captions)],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for video_id, language in ((ZH, "zh"), (EN, "en")):
        _wav(root / "audio" / f"{video_id}.wav", SECONDS)
        meta = root / "meta" / f"{video_id}.info.json"
        meta.parent.mkdir(parents=True, exist_ok=True)
        info = {"id": video_id, "title": "Example NVDA talk", "description": "About $NVDA", "language": language}
        meta.write_text(json.dumps(info), encoding="utf-8")
    captions = root / "captions"
    captions.mkdir(exist_ok=True)
    (captions / f"{ZH}.manual.zh.json3").write_text(
        _json3([(0, 4000, [("我们今天买了股票", 0)]), (5000, 4000, [("然后卖出了一百股", 0)])]), encoding="utf-8"
    )
    (captions / f"{EN}.auto.en-orig.json3").write_text(
        _json3(
            [(0, 4000, [("we", 0), (" bought", 500), (" stock", 1000)]), (5000, 4000, [("then", 0), (" sold", 800)])]
        ),
        encoding="utf-8",
    )
    layout = Layout(root)
    for video_id in (ZH, EN):
        windows = [(c.start, c.end) for c in CHUNKS]
        save_plan(
            layout,
            Plan(
                file=video_id,
                duration_s=SECONDS,
                speech=tuple((c.start, c.end) for c in CHUNKS),
                chunks=CHUNKS,
                windows=widened_windows(windows, SECONDS),
                max_chunk_s=30.0,
                vad={"threshold": 0.5},
                silero_revision="0" * 40,
            ),
        )
    return layout, load_manifest(root)


def chunk_rows(
    hypothesis: str, file: str, texts: Sequence[str], flags: Sequence[tuple[str, ...]] = ()
) -> list[ChunkHypothesis]:
    return [
        ChunkHypothesis(
            hypothesis=hypothesis,
            file=file,
            chunk=i,
            start=CHUNKS[i].start,
            end=CHUNKS[i].end,
            text=text,
            flags=flags[i] if i < len(flags) else (),  # type: ignore[arg-type]
        )
        for i, text in enumerate(texts)
    ]


def whole_file_rows(hypothesis: str, file: str, texts: Sequence[str]) -> list[ChunkHypothesis]:
    """One whole-file row whose words sit exactly in the plan's chunks, as Whisper's word times would."""
    words: list[Word] = []
    for chunk, text in zip(CHUNKS, texts, strict=False):
        pieces = text.split() if " " in text else list(text)
        step = chunk.duration / max(len(pieces), 1)
        words += [
            Word(
                text=piece if " " not in text else f" {piece}",
                start=chunk.start + i * step,
                end=chunk.start + (i + 1) * step,
            )
            for i, piece in enumerate(pieces)
        ]
    return [
        ChunkHypothesis(
            hypothesis=hypothesis,
            file=file,
            chunk=0,
            start=0.0,
            end=SECONDS,
            text=" ".join(texts) if " " in "".join(texts) else "".join(texts),
            words=tuple(words),
        )
    ]


ZH_CONTROL = ("我们今天买了股票", "然后卖出了一百股", "看NVDA的财报", "大家好")
ZH_CHALLENGER = ("我们今天买了股票", "然后卖出了二百股", "看AMD的财报", "大家好呀")
EN_CONTROL = ("we bought the stock", "then we sold it", "the cat sat down", "good night")
EN_CHALLENGER = ("we bought a stock", "then we sold it", "the cap sat down", "good night all")


def write_pair(layout: Layout, control: str = "w-prod", challenger: str = "q-base") -> None:
    write_rows(
        layout,
        control,
        [*whole_file_rows(control, ZH, ZH_CONTROL), *whole_file_rows(control, EN, EN_CONTROL)],
    )
    write_rows(
        layout,
        challenger,
        [*chunk_rows(challenger, ZH, ZH_CHALLENGER), *chunk_rows(challenger, EN, EN_CHALLENGER)],
    )


class FakeJudge:
    """A blind judge that takes the first of ``prefer`` among a span's options, else the first option listed."""

    id = "fake:judge"

    def __init__(self, prefer: Sequence[str] = ()) -> None:
        self.prefer = tuple(prefer)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        decisions = []
        for block in request.user.split("\n\n")[1:]:
            label = re.match(r"\[(s\d+)\]", block)
            assert label, block
            options = [m.group(2) for m in re.finditer(r"^ {2}(\d+)\. (.*)$", block, re.MULTILINE)]
            chosen = next((p for p in self.prefer if p in options), options[0])
            decisions.append({"span": label.group(1), "choice": options.index(chosen) + 1, "reason": "fluent"})
        return LLMResponse(text=json.dumps({"decisions": decisions}), usage=Usage(10, 5), elapsed_s=0.1)
