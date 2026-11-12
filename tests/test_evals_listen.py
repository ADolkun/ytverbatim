from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from evals_fixture import make_eval, write_pair

from evals import listen
from evals.manifest import Manifest
from evals.session import build_session, write_session
from evals.store import Layout
from ytverbatim import proc
from ytverbatim.asr import registry
from ytverbatim.errors import ConfigError


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Layout, Manifest]:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    write_pair(layout)
    write_session(layout, "main", *build_session(layout, manifest, "w-prod", "q-base", n=6, seed=2))
    return layout, manifest


class Harness:
    def __init__(self, keys: str) -> None:
        self._keys: Iterator[str] = iter(keys)
        self.played: list[Path] = []
        self.cut: list[tuple[float, float]] = []
        self.output: list[str] = []

    def getchar(self) -> str:
        return next(self._keys)

    def player(self, clip: Path) -> None:
        assert clip.exists()
        self.played.append(clip)

    def cutter(self, wav: Path, start: float, end: float, out: Path) -> None:
        assert wav.exists()
        self.cut.append((start, end))
        out.write_bytes(b"RIFF")

    def run(self, layout: Layout, manifest: Manifest) -> listen.ListenResult:
        return listen.listen(
            layout, manifest, "main", getchar=self.getchar, echo=self.output.append, player=self.player,
            cutter=self.cutter,
        )  # fmt: skip


def test_scripted_keys_record_decisions_and_resume(session: tuple[Layout, Manifest]) -> None:
    layout, manifest = session
    first = Harness("1x2rq")
    result = first.run(layout, manifest)
    assert result.quit
    assert result.decided == 2
    assert len(first.played) == 4  # three clips, one replay
    assert [d.choice for d in listen.read_decisions(layout.session_dir("main"))] == ["1", "2"]
    assert any("keys: 1 2 b 0 r s q" in line for line in first.output)
    assert "press ONE key" in first.output[0]

    second = Harness("b0s1")
    rest = second.run(layout, manifest)
    assert not rest.quit
    assert rest.decided == 4
    decisions = listen.read_decisions(layout.session_dir("main"))
    assert [d.choice for d in decisions] == ["1", "2", "both", "neither", "skip", "1"]
    assert len({d.span for d in decisions}) == 6
    assert "[3/6]" in second.output[1]
    assert "4 of 6 spans left" in second.output[0]


def test_nothing_on_screen_names_an_engine(session: tuple[Layout, Manifest]) -> None:
    layout, manifest = session
    harness = Harness("121212")
    harness.run(layout, manifest)
    screen = "\n".join(harness.output).casefold()
    for name in [*registry.HYPOTHESES, "whisper", "qwen", "control", "challenger"]:
        assert name.casefold() not in screen


def test_clips_cover_the_session_windows(session: tuple[Layout, Manifest]) -> None:
    layout, manifest = session
    harness = Harness("111111")
    harness.run(layout, manifest)
    spans = [s for s in (layout.session_dir("main") / "spans.jsonl").read_text().splitlines()]
    assert len(harness.cut) == len(spans) == 6
    assert all(end > start for start, end in harness.cut)


def test_listen_needs_a_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    with pytest.raises(ConfigError, match="build it with `session`"):
        Harness("").run(layout, manifest)


def test_ffmpeg_and_afplay_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, timeout_s: float) -> proc.ProcResult:
        calls.append(argv)
        if argv[0] == "ffmpeg":
            Path(argv[-1]).write_bytes(b"RIFF")
        return proc.ProcResult(0, "", "", 0.1)

    monkeypatch.setattr(proc, "run", fake_run)
    out = tmp_path / "clip.wav"
    listen.ffmpeg_cut(tmp_path / "in.wav", 12.25, 18.5, out)
    listen.afplay(out)
    cut, play = calls
    assert cut[cut.index("-ss") + 1] == "12.250"
    assert cut[cut.index("-t") + 1] == "6.250"
    assert cut[-1] == str(out)
    assert play == ["/usr/bin/afplay", str(out)]


def test_a_failed_cut_fails_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fail: Callable[..., proc.ProcResult] = lambda argv, *, timeout_s: proc.ProcResult(1, "", "bad input", 0.1)  # noqa: E731
    monkeypatch.setattr(proc, "run", fail)
    with pytest.raises(Exception, match="bad input"):
        listen.ffmpeg_cut(tmp_path / "in.wav", 0.0, 1.0, tmp_path / "out.wav")
