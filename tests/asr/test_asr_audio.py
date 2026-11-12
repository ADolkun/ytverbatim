import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from ytverbatim.asr.audio import SAMPLE_RATE, Audio, frame_energy, load_wav
from ytverbatim.errors import DependencyError, MediaValidationError


def write_wav(path: Path, samples: list[int], *, channels: int = 1, rate: int = SAMPLE_RATE, width: int = 2) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(struct.pack(f"<{len(samples)}h", *samples) if width == 2 else bytes(samples))


def tone(seconds: float, amplitude: int, hz: float = 440.0) -> list[int]:
    count = round(seconds * SAMPLE_RATE)
    return [round(amplitude * math.sin(2 * math.pi * hz * n / SAMPLE_RATE)) for n in range(count)]


def test_load_and_slice(tmp_path: Path) -> None:
    path = tmp_path / "-tone.wav"
    write_wav(path, tone(2.0, 16384))
    audio = load_wav(path)
    assert audio.duration == pytest.approx(2.0)
    piece = audio.slice(0.5, 1.0)
    assert piece.dtype == np.float32
    assert len(piece) == SAMPLE_RATE // 2
    assert float(np.max(np.abs(piece))) == pytest.approx(0.5, abs=0.01)
    assert len(audio.slice(1.5)) == SAMPLE_RATE // 2
    assert len(audio.slice(1.5, 99.0)) == SAMPLE_RATE // 2
    assert len(audio.slice(1.0, 0.5)) == 0
    assert len(audio.slice(-1.0, 0.1)) == SAMPLE_RATE // 10


def test_frame_energy_follows_the_signal(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    write_wav(path, [0] * SAMPLE_RATE + tone(1.0, 16384) + tone(0.05, 16384))
    energy = frame_energy(load_wav(path))
    assert len(energy) == 21
    assert all(level == 0.0 for level in energy[:10])
    assert all(level == pytest.approx(0.5 / math.sqrt(2), abs=0.01) for level in energy[10:20])
    assert energy[20] == pytest.approx(0.5 / math.sqrt(2), abs=0.02)


def test_frame_energy_spans_several_blocks() -> None:
    audio = Audio(np.full(SAMPLE_RATE * 130, 8192, dtype=np.int16))
    energy = frame_energy(audio)
    assert len(energy) == 1300
    assert all(level == pytest.approx(0.25) for level in energy)


def test_empty_audio() -> None:
    audio = Audio(np.zeros(0, dtype=np.int16))
    assert audio.duration == 0.0
    assert frame_energy(audio) == []


@pytest.mark.parametrize(
    ("kwargs", "detail"),
    [({"rate": 44100}, "44100 Hz"), ({"channels": 2}, "2 channel"), ({"width": 1}, "8-bit")],
)
def test_wrong_format_is_rejected(tmp_path: Path, kwargs: dict[str, int], detail: str) -> None:
    path = tmp_path / "bad.wav"
    write_wav(path, [0] * 64, **kwargs)
    with pytest.raises(MediaValidationError, match=detail):
        load_wav(path)


def test_not_a_wav_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "junk.wav"
    path.write_bytes(b"this is not audio")
    with pytest.raises(MediaValidationError, match="not a readable WAV"):
        load_wav(path)


def test_importing_without_numpy_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.delitem(sys.modules, "ytverbatim.asr.audio")
    with pytest.raises(DependencyError, match=r"ytverbatim\[asr\]"):
        __import__("ytverbatim.asr.audio")


def test_core_modules_do_not_import_mlx() -> None:
    code = (
        "import sys, ytverbatim.asr.chunks, ytverbatim.asr.audio\n"
        "sys.exit(any(name.split('.')[0] == 'mlx' for name in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
