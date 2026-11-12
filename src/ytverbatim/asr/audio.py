import wave
from dataclasses import dataclass
from pathlib import Path

from ytverbatim.asr.chunks import FRAME_S
from ytverbatim.errors import DependencyError, MediaValidationError

try:
    import numpy as np
    from numpy.typing import NDArray
except ImportError as exc:
    raise DependencyError("install ytverbatim[asr]") from exc

SAMPLE_RATE = 16_000
_PCM_SCALE = 32768.0
_SAMPLE_WIDTH_BYTES = 2
_BLOCK_FRAMES = 600


@dataclass(frozen=True, slots=True, eq=False)
class Audio:
    """Mono 16 kHz audio held as int16 so a multi-hour file costs half the memory of float32."""

    pcm: NDArray[np.int16]
    sample_rate: int = SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.pcm) / self.sample_rate

    def slice(self, start_s: float = 0.0, end_s: float | None = None) -> NDArray[np.float32]:
        """Float32 samples in [-1, 1) for the half-open range of seconds, clamped to the audio."""
        first = max(round(start_s * self.sample_rate), 0)
        last = len(self.pcm) if end_s is None else min(round(end_s * self.sample_rate), len(self.pcm))
        return self.pcm[first : max(last, first)].astype(np.float32) / _PCM_SCALE


def load_wav(path: Path) -> Audio:
    """Read a 16 kHz mono s16le WAV, the format the audio acquisition step produces."""
    try:
        with wave.open(str(path), "rb") as reader:
            channels, width, rate = reader.getnchannels(), reader.getsampwidth(), reader.getframerate()
            if (channels, width, rate) != (1, _SAMPLE_WIDTH_BYTES, SAMPLE_RATE):
                raise MediaValidationError(
                    f"expected 16 kHz mono 16-bit WAV, got {rate} Hz, {channels} channel(s), {width * 8}-bit"
                )
            raw = reader.readframes(reader.getnframes())
    except (wave.Error, EOFError) as exc:
        raise MediaValidationError(f"not a readable WAV file: {exc}") from exc
    return Audio(np.frombuffer(raw, dtype="<i2"))


def frame_energy(audio: Audio, frame_s: float = FRAME_S) -> list[float]:
    """RMS level of each `frame_s` frame, the last one possibly partial."""
    frame_len = round(audio.sample_rate * frame_s)
    energies: list[float] = []
    for lo in range(0, len(audio.pcm), frame_len * _BLOCK_FRAMES):
        block = audio.pcm[lo : lo + frame_len * _BLOCK_FRAMES].astype(np.float32) / _PCM_SCALE
        whole = len(block) // frame_len
        if whole:
            frames = block[: whole * frame_len].reshape(whole, frame_len)
            energies.extend(np.sqrt(np.mean(frames * frames, axis=1)).tolist())
        rest = block[whole * frame_len :]
        if rest.size:
            energies.append(float(np.sqrt(np.mean(rest * rest))))
    return energies
