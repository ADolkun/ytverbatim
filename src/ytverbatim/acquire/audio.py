import logging
import os
import secrets
import time
from collections.abc import Callable
from pathlib import Path

from ytverbatim import proc
from ytverbatim.acquire.metadata import fetch_info, info_path
from ytverbatim.errors import MediaValidationError, YtDlpError, stderr_preview
from ytverbatim.ytdlp.runner import redact
from ytverbatim.ytdlp.types import Signal, YtDlpRunner

log = logging.getLogger(__name__)

AUDIO_STEM = "audio"
WAV_NAME = "audio.wav"
DEFAULT_DOWNLOAD_TIMEOUT_S = 3600.0
DEFAULT_CONVERT_TIMEOUT_S = 1800.0
PROBE_TIMEOUT_S = 20.0
MIN_COVERAGE = 0.9

# Signed format URLs expire; past this age a 403 most likely means the saved info is stale.
INFO_MAX_AGE_S = 3600.0

_NOT_NATIVE_SUFFIXES = frozenset({".wav", ".part", ".ytdl", ".tmp", ".json"})


def find_audio(out_dir: Path) -> Path | None:
    """Return the native-container audio file yt-dlp left in `out_dir`, if any."""
    candidates = [
        path
        for path in out_dir.glob(f"{AUDIO_STEM}.*")
        if path.is_file() and path.suffix.lower() not in _NOT_NATIVE_SUFFIXES
    ]
    if not candidates:
        return None
    # Largest wins: a stray alternate-format leftover is smaller than the complete download.
    return min(candidates, key=lambda path: (-path.stat().st_size, path.name))


def fetch_audio(
    runner: YtDlpRunner,
    video_id: str,
    stage_dir: Path,
    out_dir: Path,
    *,
    expected_duration: float | None,
    cookies: Path | None = None,
    timeout_s: float = DEFAULT_DOWNLOAD_TIMEOUT_S,
    now: Callable[[], float] = time.time,
    creator_note: bool = False,
) -> Path:
    """Download the best audio in its native container and validate it; the info JSON must already be saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Removing earlier native files keeps the post-download pick unambiguous.
    for stale in out_dir.glob(f"{AUDIO_STEM}.*"):
        if stale.suffix.lower() not in {".wav", ".part", ".ytdl"}:
            stale.unlink(missing_ok=True)

    download_refreshing_stale_info(
        runner,
        video_id,
        stage_dir,
        cookies,
        lambda: _download(runner, stage_dir, out_dir, cookies, timeout_s),
        what="audio",
        now=now,
        creator_note=creator_note,
    )

    path = find_audio(out_dir)
    if path is None:
        raise MediaValidationError(f"yt-dlp finished but left no audio file in {out_dir.name}")
    try:
        validate_audio(path, expected_duration=expected_duration)
    except MediaValidationError:
        path.unlink(missing_ok=True)
        raise
    return path


def download_refreshing_stale_info(
    runner: YtDlpRunner,
    video_id: str,
    stage_dir: Path,
    cookies: Path | None,
    download: Callable[[], None],
    *,
    what: str,
    now: Callable[[], float] = time.time,
    creator_note: bool = False,
) -> None:
    """Run `download`; a 403 with an old info JSON means expired signed URLs, so extract once more and retry once.

    The extraction overwrites the info JSON, so it must ask for comments again when the first one did.
    """
    try:
        download()
    except YtDlpError as exc:
        saved = info_path(stage_dir)
        if exc.primary is not Signal.HTTP_403 or now() - saved.stat().st_mtime <= INFO_MAX_AGE_S:
            raise
        log.warning("%s download got 403 with a stale info JSON; extracting the video again", what)
        fetch_info(runner, video_id, stage_dir, cookies=cookies, creator_note=creator_note)
        download()


def validate_audio(
    path: Path, *, expected_duration: float | None, min_coverage: float = MIN_COVERAGE, kind: str = "audio"
) -> float:
    """Raise unless `path` is non-empty, decodable and long enough; return its probed duration.

    `kind` names the medium in the errors, so the same probe serves the video file.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise MediaValidationError(f"{kind} file is missing: {path.name}") from exc
    if size == 0:
        raise MediaValidationError(f"{kind} file is empty: {path.name}")
    probed = probe_duration(path, kind=kind)
    if expected_duration and expected_duration > 0 and probed < min_coverage * expected_duration:
        raise MediaValidationError(
            f"{kind} is truncated: {path.name} is {probed:.0f}s, "
            f"under {min_coverage:.0%} of the expected {expected_duration:.0f}s"
        )
    return probed


def probe_duration(path: Path, *, kind: str = "audio") -> float:
    result = proc.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        timeout_s=PROBE_TIMEOUT_S,
    )
    if result.returncode != 0:
        # ffprobe and ffmpeg echo the absolute input path, which names the user.
        raise MediaValidationError(f"{kind} is unreadable: {path.name}\n{stderr_preview(redact(result.stderr))}")
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        duration = 0.0
    if duration <= 0:
        raise MediaValidationError(f"{kind} has no readable duration: {path.name}")
    return duration


def to_wav16k(
    source: Path,
    destination: Path,
    *,
    expected_duration: float | None = None,
    timeout_s: float = DEFAULT_CONVERT_TIMEOUT_S,
) -> None:
    """Write 16 kHz mono pcm_s16le, the ASR input format; the file appears only once it validates."""
    tmp = destination.with_name(f".{destination.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        result = proc.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                str(tmp),
            ],
            timeout_s=timeout_s,
        )
        if result.returncode != 0:
            raise MediaValidationError(
                f"ffmpeg could not convert {source.name}\n{stderr_preview(redact(result.stderr))}"
            )
        validate_audio(tmp, expected_duration=expected_duration)
        os.replace(tmp, destination)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _download(runner: YtDlpRunner, stage_dir: Path, out_dir: Path, cookies: Path | None, timeout_s: float) -> None:
    runner.run(
        [
            "--load-info-json",
            str(info_path(stage_dir)),
            "-f",
            "bestaudio/best",
            "-o",
            f"{out_dir / AUDIO_STEM}.%(ext)s",
        ],
        cookies=cookies,
        timeout_s=timeout_s,
        label="audio",
    )
