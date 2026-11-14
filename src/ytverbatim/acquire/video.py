import time
from collections.abc import Callable
from pathlib import Path

from ytverbatim import paths
from ytverbatim.acquire.audio import DEFAULT_DOWNLOAD_TIMEOUT_S, download_refreshing_stale_info, validate_audio
from ytverbatim.acquire.metadata import info_path, load_metadata
from ytverbatim.errors import MediaValidationError, YtDlpError
from ytverbatim.ytdlp.types import YtDlpRunner

VIDEO_STEM = "video"
VIDEO_NAME = "video.mp4"

# Capped at 1080p to bound the size; the video track only, because the audio is its own stage.
VIDEO_FORMAT = "bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080]"


def fetch_video(
    runner: YtDlpRunner,
    video_id: str,
    stage_dir: Path,
    out_dir: Path,
    *,
    expected_duration: float | None,
    cookies: Path | None = None,
    timeout_s: float = DEFAULT_DOWNLOAD_TIMEOUT_S,
    now: Callable[[], float] = time.time,
) -> Path:
    """Download the video track as `video.mp4` and validate it; the info JSON must already be saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # yt-dlp skips a download whose target exists, so a corrupt earlier file has to go first.
    for stale in out_dir.glob(f"{VIDEO_STEM}.*"):
        if stale.suffix.lower() not in {".part", ".ytdl"}:
            stale.unlink(missing_ok=True)

    download_refreshing_stale_info(
        runner,
        video_id,
        stage_dir,
        cookies,
        lambda: _download(runner, stage_dir, out_dir, cookies, timeout_s),
        what="video",
        now=now,
    )

    path = out_dir / VIDEO_NAME
    if not path.is_file():
        raise MediaValidationError(f"yt-dlp finished but left no {VIDEO_NAME} in {out_dir.name}")
    try:
        validate_audio(path, expected_duration=expected_duration, kind="video")
    except MediaValidationError:
        path.unlink(missing_ok=True)
        raise
    return path


def ready_video(vdir: Path) -> Path | None:
    """Return the video file when it is whole: it decodes and covers the saved metadata's duration."""
    path = vdir / VIDEO_NAME
    if not path.is_file():
        return None
    try:
        expected = load_metadata(vdir).duration
        validate_audio(path, expected_duration=expected, kind="video")
    except (OSError, YtDlpError, MediaValidationError):
        return None
    return path


def video_file(video_id: str, out: Path | None = None) -> Path | None:
    """Return the validated video file `process` or `fetch` saved for `video_id` under `out`, else None.

    A file that is missing, corrupt or shorter than 90% of the video's duration counts as absent. The video is
    only downloaded when `[ytdlp] video = true`; this never downloads.
    """
    return ready_video(paths.video_dir(video_id, out))


def _download(runner: YtDlpRunner, stage_dir: Path, out_dir: Path, cookies: Path | None, timeout_s: float) -> None:
    runner.run(
        [
            "--load-info-json",
            str(info_path(stage_dir)),
            "-f",
            VIDEO_FORMAT,
            "--remux-video",
            "mp4",
            "-o",
            f"{out_dir / VIDEO_STEM}.%(ext)s",
        ],
        cookies=cookies,
        timeout_s=timeout_s,
        label="video",
    )
