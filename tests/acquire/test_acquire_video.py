import os
from pathlib import Path

import pytest

import ytverbatim
from ytverbatim import paths
from ytverbatim.acquire import video
from ytverbatim.acquire.metadata import info_path
from ytverbatim.errors import MediaValidationError, YtDlpError
from ytverbatim.ytdlp.types import Signal, YtDlpResult

VIDEO_ID = "aaaaaaaaaaa"


def arg_after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def write_video(args: list[str], suffix: str = "mp4", size: int = 100) -> None:
    Path(arg_after(args, "-o").replace("%(ext)s", suffix)).write_bytes(b"x" * size)


def stage(tmp_path: Path, duration: float = 600.0) -> Path:
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    info_path(stage_dir).write_text(f'{{"id": "{VIDEO_ID}", "duration": {duration}}}', encoding="utf-8")
    return stage_dir


def test_the_download_asks_for_the_video_track_only_as_mp4(fakes, tmp_path: Path, fake_proc) -> None:
    stage_dir = stage(tmp_path)
    out_dir = tmp_path / "out"
    runner = fakes.runner(lambda args, _label: (write_video(args), fakes.ok())[1])
    fake_proc.probe_returns("600\n")
    cookies = tmp_path / "cookies.txt"

    path = video.fetch_video(
        runner, VIDEO_ID, stage_dir, out_dir, expected_duration=600.0, cookies=cookies, timeout_s=99.0
    )

    assert path == out_dir / "video.mp4" and video.VIDEO_NAME == "video.mp4"
    (call,) = runner.calls
    assert call.args == [
        "--load-info-json",
        str(info_path(stage_dir)),
        "-f",
        "bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080]",
        "--remux-video",
        "mp4",
        "-o",
        f"{out_dir / 'video'}.%(ext)s",
    ]
    assert (call.cookies, call.timeout_s, call.label) == (cookies, 99.0, "video")


@pytest.mark.parametrize(
    ("duration", "timeout_s"),
    [(None, 3600.0), (600.0, 3600.0), (7200.0, 21600.0), (20 * 3600.0, 12 * 3600.0)],
)
def test_the_download_limit_scales_with_the_duration_between_a_floor_and_a_ceiling(
    fakes, tmp_path: Path, fake_proc, duration: float | None, timeout_s: float
) -> None:
    runner = fakes.runner(lambda args, _label: (write_video(args), fakes.ok())[1])
    fake_proc.probe_returns(f"{duration or 600.0}\n")
    video.fetch_video(runner, VIDEO_ID, stage(tmp_path), tmp_path / "out", expected_duration=duration)
    (call,) = runner.calls
    assert call.timeout_s == timeout_s


def test_an_earlier_file_is_removed_so_yt_dlp_does_not_skip_the_download(fakes, tmp_path: Path, fake_proc) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "video.mp4").write_bytes(b"corrupt")
    (out_dir / "video.webm").write_bytes(b"leftover")
    (out_dir / "audio.webm").write_bytes(b"audio stays")
    existed: list[bool] = []

    def handler(args: list[str], _label: str) -> YtDlpResult:
        existed.append((out_dir / "video.mp4").exists())
        write_video(args, size=500)
        return fakes.ok()

    fake_proc.probe_returns("600\n")
    video.fetch_video(fakes.runner(handler), VIDEO_ID, stage(tmp_path), out_dir, expected_duration=600.0)

    assert existed == [False]
    assert (out_dir / "video.mp4").stat().st_size == 500
    assert not (out_dir / "video.webm").exists() and (out_dir / "audio.webm").exists()


def test_a_truncated_download_is_deleted_and_raises(fakes, tmp_path: Path, fake_proc) -> None:
    out_dir = tmp_path / "out"
    runner = fakes.runner(lambda args, _label: (write_video(args), fakes.ok())[1])
    fake_proc.probe_returns("100\n")
    with pytest.raises(MediaValidationError, match="video is truncated"):
        video.fetch_video(runner, VIDEO_ID, stage(tmp_path), out_dir, expected_duration=600.0)
    assert not (out_dir / "video.mp4").exists()


def test_no_output_file_raises(fakes, tmp_path: Path) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok())
    with pytest.raises(MediaValidationError, match=r"no video\.mp4"):
        video.fetch_video(runner, VIDEO_ID, stage(tmp_path), tmp_path / "out", expected_duration=None)


def test_403_with_stale_info_re_extracts_once_then_retries(fakes, tmp_path: Path, fake_proc) -> None:
    stage_dir = stage(tmp_path)
    old = info_path(stage_dir).stat().st_mtime - 2 * 3600
    os.utime(info_path(stage_dir), (old, old))
    fake_proc.probe_returns("600\n")
    attempts = 0

    def handler(args: list[str], label: str) -> YtDlpResult:
        nonlocal attempts
        if label == "metadata":
            return fakes.ok('{"id": "aaaaaaaaaaa"}')
        attempts += 1
        if attempts == 1:
            raise fakes.fail(Signal.HTTP_403)
        write_video(args)
        return fakes.ok()

    runner = fakes.runner(handler)
    video.fetch_video(runner, VIDEO_ID, stage_dir, tmp_path / "out", expected_duration=600.0)

    assert [call.label for call in runner.calls] == ["video", "metadata", "video"]
    assert info_path(stage_dir).stat().st_mtime > old


def test_403_with_fresh_info_is_not_re_extracted(fakes, tmp_path: Path) -> None:
    def handler(_args: list[str], _label: str) -> YtDlpResult:
        raise fakes.fail(Signal.HTTP_403)

    runner = fakes.runner(handler)
    with pytest.raises(YtDlpError):
        video.fetch_video(runner, VIDEO_ID, stage(tmp_path), tmp_path / "out", expected_duration=None)
    assert len(runner.calls) == 1


def test_video_file_returns_a_whole_file(tmp_path: Path, fake_proc, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YTVERBATIM_HOME", str(tmp_path / "home"))
    vdir = paths.video_dir(VIDEO_ID)
    vdir.mkdir(parents=True)
    info_path(vdir).write_text('{"id": "aaaaaaaaaaa", "duration": 600}', encoding="utf-8")
    assert video.video_file(VIDEO_ID) is None

    (vdir / "video.mp4").write_bytes(b"x")
    fake_proc.probe_returns("600\n")
    assert video.video_file(VIDEO_ID) == vdir / "video.mp4"
    assert video.video_file(VIDEO_ID, tmp_path / "home" / "videos") == vdir / "video.mp4"
    assert video.video_file(VIDEO_ID, tmp_path / "elsewhere") is None


@pytest.mark.parametrize(("probed", "returncode"), [("100\n", 0), ("", 1)])
def test_video_file_treats_a_short_or_undecodable_file_as_absent(
    tmp_path: Path, fake_proc, probed: str, returncode: int
) -> None:
    vdir = paths.video_dir(VIDEO_ID, tmp_path)
    vdir.mkdir(parents=True)
    info_path(vdir).write_text('{"id": "aaaaaaaaaaa", "duration": 600}', encoding="utf-8")
    (vdir / "video.mp4").write_bytes(b"x")
    fake_proc.probe_returns(probed, returncode)
    assert video.video_file(VIDEO_ID, tmp_path) is None


def test_video_file_without_saved_metadata_is_absent(tmp_path: Path, fake_proc) -> None:
    vdir = paths.video_dir(VIDEO_ID, tmp_path)
    vdir.mkdir(parents=True)
    (vdir / "video.mp4").write_bytes(b"x")
    fake_proc.probe_returns("600\n")
    assert video.video_file(VIDEO_ID, tmp_path) is None


def test_video_file_is_exported_lazily_from_the_package() -> None:
    assert "video_file" in ytverbatim.__all__
    assert ytverbatim.video_file is video.video_file
