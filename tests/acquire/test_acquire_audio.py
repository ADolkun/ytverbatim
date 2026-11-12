import os
from pathlib import Path

import pytest

from ytverbatim import proc
from ytverbatim.acquire import audio
from ytverbatim.acquire.metadata import info_path
from ytverbatim.errors import DependencyError, MediaValidationError, YtDlpError
from ytverbatim.ytdlp.types import Signal, YtDlpResult

VIDEO_ID = "aaaaaaaaaaa"


def arg_after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def write_audio(args: list[str], suffix: str = "webm", size: int = 100) -> None:
    template = arg_after(args, "-o")
    Path(template.replace("%(ext)s", suffix)).write_bytes(b"x" * size)


def stage(tmp_path: Path) -> Path:
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    info_path(stage_dir).write_text("{}", encoding="utf-8")
    return stage_dir


# --- validation ---


def test_validate_accepts_a_complete_file(tmp_path: Path, fake_proc) -> None:
    media = tmp_path / "audio.webm"
    media.write_bytes(b"x")
    fake_proc.probe_returns("540.2\n")
    assert audio.validate_audio(media, expected_duration=600.0) == 540.2
    assert fake_proc.calls[0][0] == "ffprobe"
    assert fake_proc.calls[0][-1] == str(media)


def test_validate_rejects_a_file_under_ninety_percent(tmp_path: Path, fake_proc) -> None:
    media = tmp_path / "audio.webm"
    media.write_bytes(b"x")
    fake_proc.probe_returns("539.9\n")
    with pytest.raises(MediaValidationError, match="truncated"):
        audio.validate_audio(media, expected_duration=600.0)


def test_validate_without_an_expected_duration_only_needs_a_decodable_file(tmp_path: Path, fake_proc) -> None:
    media = tmp_path / "audio.webm"
    media.write_bytes(b"x")
    fake_proc.probe_returns("3.0\n")
    assert audio.validate_audio(media, expected_duration=None) == 3.0


def test_validate_rejects_missing_and_empty_files_without_probing(tmp_path: Path, fake_proc) -> None:
    with pytest.raises(MediaValidationError, match="missing"):
        audio.validate_audio(tmp_path / "nope.webm", expected_duration=1.0)
    empty = tmp_path / "audio.webm"
    empty.write_bytes(b"")
    with pytest.raises(MediaValidationError, match="empty"):
        audio.validate_audio(empty, expected_duration=1.0)
    assert fake_proc.calls == []


@pytest.mark.parametrize(
    ("stdout", "returncode", "message"),
    [("", 1, "unreadable"), ("N/A\n", 0, "no readable duration"), ("0.000000\n", 0, "no readable duration")],
)
def test_validate_rejects_undecodable_files(
    tmp_path: Path, fake_proc, stdout: str, returncode: int, message: str
) -> None:
    media = tmp_path / "audio.webm"
    media.write_bytes(b"x")
    fake_proc.probe_returns(stdout, returncode, "Invalid data found")
    with pytest.raises(MediaValidationError, match=message):
        audio.validate_audio(media, expected_duration=None)


def test_a_missing_ffprobe_is_a_dependency_error_not_a_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args: object, **_kwargs: object) -> proc.ProcResult:
        raise DependencyError("required executable not found: ffprobe")

    monkeypatch.setattr(proc, "run", missing)
    media = tmp_path / "audio.webm"
    media.write_bytes(b"x")
    with pytest.raises(DependencyError):
        audio.validate_audio(media, expected_duration=None)


# --- find ---


def test_find_audio_prefers_the_largest_native_file(tmp_path: Path) -> None:
    (tmp_path / "audio.m4a").write_bytes(b"x" * 10)
    (tmp_path / "audio.webm").write_bytes(b"x" * 50)
    (tmp_path / "audio.wav").write_bytes(b"x" * 500)
    (tmp_path / "audio.webm.part").write_bytes(b"x" * 900)
    assert audio.find_audio(tmp_path) == tmp_path / "audio.webm"


def test_find_audio_is_none_when_only_wav_exists(tmp_path: Path) -> None:
    (tmp_path / "audio.wav").write_bytes(b"x")
    assert audio.find_audio(tmp_path) is None


# --- fetch ---


def test_fetch_audio_loads_the_saved_info_and_keeps_the_native_container(fakes, tmp_path: Path, fake_proc) -> None:
    stage_dir = stage(tmp_path)
    out_dir = tmp_path / "out"
    (out_dir).mkdir()
    (out_dir / "audio.m4a").write_bytes(b"stale")
    runner = fakes.runner(lambda args, _label: (write_audio(args), fakes.ok())[1])
    fake_proc.probe_returns("600\n")
    cookies = tmp_path / "cookies.txt"

    path = audio.fetch_audio(
        runner, VIDEO_ID, stage_dir, out_dir, expected_duration=600.0, cookies=cookies, timeout_s=99.0
    )

    assert path == out_dir / "audio.webm"
    assert not (out_dir / "audio.m4a").exists()
    call = runner.calls[0]
    assert arg_after(call.args, "--load-info-json") == str(info_path(stage_dir))
    assert arg_after(call.args, "-f") == "bestaudio/best"
    assert (call.cookies, call.timeout_s, call.label) == (cookies, 99.0, "audio")
    assert len(runner.calls) == 1


def test_a_truncated_download_is_deleted_and_raises(fakes, tmp_path: Path, fake_proc) -> None:
    out_dir = tmp_path / "out"
    runner = fakes.runner(lambda args, _label: (write_audio(args), fakes.ok())[1])
    fake_proc.probe_returns("100\n")
    with pytest.raises(MediaValidationError, match="truncated"):
        audio.fetch_audio(runner, VIDEO_ID, stage(tmp_path), out_dir, expected_duration=600.0)
    assert audio.find_audio(out_dir) is None


def test_no_output_file_raises(fakes, tmp_path: Path) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok())
    with pytest.raises(MediaValidationError, match="no audio file"):
        audio.fetch_audio(runner, VIDEO_ID, stage(tmp_path), tmp_path / "out", expected_duration=None)


def test_403_with_stale_info_re_extracts_once_then_retries(fakes, tmp_path: Path, fake_proc) -> None:
    stage_dir = stage(tmp_path)
    old = info_path(stage_dir).stat().st_mtime - 2 * 3600
    os.utime(info_path(stage_dir), (old, old))
    fake_proc.probe_returns("600\n")
    audio_attempts = 0

    def handler(args: list[str], label: str) -> YtDlpResult:
        nonlocal audio_attempts
        if label == "metadata":
            return fakes.ok('{"id": "aaaaaaaaaaa"}')
        audio_attempts += 1
        if audio_attempts == 1:
            raise fakes.fail(Signal.HTTP_403)
        write_audio(args)
        return fakes.ok()

    runner = fakes.runner(handler)
    audio.fetch_audio(runner, VIDEO_ID, stage_dir, tmp_path / "out", expected_duration=600.0)

    assert [call.label for call in runner.calls] == ["audio", "metadata", "audio"]
    assert info_path(stage_dir).stat().st_mtime > old


def test_403_with_fresh_info_is_not_re_extracted(fakes, tmp_path: Path) -> None:
    def handler(_args: list[str], _label: str) -> YtDlpResult:
        raise fakes.fail(Signal.HTTP_403)

    runner = fakes.runner(handler)
    with pytest.raises(YtDlpError):
        audio.fetch_audio(runner, VIDEO_ID, stage(tmp_path), tmp_path / "out", expected_duration=None)
    assert len(runner.calls) == 1


def test_other_download_failures_propagate_even_with_stale_info(fakes, tmp_path: Path) -> None:
    stage_dir = stage(tmp_path)
    old = info_path(stage_dir).stat().st_mtime - 2 * 3600
    os.utime(info_path(stage_dir), (old, old))

    def handler(_args: list[str], _label: str) -> YtDlpResult:
        raise fakes.fail(Signal.NO_FORMATS)

    runner = fakes.runner(handler)
    with pytest.raises(YtDlpError):
        audio.fetch_audio(runner, VIDEO_ID, stage_dir, tmp_path / "out", expected_duration=None)
    assert len(runner.calls) == 1


# --- wav ---


def converting_ffmpeg(fake_proc, *, returncode: int = 0) -> None:
    def ffmpeg(argv: list[str]) -> proc.ProcResult:
        if returncode == 0:
            Path(argv[-1]).write_bytes(b"RIFF")
        return proc.ProcResult(returncode, "", "Invalid data" if returncode else "", 0.0)

    fake_proc.responses["ffmpeg"] = ffmpeg


def test_to_wav16k_converts_to_the_asr_format_and_writes_atomically(tmp_path: Path, fake_proc) -> None:
    source = tmp_path / "audio.webm"
    source.write_bytes(b"x")
    destination = tmp_path / "audio.wav"
    converting_ffmpeg(fake_proc)
    fake_proc.probe_returns("600\n")

    audio.to_wav16k(source, destination, expected_duration=600.0)

    argv = fake_proc.calls[0]
    assert argv[0] == "ffmpeg"
    assert arg_after(argv, "-i") == str(source)
    assert (arg_after(argv, "-acodec"), arg_after(argv, "-ar"), arg_after(argv, "-ac")) == ("pcm_s16le", "16000", "1")
    assert argv[-1] != str(destination)
    assert destination.read_bytes() == b"RIFF"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audio.wav", "audio.webm"]


def test_probe_stderr_shows_the_home_directory_as_tilde(
    tmp_path: Path, fake_proc, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    media = tmp_path / "audio.webm"
    media.write_bytes(b"x")
    fake_proc.probe_returns("", 1, f"{media}: Invalid data found when processing input")
    with pytest.raises(MediaValidationError) as caught:
        audio.validate_audio(media, expected_duration=None)
    assert str(tmp_path) not in str(caught.value)
    assert "~/audio.webm: Invalid data" in str(caught.value)


def test_failed_conversion_leaves_no_partial_wav(tmp_path: Path, fake_proc) -> None:
    source = tmp_path / "audio.webm"
    source.write_bytes(b"x")
    converting_ffmpeg(fake_proc, returncode=1)
    with pytest.raises(MediaValidationError, match="could not convert"):
        audio.to_wav16k(source, tmp_path / "audio.wav")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audio.webm"]


def test_a_short_conversion_is_not_published(tmp_path: Path, fake_proc) -> None:
    source = tmp_path / "audio.webm"
    source.write_bytes(b"x")
    destination = tmp_path / "audio.wav"
    destination.write_bytes(b"previous")
    converting_ffmpeg(fake_proc)
    fake_proc.probe_returns("10\n")
    with pytest.raises(MediaValidationError, match="truncated"):
        audio.to_wav16k(source, destination, expected_duration=600.0)
    assert destination.read_bytes() == b"previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audio.wav", "audio.webm"]
