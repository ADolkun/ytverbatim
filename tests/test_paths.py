import stat
from pathlib import Path

import pytest

from ytverbatim import paths


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "YTVERBATIM_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_defaults_under_home(home: Path) -> None:
    assert paths.config_file() == home / ".config" / "ytverbatim" / "config.toml"
    assert paths.state_dir() == home / ".local" / "state" / "ytverbatim"
    assert paths.auth_dir() == home / ".local" / "state" / "ytverbatim" / "auth"
    assert paths.locks_dir() == home / ".local" / "state" / "ytverbatim" / "locks"
    assert paths.cache_dir() == home / "Library" / "Caches" / "ytverbatim"
    assert paths.pins_dir() == paths.cache_dir() / "pins"
    assert paths.llm_dir("claude") == paths.cache_dir() / "llm" / "claude"
    assert paths.home_dir() == home / ".local" / "share" / "ytverbatim"
    assert paths.default_output_root() == paths.home_dir() / "videos"


def test_xdg_and_home_overrides(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / "state"))
    monkeypatch.setenv("YTVERBATIM_HOME", str(home / "data"))
    assert paths.config_file() == home / "cfg" / "ytverbatim" / "config.toml"
    assert paths.auth_dir() == home / "state" / "ytverbatim" / "auth"
    assert paths.default_output_root() == home / "data" / "videos"


def test_relative_xdg_value_is_ignored(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", "relative/state")
    assert paths.state_dir() == home / ".local" / "state" / "ytverbatim"


def test_ytverbatim_home_expands_user(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YTVERBATIM_HOME", "~/data")
    assert paths.home_dir() == home / "data"


def test_private_dirs_are_0700(home: Path) -> None:
    created = [
        paths.state_dir(),
        paths.auth_dir(),
        paths.locks_dir(),
        paths.cache_dir(),
        paths.pins_dir(),
        paths.llm_dir("claude"),
        paths.cache_dir() / "llm",
    ]
    assert all(_mode(p) == 0o700 for p in created)


def test_existing_loose_dir_is_tightened(home: Path) -> None:
    loose = home / ".local" / "state" / "ytverbatim"
    loose.mkdir(parents=True, mode=0o755)
    loose.chmod(0o755)
    assert paths.state_dir() == loose
    assert _mode(loose) == 0o700


def test_config_file_and_output_dirs_are_not_created(home: Path) -> None:
    paths.config_file()
    paths.default_output_root()
    assert not (home / ".config").exists()
    assert not (home / ".local" / "share").exists()


def test_video_dir(home: Path, tmp_path: Path) -> None:
    assert paths.video_dir("abcDEF_1234") == paths.default_output_root() / "abcDEF_1234"
    assert paths.video_dir("abcDEF_1234", tmp_path / "out") == tmp_path / "out" / "abcDEF_1234"


@pytest.mark.parametrize("bad", ["", "../etc/passwd", "short", "abcDEF_1234/.."])
def test_video_dir_rejects_non_ids(bad: str) -> None:
    with pytest.raises(ValueError, match="not a YouTube video id"):
        paths.video_dir(bad)
