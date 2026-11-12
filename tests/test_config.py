from pathlib import Path

import pytest

from ytverbatim.config import AuthOptions, LLMOptions, Options, load_options
from ytverbatim.errors import ConfigError


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file_gives_defaults(tmp_path: Path) -> None:
    assert load_options(tmp_path / "absent.toml") == Options()


def test_default_path_comes_from_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    (tmp_path / "ytverbatim").mkdir()
    (tmp_path / "ytverbatim" / "config.toml").write_text('[auth]\npolicy = "never"\n', encoding="utf-8")
    assert load_options().auth.policy == "never"


def test_valid_file(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
[auth]
policy = "always"
cookie_channels = ["UCexample1", "UCexample2"]

[ytdlp]
max_videos_per_run = 5
creator_note = true

[asr]
source = "captions"

[text]
chapters = "none"

[llm]
provider = "openai-compat"
base_url = "http://localhost:8080/v1"
timeout_s = 60
""",
    )
    options = load_options(path)
    assert options.auth == AuthOptions(policy="always", cookie_channels=("UCexample1", "UCexample2"))
    assert options.ytdlp.max_videos_per_run == 5
    assert options.ytdlp.creator_note is True
    assert options.asr.source == "captions"
    assert options.asr.engine == "w-prod"
    assert options.text.chapters == "none"
    assert options.llm == LLMOptions(provider="openai-compat", base_url="http://localhost:8080/v1", timeout_s=60.0)
    assert isinstance(options.llm.timeout_s, float)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"unknown option\(s\) in \[auth\]: polcy"):
        load_options(_write(tmp_path, '[auth]\npolcy = "auto"\n'))


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown config section"):
        load_options(_write(tmp_path, "[nope]\nx = 1\n"))


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown config section"):
        load_options(_write(tmp_path, "x = 1\n"))


def test_section_must_be_a_table(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a table"):
        load_options(_write(tmp_path, "auth = 3\n"))


@pytest.mark.parametrize(
    "body",
    [
        '[auth]\npolicy = "sometimes"\n',
        "[auth]\ncookie_channels = [1]\n",
        '[auth]\ncookie_channels = "UCexample"\n',
        '[ytdlp]\nmax_videos_per_run = "5"\n',
        "[ytdlp]\nmax_videos_per_run = true\n",
        "[ytdlp]\ncreator_note = 1\n",
        "[llm]\nmodel = 3\n",
        '[llm]\ntimeout_s = "slow"\n',
    ],
)
def test_wrong_value_is_rejected(tmp_path: Path, body: str) -> None:
    with pytest.raises(ConfigError):
        load_options(_write(tmp_path, body))


def test_malformed_toml_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_options(_write(tmp_path, "[auth\n"))


def test_read_error_shows_the_home_directory_as_tilde(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ConfigError) as caught:
        load_options(_write(tmp_path, "[auth\n"))
    assert str(tmp_path) not in str(caught.value)
    assert "cannot read config file ~/" in str(caught.value)


def test_override_wins_over_file(tmp_path: Path) -> None:
    path = _write(tmp_path, '[auth]\npolicy = "never"\n[ytdlp]\nmax_videos_per_run = 5\n')
    options = load_options(path, {"auth": {"policy": "always"}, "llm": {"model": "m1"}})
    assert options.auth.policy == "always"
    assert options.ytdlp.max_videos_per_run == 5
    assert options.llm.model == "m1"


def test_none_override_means_not_given(tmp_path: Path) -> None:
    path = _write(tmp_path, '[auth]\npolicy = "never"\n')
    assert load_options(path, {"auth": {"policy": None}}).auth.policy == "never"


def test_overrides_are_validated_too(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown option"):
        load_options(tmp_path / "absent.toml", {"auth": {"bogus": 1}})
    with pytest.raises(ConfigError, match="must be one of"):
        load_options(tmp_path / "absent.toml", {"auth": {"policy": "x"}})
    with pytest.raises(ConfigError, match="unknown config section"):
        load_options(tmp_path / "absent.toml", {"bogus": {"a": 1}})
