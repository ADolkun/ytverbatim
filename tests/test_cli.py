import subprocess
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import ytverbatim
from ytverbatim.cli import LazyGroup, main


def test_version_option() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"ytverbatim, version {ytverbatim.__version__}"


def test_python_dash_m_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ytverbatim", "--version"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"ytverbatim, version {ytverbatim.__version__}"


@pytest.fixture
def lazy_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    name = "ytv_lazy_probe"
    (tmp_path / f"{name}.py").write_text(
        "import click\n\n"
        "@click.command()\n"
        "def hello() -> None:\n"
        "    click.echo('hello from lazy')\n\n"
        "not_a_command = 3\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    return name


def _group(lazy_commands: dict[str, str]) -> click.Group:
    @click.group(cls=LazyGroup, lazy_commands=lazy_commands)
    def group() -> None:
        pass

    @group.command()
    def eager() -> None:
        click.echo("eager")

    return group


def test_lazy_command_imports_only_when_invoked(lazy_module: str) -> None:
    group = _group({"hello": f"{lazy_module}:hello"})
    runner = CliRunner()

    result = runner.invoke(group, ["eager"])
    assert result.output.strip() == "eager"
    assert lazy_module not in sys.modules

    result = runner.invoke(group, ["hello"])
    assert result.exit_code == 0
    assert result.output.strip() == "hello from lazy"
    assert lazy_module in sys.modules


def test_lazy_command_is_listed_without_importing_until_help_needs_it(lazy_module: str) -> None:
    group = _group({"hello": f"{lazy_module}:hello"})
    ctx = click.Context(group)
    assert group.list_commands(ctx) == ["eager", "hello"]
    assert lazy_module not in sys.modules


def test_unknown_command_still_errors(lazy_module: str) -> None:
    result = CliRunner().invoke(_group({"hello": f"{lazy_module}:hello"}), ["nope"])
    assert result.exit_code == 2
    assert "No such command" in result.output


def test_lazy_target_must_be_a_click_command(lazy_module: str) -> None:
    result = CliRunner().invoke(_group({"bad": f"{lazy_module}:not_a_command"}), ["bad"])
    assert isinstance(result.exception, TypeError)
