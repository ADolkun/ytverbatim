import subprocess
import sys

from click.testing import CliRunner

import ytverbatim
from ytverbatim.cli import main


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
