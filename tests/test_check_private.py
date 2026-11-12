import re
import subprocess
import sys
from pathlib import Path

import check_private
import pytest

SCRIPT = Path(check_private.__file__)


@pytest.mark.parametrize(
    "line",
    [
        "path = '/Users/jane/projects/x'",
        "cd /home/bob/work",
        "# Netscape HTTP Cookie File",
        "SAPISID=abc123",
        ".youtube.com\tTRUE\t/\tTRUE\t0\t__Secure-3PSID\tabc",
        "LOGIN_INFO\tvalue",
        "curl -b __Secure-1PSID=xyz",
    ],
)
def test_flags_private_lines(line: str) -> None:
    assert check_private.match_line(line) is not None


@pytest.mark.parametrize(
    "line",
    [
        "/Users/Shared/cache",
        "/Users/you/project",
        "/Users/example/project",
        "/home/user/.config",
        "see ~/.local/share/ytverbatim/eval",
        "The SAPISID cookie is never logged",
        "MY_SSID=1",
        "ordinary prose",
    ],
)
def test_allows_generic_lines(line: str) -> None:
    assert check_private.match_line(line) is None


def test_names_matched_pattern() -> None:
    assert check_private.match_line("/Users/jane/x") == "home-path"
    assert check_private.match_line("# Netscape HTTP Cookie File") == "netscape-cookie-header"


def test_extra_patterns_apply() -> None:
    extra = check_private.parse_patterns("# comment\n\nsecret-title\n")
    assert extra[0][0] == "local-pattern-1"
    assert check_private.match_line("a secret-title here", extra) == "local-pattern-1"
    assert check_private.match_line("a secret-title here") is None


def test_parse_patterns_skips_comments_and_blanks() -> None:
    parsed = check_private.parse_patterns("# c\n\n  \nfoo\nbar\n")
    assert [pattern.pattern for _, pattern in parsed] == ["foo", "bar"]


def test_scan_text_reports_line_numbers() -> None:
    hits = list(check_private.scan_text("f.txt", "ok\n/home/bob/x\nok\n"))
    assert [(path, number, name) for path, number, name, _ in hits] == [("f.txt", 2, "home-path")]


def test_parse_added_lines_tracks_new_line_numbers() -> None:
    diff = "\n".join(
        [
            "diff --git a/a.txt b/a.txt",
            "--- a/a.txt",
            "+++ b/a.txt",
            "@@ -1,0 +5,2 @@",
            "+first",
            "+second",
            "@@ -9 +12 @@",
            "-old",
            "+new",
            "diff --git a/gone.txt b/gone.txt",
            "--- a/gone.txt",
            "+++ /dev/null",
            "@@ -1 +0,0 @@",
            "-removed",
        ]
    )
    assert list(check_private.parse_added_lines(diff)) == [
        ("a.txt", 5, "first"),
        ("a.txt", 6, "second"),
        ("a.txt", 12, "new"),
    ]


def run_script(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True, check=False)


def test_cli_paths_mode_exit_codes_and_output(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    dirty = tmp_path / "dirty.txt"
    dirty.write_text("ok\n/Users/jane/" + "x" * 200 + "\n")
    clean = tmp_path / "clean.txt"
    clean.write_text("nothing here\n")
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\0/Users/jane/x")

    bad = run_script(str(dirty), cwd=tmp_path)
    assert bad.returncode == 1
    first = bad.stdout.splitlines()[0]
    assert re.match(rf"{re.escape(str(dirty))}:2: matched home-path: ", first)
    assert len(first.split("matched home-path: ", 1)[1]) <= check_private.MAX_SHOWN

    assert run_script(str(clean), str(binary), cwd=tmp_path).returncode == 0


def test_cli_reads_local_patterns_file(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".git" / "info").mkdir(exist_ok=True)
    (tmp_path / ".git" / "info" / "private-patterns").write_text("# c\nneedle-\\d+\n")
    target = tmp_path / "f.txt"
    target.write_text("has needle-42 inside\n")
    result = run_script(str(target), cwd=tmp_path)
    assert result.returncode == 1
    assert "matched local-pattern-1" in result.stdout
