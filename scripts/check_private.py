"""Block personal data from entering a public repository.

Scans added lines (``--staged``), tracked files (``--all-tracked``) or explicit
paths. Extra regexes come from the local, never-committed
``<git-common-dir>/info/private-patterns`` (shared by all worktrees) file.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

MAX_SHOWN = 80

ALLOWED_HOME_USERS = frozenset({"Shared", "you", "example", "user", "username", "name"})

_HOME_PATH = re.compile(r"(?:/Users|/home)/([A-Za-z0-9._-]+)/")
_COOKIE_NAMES = r"(?:SAPISID|APISID|HSID|SSID|SID|__Secure-[13]PSID|__Secure-[13]PAPISID|LOGIN_INFO)"

BUILTIN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("netscape-cookie-header", re.compile(r"# Netscape HTTP Cookie File")),
    ("session-cookie", re.compile(rf"(?<![A-Za-z0-9_-]){_COOKIE_NAMES}(?:=|\t)")),
)

_HOME_PATH_NAME = "home-path"

# These files must spell out the patterns they test, so they would flag themselves.
SELF_EXEMPT = frozenset({"scripts/check_private.py", "tests/test_check_private.py"})


def match_line(line: str, extra: Sequence[tuple[str, re.Pattern[str]]] = ()) -> str | None:
    """Return the name of the first pattern the line matches, or None."""
    for found in _HOME_PATH.finditer(line):
        if found.group(1) not in ALLOWED_HOME_USERS:
            return _HOME_PATH_NAME
    for name, pattern in (*BUILTIN_PATTERNS, *extra):
        if pattern.search(line):
            return name
    return None


def scan_text(
    path: str, text: str, extra: Sequence[tuple[str, re.Pattern[str]]] = ()
) -> Iterator[tuple[str, int, str, str]]:
    """Yield (path, line number, pattern name, line) for every matching line."""
    for number, line in enumerate(text.splitlines(), start=1):
        name = match_line(line, extra)
        if name is not None:
            yield path, number, name, line


def parse_patterns(text: str) -> list[tuple[str, re.Pattern[str]]]:
    """Parse a patterns file: one regex per line, ``#`` comments and blanks ignored."""
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append((f"local-pattern-{len(patterns) + 1}", re.compile(line)))
    return patterns


def parse_added_lines(diff: str) -> Iterator[tuple[str, int, str]]:
    """Yield (path, new line number, text) for each added line of a ``-U0`` diff."""
    path = ""
    number = 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else ""
        elif line.startswith("@@"):
            hunk = re.search(r"\+(\d+)", line)
            number = int(hunk.group(1)) if hunk else 0
        elif line.startswith("+") and path:
            yield path, number, line[1:]
            number += 1


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def load_local_patterns() -> list[tuple[str, re.Pattern[str]]]:
    git_dir = Path(_git("rev-parse", "--git-common-dir").decode().strip())
    patterns_file = git_dir / "info" / "private-patterns"
    if not patterns_file.is_file():
        return []
    return parse_patterns(patterns_file.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\0" in data:
        return None
    return data.decode("utf-8", errors="replace")


def _scan_paths(
    paths: Iterable[Path], extra: Sequence[tuple[str, re.Pattern[str]]]
) -> Iterator[tuple[str, int, str, str]]:
    for path in paths:
        if not path.is_file():
            continue
        text = _read_text(path)
        if text is not None:
            yield from scan_text(str(path), text, extra)


def _scan_staged(extra: Sequence[tuple[str, re.Pattern[str]]]) -> Iterator[tuple[str, int, str, str]]:
    diff = _git("diff", "--cached", "-U0", "--no-color", "--no-ext-diff").decode("utf-8", errors="replace")
    for path, number, text in parse_added_lines(diff):
        if path in SELF_EXEMPT:
            continue
        name = match_line(text, extra)
        if name is not None:
            yield path, number, name, text


def _tracked_files() -> list[Path]:
    names = _git("ls-files", "-z").decode().split("\0")
    return [Path(name) for name in names if name and name not in SELF_EXEMPT]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--staged", action="store_true", help="scan added lines of the staged diff")
    mode.add_argument("--all-tracked", action="store_true", help="scan every tracked text file")
    parser.add_argument("paths", nargs="*", type=Path, help="files to scan")
    args = parser.parse_args(argv)
    if not (args.staged or args.all_tracked or args.paths):
        parser.error("choose --staged, --all-tracked or file paths")

    extra = load_local_patterns()
    if args.staged:
        hits = _scan_staged(extra)
    elif args.all_tracked:
        hits = _scan_paths(_tracked_files(), extra)
    else:
        hits = _scan_paths(args.paths, extra)

    found = False
    for path, number, name, line in hits:
        found = True
        shown = line.strip()[:MAX_SHOWN]
        print(f"{path}:{number}: matched {name}: {shown}")
    if found:
        print("Private data detected; remove it before committing.", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
