from pathlib import Path

import pytest

from ytverbatim.auth import cookiefile
from ytverbatim.auth.cookiefile import CookieEntry
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError

# Built from parts so the repository privacy scanner does not mistake fixtures for a leaked export.
LONG_HEADER = "# Netscape " + "HTTP Cookie File"
SECRET = "synthetic-secret-value"


def _row(*fields: str) -> str:
    return "\t".join(fields)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cookies.txt"
    path.write_bytes(text.encode("utf-8"))
    return path


@pytest.mark.parametrize("header", [LONG_HEADER, "# HTTP Cookie File"])
def test_reads_both_headers_and_httponly_rows(tmp_path: Path, header: str) -> None:
    text = "\n".join(
        [
            header,
            "# a comment",
            "",
            _row("#HttpOnly_.youtube.com", "TRUE", "/", "TRUE", "1900000000", "PREF", SECRET),
            _row("www.example.com", "FALSE", "/a", "FALSE", "0", "k", "v"),
            "",
        ]
    )
    entries = cookiefile.read_cookie_file(_write(tmp_path, text))
    assert entries == [
        CookieEntry(".youtube.com", "/", True, 1_900_000_000, "PREF", SECRET, http_only=True),
        CookieEntry("www.example.com", "/a", False, None, "k", "v"),
    ]


def test_crlf_is_rejected(tmp_path: Path) -> None:
    text = "# HTTP Cookie File\r\n" + _row(".youtube.com", "TRUE", "/", "TRUE", "0", "PREF", SECRET) + "\r\n"
    with pytest.raises(AuthError) as caught:
        cookiefile.read_cookie_file(_write(tmp_path, text))
    assert caught.value.state is AuthState.COOKIE_FILE_INVALID
    assert "CRLF" in caught.value.detail


@pytest.mark.parametrize("first", ['[{"name": "PREF"}]', "PREF\tvalue", "", "#HTTP Cookie File"])
def test_bad_header_is_rejected(tmp_path: Path, first: str) -> None:
    with pytest.raises(AuthError) as caught:
        cookiefile.read_cookie_file(_write(tmp_path, first + "\n"))
    assert caught.value.state is AuthState.COOKIE_FILE_INVALID
    assert "first line" in caught.value.detail


def test_malformed_row_error_never_echoes_the_value(tmp_path: Path) -> None:
    text = "# HTTP Cookie File\n" + _row(".youtube.com", "TRUE", "/", "PREF", SECRET) + "\n"
    with pytest.raises(AuthError) as caught:
        cookiefile.read_cookie_file(_write(tmp_path, text))
    assert "line 2" in str(caught.value)
    assert SECRET not in str(caught.value)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AuthError) as caught:
        cookiefile.read_cookie_file(tmp_path / "absent.txt")
    assert caught.value.state is AuthState.COOKIE_FILE_INVALID


def test_errors_show_the_home_directory_as_tilde(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(AuthError) as caught:
        cookiefile.read_cookie_file(tmp_path / "absent.txt")
    assert caught.value.detail == "~/absent.txt: file not found"


def test_render_round_trips(tmp_path: Path) -> None:
    entries = [
        CookieEntry(".youtube.com", "/", True, 1_900_000_000, "PREF", SECRET, http_only=True),
        CookieEntry("accounts.google.com", "/", True, None, "k", "v"),
    ]
    assert cookiefile.parse(tmp_path / "x", cookiefile.render(entries).encode()) == entries


def test_repr_hides_value() -> None:
    assert SECRET not in repr(CookieEntry(".youtube.com", "/", True, None, "PREF", SECRET))
