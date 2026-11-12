import http.cookiejar
import json
import logging
import os
import stat
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import pytest

from ytverbatim import paths, proc
from ytverbatim.auth import _snapshot_child as child
from ytverbatim.auth import chromium, cookiefile, snapshot
from ytverbatim.auth.source import BrowserSource, CookieFileSource
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError, ProcTimeout, YtverbatimError

SECRET = "synthetic-login-secret"
FUTURE = 4_000_000_000
PAST = 1_000_000_000


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "YTVERBATIM_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def store(home: Path) -> Path:
    path = chromium.browser_root("chrome") / "Default" / "Cookies"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"sqlite")
    return path


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _cookie(name: str, domain: str, *, value: str = "v", expires: int | None = FUTURE) -> http.cookiejar.Cookie:
    return http.cookiejar.Cookie(
        0, name, value, None, False, domain, True, domain.startswith("."), "/", True, True, expires,
        expires is None, None, None, {},
    )  # fmt: skip


def _jar(*, login_expires: int | None = FUTURE, sapisid: bool = True) -> list[http.cookiejar.Cookie]:
    cookies = [
        _cookie("LOGIN_INFO", ".youtube.com", value=SECRET, expires=login_expires),
        _cookie("PREF", ".youtube.com"),
        _cookie("VISITOR", "m.youtube.com", expires=None),
        _cookie("NID", ".google.com"),
        _cookie("ACCOUNT", "accounts.google.com"),
        _cookie("other", ".example.com", value=SECRET),
        _cookie("lookalike", ".notyoutube.com"),
        _cookie("gone", ".google.com", expires=PAST),
    ]
    if sapisid:
        cookies.append(_cookie("__Secure-3PAPISID", ".youtube.com", value=SECRET))
    return cookies


class FakeExtractor:
    def __init__(
        self,
        jar: list[http.cookiejar.Cookie],
        *,
        warnings: Sequence[str] = (),
        info: Sequence[str] = (),
        error: Exception | None = None,
    ) -> None:
        self.jar = jar
        self.warnings = warnings
        self.info = info
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def __call__(self, browser: str, profile: str, logger: child.CaptureLogger) -> Iterable[http.cookiejar.Cookie]:
        self.calls.append((browser, profile))
        logger.debug(f"Extracting cookies from: {profile} value {SECRET}")
        for message in self.info:
            logger.info(message)
        for message in self.warnings:
            logger.warning(message, only_once=True)
        if self.error is not None:
            raise self.error
        return self.jar


def _run_child(out: Path, extractor: FakeExtractor, capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    code = child.main(["--browser", "chrome", "--profile", "Default", "--out", str(out)], extract=extractor)
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    result = json.loads(captured.out)
    assert code == (0 if result["ok"] else 1)
    return result


def test_child_filters_domains_and_writes_private_file(store: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = snapshot.new_snapshot_path()
    extractor = FakeExtractor(_jar(), info=["Extracted 9 cookies from chrome (2 could not be decrypted)"])

    result = _run_child(out, extractor, capsys)

    assert result == {
        "ok": True,
        "counts": {"youtube.com": 4, "google.com": 2},
        "decrypt_failures": 2,
        "logged_in": True,
        "warnings": [],
    }
    [(browser, pin_dir)] = extractor.calls
    assert browser == "chrome"
    assert (Path(pin_dir) / "Cookies").readlink() == store.resolve()
    assert _mode(out) == 0o600
    assert _mode(out.parent) == 0o700
    domains = {entry.domain for entry in cookiefile.read_cookie_file(out)}
    assert domains == {".youtube.com", "m.youtube.com", ".google.com", "accounts.google.com"}
    assert "gone" not in out.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("jar", "label"),
    [(_jar(login_expires=PAST), "expired"), (_jar(sapisid=False), "no-sapisid"), ([], "empty")],
)
def test_child_logged_out_writes_nothing(
    store: Path, capsys: pytest.CaptureFixture[str], jar: list[http.cookiejar.Cookie], label: str
) -> None:
    out = snapshot.new_snapshot_path()
    result = _run_child(out, FakeExtractor(jar), capsys)
    assert result["state"] == AuthState.LOGGED_OUT.value, label
    assert not out.exists()


def test_login_rule_only_counts_cookies_sent_to_www() -> None:
    now = 2_000_000_000.0
    entries = [
        cookiefile.CookieEntry("accounts.youtube.com", "/", True, None, "LOGIN_INFO", "x"),
        cookiefile.CookieEntry("www.youtube.com", "/", True, None, "SAPISID", "x"),
    ]
    assert not snapshot.is_logged_in(entries, now)
    entries.append(cookiefile.CookieEntry("youtube.com", "/", True, None, "LOGIN_INFO", "x"))
    assert snapshot.is_logged_in(entries, now)


@pytest.mark.parametrize(
    ("extractor", "state"),
    [
        (FakeExtractor(_jar(), warnings=["find-generic-password failed"]), AuthState.KEYCHAIN_DENIED),
        (FakeExtractor(_jar(), warnings=["cannot decrypt v10 cookies: no key found"]), AuthState.KEYCHAIN_DENIED),
        (
            FakeExtractor([], info=["Extracted 0 cookies from chrome (940 could not be decrypted)"]),
            AuthState.KEYCHAIN_DENIED,
        ),
        (
            FakeExtractor([], error=FileNotFoundError("could not find chrome cookies database")),
            AuthState.PROFILE_NOT_FOUND,
        ),
    ],
)
def test_child_maps_failures(
    store: Path, capsys: pytest.CaptureFixture[str], extractor: FakeExtractor, state: AuthState
) -> None:
    out = snapshot.new_snapshot_path()
    result = _run_child(out, extractor, capsys)
    assert result["ok"] is False
    assert result["state"] == state.value
    assert not out.exists()


def test_child_reports_missing_profile_with_choices(store: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = child.main(["--browser", "chrome", "--profile", "Nope", "--out", "x"], extract=FakeExtractor(_jar()))
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["state"] == AuthState.PROFILE_NOT_FOUND.value
    assert "'Default'" in result["detail"]


def test_child_warning_kinds_never_carry_messages(store: Path, capsys: pytest.CaptureFixture[str]) -> None:
    extractor = FakeExtractor(_jar(), warnings=[f"strange thing {SECRET}", f'unknown cookie version: "{SECRET}"'])
    result = _run_child(snapshot.new_snapshot_path(), extractor, capsys)
    assert result["warnings"] == ["other", "unknown_cookie_version"]


def test_child_module_runs_as_subprocess(home: Path) -> None:
    out = snapshot.new_snapshot_path()
    result = proc.run(
        [sys.executable, "-m", snapshot.CHILD_MODULE, "--browser", "chrome", "--out", str(out)], timeout_s=60
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["state"] == AuthState.BROWSER_NOT_FOUND.value


class FakeRun:
    def __init__(self, stdout: str = "", *, raise_timeout: bool = False, write: bool = True) -> None:
        self.stdout = stdout
        self.raise_timeout = raise_timeout
        self.write = write
        self.argv: list[str] = []
        self.timeout_s = 0.0

    def __call__(self, argv: Sequence[str], *, timeout_s: float, **_: object) -> proc.ProcResult:
        self.argv = list(argv)
        self.timeout_s = timeout_s
        if self.raise_timeout:
            raise ProcTimeout(Path(argv[0]).name, timeout_s)
        if self.write:
            out = Path(self.argv[self.argv.index("--out") + 1])
            out.write_text("# HTTP Cookie File\n", encoding="utf-8")
        return proc.ProcResult(0, self.stdout, "", 0.5)


_OK = json.dumps(
    {"ok": True, "counts": {"youtube.com": 3, "google.com": 1}, "decrypt_failures": 0, "logged_in": True,
     "warnings": ["other"]}
)  # fmt: skip


def test_browser_snapshot_runs_child_and_deletes_file(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRun(_OK)
    monkeypatch.setattr(proc, "run", fake)

    with snapshot.cookie_snapshot(BrowserSource("chrome", "Default"), timeout_s=7) as snap:
        assert snap.path.exists()
        assert snap.path.parent == paths.snapshots_dir()
        assert snap.summary == snapshot.SnapshotSummary({"youtube.com": 3, "google.com": 1}, 0, True, ("other",))

    assert not snap.path.exists()
    assert fake.argv == [
        sys.executable, "-m", "ytverbatim.auth._snapshot_child", "--browser", "chrome", "--out", str(snap.path),
        "--profile", "Default",
    ]  # fmt: skip
    assert fake.timeout_s == 7


def test_snapshot_file_is_deleted_when_the_job_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proc, "run", FakeRun(_OK))
    with pytest.raises(RuntimeError), snapshot.cookie_snapshot(BrowserSource("chrome")) as snap:
        raise RuntimeError("job failed")
    assert not snap.path.exists()
    assert list(paths.snapshots_dir().iterdir()) == []


def test_timeout_maps_to_keychain_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proc, "run", FakeRun(raise_timeout=True))
    with pytest.raises(AuthError) as caught, snapshot.cookie_snapshot(BrowserSource("chrome"), timeout_s=3):
        pass
    assert caught.value.state is AuthState.KEYCHAIN_DENIED
    assert "Keychain prompt" in caught.value.detail
    assert "Chrome Safe Storage" in caught.value.detail


def test_child_error_state_is_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"ok": False, "state": "profile_not_found", "detail": "profiles with cookies: 'Default'"})
    monkeypatch.setattr(proc, "run", FakeRun(payload, write=False))
    with pytest.raises(AuthError) as caught, snapshot.cookie_snapshot(BrowserSource("chrome", "X")):
        pass
    assert caught.value.state is AuthState.PROFILE_NOT_FOUND
    assert caught.value.detail == "profiles with cookies: 'Default'"


@pytest.mark.parametrize("stdout", ["", "not json", json.dumps({"ok": False, "state": None, "detail": "boom"})])
def test_unexpected_child_output_fails_loud(monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
    monkeypatch.setattr(proc, "run", FakeRun(stdout, write=False))
    with pytest.raises(YtverbatimError), snapshot.cookie_snapshot(BrowserSource("chrome")):
        pass


def _cookie_file(home: Path, rows: list[cookiefile.CookieEntry]) -> Path:
    path = home / "exported.txt"
    path.write_text(cookiefile.render(rows), encoding="utf-8")
    return path


def test_cookie_file_snapshot_is_a_filtered_private_copy(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proc, "run", FakeRun(raise_timeout=True))
    entries = [child._entry(c) for c in _jar()]
    original = _cookie_file(home, entries)
    before = original.read_bytes()

    with snapshot.cookie_snapshot(CookieFileSource(original)) as snap:
        assert snap.path != original
        assert _mode(snap.path) == 0o600
        kept = cookiefile.read_cookie_file(snap.path)
        assert {e.domain for e in kept} == {".youtube.com", "m.youtube.com", ".google.com", "accounts.google.com"}
        assert snap.summary.logged_in

    assert not snap.path.exists()
    assert original.read_bytes() == before
    assert sorted(p.name for p in home.iterdir() if p.is_file()) == ["exported.txt"]


def test_cookie_file_logged_out(home: Path) -> None:
    original = _cookie_file(home, [child._entry(c) for c in _jar(login_expires=PAST)])
    with pytest.raises(AuthError) as caught, snapshot.cookie_snapshot(CookieFileSource(original)):
        pass
    assert caught.value.state is AuthState.LOGGED_OUT
    assert list(paths.snapshots_dir().iterdir()) == []


def test_stale_snapshots_are_cleaned_on_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    old = paths.snapshots_dir() / "1-1.txt"
    leftover = paths.snapshots_dir() / ".2-2.txt.123.abcd.tmp"
    fresh = paths.snapshots_dir() / "3-3.txt"
    for path in (old, leftover, fresh):
        path.write_text("x", encoding="utf-8")
    stale = time.time() - snapshot.STALE_AFTER_S - 60
    os.utime(old, (stale, stale))
    os.utime(leftover, (stale, stale))
    monkeypatch.setattr(proc, "run", FakeRun(_OK))

    with snapshot.cookie_snapshot(BrowserSource("chrome")):
        pass

    assert sorted(p.name for p in paths.snapshots_dir().iterdir()) == ["3-3.txt"]


def test_logs_never_contain_cookie_values(
    home: Path, store: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, capsys
) -> None:
    caplog.set_level(logging.DEBUG)
    out = snapshot.new_snapshot_path()
    child_result = _run_child(out, FakeExtractor(_jar(), warnings=[f"x {SECRET}"]), capsys)
    out.unlink()
    monkeypatch.setattr(proc, "run", FakeRun(json.dumps(child_result)))
    with snapshot.cookie_snapshot(BrowserSource("chrome", "Default")):
        pass
    original = _cookie_file(home, [child._entry(c) for c in _jar()])
    with snapshot.cookie_snapshot(CookieFileSource(original)):
        pass
    assert caplog.records
    assert SECRET not in caplog.text
