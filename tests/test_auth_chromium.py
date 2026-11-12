import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ytverbatim import paths
from ytverbatim.auth import chromium
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "YTVERBATIM_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    path = chromium.browser_root("chrome")
    path.mkdir(parents=True)
    return path


def _store(root: Path, *parts: str, mtime: float = 1_000) -> Path:
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"sqlite")
    os.utime(path, (mtime, mtime))
    return path


def _local_state(root: Path) -> None:
    state = {
        "profile": {
            "last_used": "Work",
            "info_cache": {
                "Default": {"name": "Person 1", "user_name": "alice@example.com", "active_time": 1_700_000_000.5},
                "Work": {"name": "Work", "user_name": "bob@example.com", "active_time": 1_800_000_000},
                "Guest Dir": {"name": "Guest", "user_name": ""},
            },
        }
    }
    (root / "Local State").write_text(json.dumps(state), encoding="utf-8")


def test_list_profiles_reads_local_state_without_emails(root: Path) -> None:
    _local_state(root)
    legacy = _store(root, "Default", "Cookies")
    network = _store(root, "Work", "Network", "Cookies")
    _store(root, "Work", "Cookies")

    profiles = chromium.list_profiles("chrome")

    assert [p.dir for p in profiles] == ["Default", "Guest Dir", "Work"]
    default, guest, work = profiles
    assert default.name == "Person 1"
    assert default.active_time == datetime.fromtimestamp(1_700_000_000.5, UTC)
    assert default.cookie_db == legacy
    assert work.cookie_db == network
    assert work.last_used and not default.last_used
    assert guest.cookie_db is None
    assert guest.active_time is None
    assert all(p.email is None for p in profiles)


def test_list_profiles_masks_emails_unless_full_requested(root: Path) -> None:
    _local_state(root)
    masked = chromium.list_profiles("chrome", emails="masked")
    assert [p.email for p in masked] == ["a***@example.com", None, "b***@example.com"]
    full = chromium.list_profiles("chrome", emails="full")
    assert full[0].email == "alice@example.com"


def test_list_profiles_missing_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(AuthError) as caught:
        chromium.list_profiles("chrome")
    assert caught.value.state is AuthState.BROWSER_NOT_FOUND


def test_unknown_browser_is_rejected(root: Path) -> None:
    with pytest.raises(AuthError) as caught:
        chromium.cookie_db("netscape", None)
    assert caught.value.state is AuthState.BROWSER_NOT_FOUND
    assert "chrome" in caught.value.detail


def test_named_profile_ignores_newer_extension_store(root: Path) -> None:
    real = _store(root, "Default", "Cookies", mtime=1_000)
    _store(root, "Default", "Storage", "ext", "abc", "def", "Cookies", mtime=2_000)
    assert chromium.cookie_db("chrome", "Default") == real


def test_network_store_wins_over_legacy_location(root: Path) -> None:
    _store(root, "Default", "Cookies", mtime=2_000)
    network = _store(root, "Default", "Network", "Cookies", mtime=1_000)
    assert chromium.cookie_db("chrome", "Default") == network


def test_unnamed_profile_picks_most_recent_profile_store(root: Path) -> None:
    _store(root, "Default", "Cookies", mtime=1_000)
    _store(root, "Default", "Storage", "ext", "abc", "Cookies", mtime=9_000)
    recent = _store(root, "Guest Dir", "Cookies", mtime=2_000)
    assert chromium.cookie_db("chrome", None) == recent


def test_missing_profile_lists_available_dirs(root: Path) -> None:
    _store(root, "Default", "Cookies")
    _store(root, "Side", "Network", "Cookies")
    (root / "Empty").mkdir()
    with pytest.raises(AuthError) as caught:
        chromium.cookie_db("chrome", "Missing")
    assert caught.value.state is AuthState.PROFILE_NOT_FOUND
    assert "'Default', 'Side'" in caught.value.detail
    assert "Empty" not in caught.value.detail


def test_pin_holds_only_a_symlink_and_is_idempotent(root: Path) -> None:
    store = _store(root, "Default", "Cookies")
    pin = chromium.pin_cookie_db(store)
    assert pin.parent == paths.pins_dir()
    assert stat.S_IMODE(pin.stat().st_mode) == 0o700
    assert [entry.name for entry in pin.iterdir()] == ["Cookies"]
    link = pin / "Cookies"
    assert link.is_symlink() and link.readlink() == store.resolve()
    inode = link.lstat().st_ino

    assert chromium.pin_cookie_db(store) == pin
    assert link.lstat().st_ino == inode


def test_pin_replaces_a_stale_link_atomically(root: Path) -> None:
    store = _store(root, "Default", "Cookies")
    pin = chromium.pin_cookie_db(store)
    link = pin / "Cookies"
    link.unlink()
    link.symlink_to(root / "elsewhere")

    assert chromium.pin_cookie_db(store) == pin
    assert link.readlink() == store.resolve()
    assert [entry.name for entry in pin.iterdir()] == ["Cookies"]


def test_mask_email() -> None:
    assert chromium.mask_email("someone@example.com") == "s***@example.com"
    assert chromium.mask_email("no-at-sign") == "***"
