from datetime import date
from importlib.metadata import PackageNotFoundError

import pytest

from ytverbatim.errors import DependencyError
from ytverbatim.ytdlp import version


def test_installed_matches_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(version, "version", lambda dist: "2026.10.26")
    found = version.installed()
    assert (found.version, found.released) == ("2026.10.26", date(2026, 10, 26))
    assert found.age_days(today=date(2026, 11, 11)) == 16


def test_installed_reads_the_real_distribution() -> None:
    found = version.installed()
    assert found.version.startswith(str(found.released.year))


@pytest.mark.parametrize(
    ("text", "released"),
    [("2026.10.26", date(2026, 10, 26)), ("2026.1.5", date(2026, 1, 5)), ("2026.10.26.232849", date(2026, 10, 26))],
)
def test_release_date(text: str, released: date) -> None:
    assert version.release_date(text) == released


@pytest.mark.parametrize("text", ["1.2", "nightly", "20261026"])
def test_release_date_rejects_non_dates(text: str) -> None:
    with pytest.raises(ValueError, match="date-based"):
        version.release_date(text)


def test_missing_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(dist: str) -> str:
        raise PackageNotFoundError(dist)

    monkeypatch.setattr(version, "version", missing)
    with pytest.raises(DependencyError):
        version.installed()
