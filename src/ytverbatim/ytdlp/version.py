import re
from dataclasses import dataclass
from datetime import date
from importlib.metadata import PackageNotFoundError, version

from ytverbatim.errors import DependencyError

# Distribution metadata is the same interpreter `python -m yt_dlp` runs from, and reading it avoids
# importing yt_dlp's whole extractor tree.
_DIST = "yt-dlp"
_DATE = re.compile(r"(\d{4})\.(\d{1,2})\.(\d{1,2})(?:\D|$)")


@dataclass(frozen=True)
class YtDlpVersion:
    version: str
    released: date

    def age_days(self, today: date | None = None) -> int:
        return ((today or date.today()) - self.released).days


def installed() -> YtDlpVersion:
    try:
        text = version(_DIST)
    except PackageNotFoundError as exc:
        raise DependencyError("yt-dlp is not installed in this environment") from exc
    return YtDlpVersion(text, release_date(text))


def release_date(text: str) -> date:
    """yt-dlp versions are release dates (`2026.10.26`, nightlies append a build suffix)."""
    match = _DATE.match(text)
    if match is None:
        raise ValueError(f"not a date-based yt-dlp version: {text!r}")
    return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
