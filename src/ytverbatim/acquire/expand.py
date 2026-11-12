import logging
from collections.abc import Sequence
from pathlib import Path

from ytverbatim.ids import RefKind, classify, is_video_id
from ytverbatim.ytdlp.types import YtDlpRunner

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 300.0


def expand_target(
    runner: YtDlpRunner,
    target: str,
    *,
    limit: int = 0,
    cookies: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[str]:
    """Video ids behind one URL or id, in order and without repeats; a single video costs no request."""
    ref = classify(target)
    if ref is None:
        raise ValueError(f"not a YouTube video, playlist or channel: {target!r}")
    if ref.kind is RefKind.VIDEO:
        return [ref.id]
    args = ["--flat-playlist", "--print", "%(id)s"]
    if limit > 0:
        args += ["--playlist-end", str(limit)]
    result = runner.run([*args, ref.url], cookies=cookies, timeout_s=timeout_s, label="expand")
    ids = (line.strip() for line in result.stdout.splitlines())
    # Channel pages list their tabs as entries too; only video ids are wanted.
    return list(dict.fromkeys(video_id for video_id in ids if is_video_id(video_id)))


def expand_targets(
    runner: YtDlpRunner,
    targets: Sequence[str],
    *,
    limit: int = 0,
    cookies: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[str]:
    """Like `expand_target` over several targets, de-duplicated across all of them; the first failure raises."""
    ids: dict[str, None] = {}
    for target in targets:
        for video_id in expand_target(runner, target, limit=limit, cookies=cookies, timeout_s=timeout_s):
            ids.setdefault(video_id)
    return list(ids)
