"""Disk budget for candidate weights: a gate before a download and a verified cleanup after a hypothesis is done.

Candidate weights live in their own Hugging Face cache (``--hf-cache``), never in the default one, which holds the
Whisper control and other projects' models.
"""

import importlib
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from ytverbatim.asr.protocol import EngineDef
from ytverbatim.errors import ConfigError, DependencyError

GATE_HEADROOM_BYTES = 1_500_000_000

# Download sizes from the Hugging Face API (plan appendix B1), rounded up to 10 MB.
REPO_BYTES: Mapping[str, int] = {
    "mlx-community/Qwen3-ASR-1.7B-8bit": 2_470_000_000,
    "mlx-community/FireRedASR2-AED-mlx": 4_570_000_000,
    "mlx-community/MiMo-V2.5-ASR-MLX": 4_530_000_000,
    "mlx-community/MiMo-Audio-Tokenizer": 2_580_000_000,
    "mlx-community/whisper-large-v3-mlx": 2_900_000_000,
    "mlx-community/Qwen3.5-9B-8bit": 10_450_000_000,
    "mlx-community/Qwen3-8B-8bit": 8_720_000_000,
}

# The Whisper adapter reads its weights from the default cache and never downloads; nothing of it may be deleted.
DEFAULT_CACHE_FAMILIES = frozenset({"whisper"})


@dataclass(frozen=True)
class Weights:
    repo: str
    revision: str


def weights(definition: EngineDef) -> tuple[Weights, ...]:
    """Every repo a hypothesis downloads: the model, plus a tokenizer repo when its variant names one."""
    spec = definition.spec
    found = [Weights(spec.repo, spec.revision)]
    repo, revision = definition.variant.get("tokenizer_repo"), definition.variant.get("tokenizer_revision")
    if isinstance(repo, str) and isinstance(revision, str):
        found.append(Weights(repo, revision))
    return tuple(found)


def repo_dir(cache: Path, repo: str) -> Path:
    return cache / f"models--{repo.replace('/', '--')}"


def present(cache: Path, item: Weights) -> bool:
    snapshot = repo_dir(cache, item.repo) / "snapshots" / item.revision
    return snapshot.is_dir() and any(snapshot.iterdir())


def free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def check_disk(definition: EngineDef, cache: Path, *, free: Callable[[Path], int] = free_bytes) -> int:
    """Raise unless the missing weights fit with headroom; return the bytes still to download."""
    if definition.spec.family in DEFAULT_CACHE_FAMILIES:
        return 0
    return check_weights(weights(definition), cache, definition.spec.id, free=free)


def check_weights(
    items: Sequence[Weights], cache: Path, label: str, *, free: Callable[[Path], int] = free_bytes
) -> int:
    """Raise unless the missing ``items`` fit with headroom; return the bytes still to download."""
    missing = [item for item in items if not present(cache, item)]
    if not missing:
        return 0
    unknown = [item.repo for item in missing if item.repo not in REPO_BYTES]
    if unknown:
        raise ConfigError(f"no size on record for {', '.join(unknown)}; add it to evals/disk.py REPO_BYTES")
    needed = sum(REPO_BYTES[item.repo] for item in missing)
    available = free(cache)
    if available < needed + GATE_HEADROOM_BYTES:
        raise ConfigError(
            f"{label} needs {_gb(needed)} of weights plus {_gb(GATE_HEADROOM_BYTES)} headroom, but only "
            f"{_gb(available)} is free on the volume of {cache}; free space or clean up a finished hypothesis first"
        )
    return needed


class _Revision(Protocol):
    @property
    def commit_hash(self) -> str: ...


class _Repo(Protocol):
    @property
    def repo_id(self) -> str: ...
    @property
    def revisions(self) -> Iterable[_Revision]: ...


class _Strategy(Protocol):
    @property
    def expected_freed_size(self) -> int: ...
    def execute(self) -> None: ...


class _CacheInfo(Protocol):
    @property
    def repos(self) -> Iterable[_Repo]: ...
    def delete_revisions(self, *revisions: str) -> _Strategy: ...


def delete_repo(cache: Path, repo: str) -> int:
    """Delete every cached revision of ``repo`` and return the bytes freed.

    huggingface_hub 2.x keeps large files in a cache-wide shared blob store with reference manifests, so removing
    the ``models--org--name`` directory alone would leave the weights on disk. Its own delete strategy also sweeps
    shared blobs nothing else references.
    """
    try:
        manager = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise DependencyError("install ytverbatim[asr] to manage the candidate weight cache") from exc
    info = cast(_CacheInfo, manager.scan_cache_dir(cache))
    revisions = [r.commit_hash for entry in info.repos if entry.repo_id == repo for r in entry.revisions]
    if not revisions:
        return 0
    strategy = info.delete_revisions(*revisions)
    strategy.execute()
    if repo_dir(cache, repo).exists():
        raise ConfigError(f"{repo_dir(cache, repo)} is still present after deletion; remove it by hand")
    return strategy.expected_freed_size


def default_cache() -> Path:
    try:
        constants = importlib.import_module("huggingface_hub.constants")
        return Path(cast(str, constants.default_cache_path)).expanduser().resolve()
    except ImportError:
        return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def require_candidate_cache(cache: Path) -> None:
    resolved = cache.expanduser().resolve()
    protected = (default_cache(), (Path.home() / ".cache" / "huggingface").resolve())
    if any(resolved == p or p in resolved.parents or resolved in p.parents for p in protected):
        raise ConfigError(f"{cache} is (inside) the default Hugging Face cache; candidate cleanup never touches it")


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"
