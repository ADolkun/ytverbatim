import os
import re
import shutil
from dataclasses import dataclass

from ytverbatim import proc
from ytverbatim.errors import DependencyError, ProcTimeout

ENV_OVERRIDE = "YTVERBATIM_JS_RUNTIMES"
# Older node fails YouTube's n-challenge intermittently.
NODE_MIN_MAJOR = 22
_NODE_VERSION_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class JsRuntime:
    name: str | None
    args: tuple[str, ...]
    finding: str | None


def detect() -> JsRuntime:
    """Pick the runtime yt-dlp uses to solve YouTube's JS challenges; `finding` explains a missing one."""
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        return JsRuntime(override.split(":", 1)[0], ("--js-runtimes", override), None)
    # yt-dlp enables deno by default, so it needs no flag.
    if shutil.which("deno"):
        return JsRuntime("deno", (), None)
    major = node_major_version()
    if major is not None and major >= NODE_MIN_MAJOR:
        return JsRuntime("node", ("--js-runtimes", "node"), None)
    if major is not None:
        finding = (
            f"node {major} is too old to solve YouTube's JS challenges (need {NODE_MIN_MAJOR}+); "
            f"install deno or a newer node, or set {ENV_OVERRIDE}"
        )
    else:
        finding = (
            f"no JavaScript runtime found to solve YouTube's JS challenges; install deno or node "
            f"{NODE_MIN_MAJOR}+, or set {ENV_OVERRIDE}"
        )
    return JsRuntime(None, (), finding)


def node_major_version() -> int | None:
    node = shutil.which("node")
    if node is None:
        return None
    try:
        result = proc.run([node, "--version"], timeout_s=_NODE_VERSION_TIMEOUT_S)
    except (DependencyError, ProcTimeout):
        return None
    if result.returncode != 0:
        return None
    match = re.match(r"v?(\d+)\.", result.stdout.strip())
    return int(match.group(1)) if match else None
