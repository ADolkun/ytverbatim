from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ytverbatim.asr.chunks import Chunk
from ytverbatim.asr.protocol import AsrContext, ChunkFlag, EngineDef
from ytverbatim.contract.models import Word

HYPOTHESES_FILE = "hypotheses.jsonl"
METRICS_FILE = "metrics.json"
ERROR_FILE = "error.json"
REQUEST_FILE = "request.json"
FINGERPRINT_FILE = "fingerprint.json"

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChunkHypothesis(_Model):
    """One line of `hypotheses.jsonl`; times are absolute seconds in the source file."""

    hypothesis: str
    file: str
    chunk: Annotated[int, Field(ge=0)]
    start: Seconds
    end: Seconds
    text: str
    language: str | None = None
    words: tuple[Word, ...] = ()
    generated_tokens: int | None = None
    max_tokens: int | None = None
    confidence: float | None = Field(default=None, allow_inf_nan=False)
    flags: tuple[ChunkFlag, ...] = ()

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError(f"chunk {self.chunk} of {self.file} ends ({self.end}) before it starts ({self.start})")
        return self


class RunMetrics(_Model):
    """What one worker invocation did. A resumed run counts only the chunks it decoded itself."""

    hypothesis: str
    files: int
    chunks_decoded: int
    chunks_skipped: int
    load_s: Seconds
    decode_s: Seconds
    audio_s: Seconds
    peak_memory_bytes: int | None
    max_rss_bytes: int
    # The worker's own lifetime peak physical footprint, which the memory watch's cap is measured against.
    peak_footprint_bytes: int | None = None


class FileJob(_Model):
    """One WAV to transcribe. Without `chunks` the worker plans them with the VAD.

    `language` is the metadata language, as production would pass it; the engine's language policy decides whether
    the engine sees it.
    """

    id: str
    wav: str
    language: str | None = None
    context: AsrContext | None = None
    chunks: tuple[Chunk, ...] | None = None


class WorkerRequest(_Model):
    engine: EngineDef
    out_dir: str
    files: tuple[FileJob, ...]
    hf_hub_cache: str | None = None
    # The parent kills the worker above this footprint; the worker also hands it to MLX as its memory limit.
    memory_cap_bytes: Annotated[int, Field(gt=0)] | None = None
    # Off only for diagnostics: the guard hides an engine's raw failure rate by re-decoding what failed.
    quality_guard: bool = True

    @model_validator(mode="after")
    def _plans_match_chunking(self) -> Self:
        ids = [job.id for job in self.files]
        if len(set(ids)) != len(ids):
            raise ValueError("file ids must be unique within a request")
        if self.engine.spec.max_chunk_s is None and any(job.chunks is not None for job in self.files):
            raise ValueError(f"{self.engine.spec.id} decodes whole files and cannot take a chunk plan")
        self.engine.language_policy()
        return self

    def fingerprint(self) -> dict[str, JsonValue]:
        """Everything that decides the decoded text; where the output and the weights live and the memory cap do not."""
        return self.model_dump(mode="json", exclude={"out_dir", "hf_hub_cache", "memory_cap_bytes"})


class WorkerError(_Model):
    """The structured failure the worker leaves in `error.json` for the parent to read."""

    type: str
    message: str
    traceback: str
