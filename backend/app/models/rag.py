from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from backend.app.models.retrieval import IndexManifest, SearchResult

AbstentionReason = Literal[
    "no_retrieval_results",
    "model_abstained",
    "missing_citations",
    "unknown_citations",
]


class RetrievalDiagnostics(BaseModel):
    duration_seconds: float
    requested_limit: int
    results: list[SearchResult]
    index_manifest: IndexManifest | None = None


class GenerationConfiguration(BaseModel):
    provider: str
    requested_model: str
    context_size: int
    max_output_tokens: int
    think: bool = False
    temperature: float | None = None
    temperature_source: Literal["provider_default"] = "provider_default"


class GenerationDiagnostics(BaseModel):
    attempted: bool
    duration_seconds: float | None = None
    raw_answer: str | None = None
    reported_model: str | None = None
    done_reason: str | None = None
    output_truncated: bool | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


class RagDiagnostics(BaseModel):
    diagnostics_version: Literal["1"] = "1"
    configuration: GenerationConfiguration
    retrieval: RetrievalDiagnostics
    generation: GenerationDiagnostics
    abstention_reason: AbstentionReason | None = None


@dataclass(frozen=True)
class Citation:
    number: int
    chunk_id: str
    title: str
    url: str
    excerpt: str


@dataclass(frozen=True)
class GroundedAnswer:
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    citations: tuple[Citation, ...]
    diagnostics: RagDiagnostics | None = None
