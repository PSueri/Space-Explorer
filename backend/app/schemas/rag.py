from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from backend.app.models.rag import RagDiagnostics


class AskSourcesRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    question: str = Field(min_length=1, max_length=500)


class CitationResponse(BaseModel):
    number: int
    chunk_id: str
    title: str
    url: HttpUrl
    excerpt: str


class AskSourcesResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    citations: list[CitationResponse]
    diagnostics: RagDiagnostics | None = None
