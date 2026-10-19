from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from backend.app.models.source import SourceDefinition


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)


class ChunkedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: SourceDefinition
    fetched_url: HttpUrl
    retrieved_at: datetime
    content_hash: str
    parser_version: Literal["1", "2"]

    chunker_version: Literal["1", "2"] = "2"
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)
    chunks: list[DocumentChunk]
