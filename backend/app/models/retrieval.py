from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from backend.app.models.chunk import DocumentChunk
from backend.app.models.source import SourceDefinition


class IndexedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: DocumentChunk
    source: SourceDefinition
    fetched_url: HttpUrl
    retrieved_at: datetime
    content_hash: str
    parser_version: str
    chunker_version: str


class SearchResult(BaseModel):
    score: float
    record: IndexedChunk


class IndexManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    collection_name: str
    embedding_model: str
    embedding_digest: str
    vector_dimension: int = Field(gt=0)
    created_at: datetime
