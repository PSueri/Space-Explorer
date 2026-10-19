from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class SourceDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    url: HttpUrl
    usage_guidelines_url: HttpUrl


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: SourceDefinition
    fetched_url: HttpUrl
    retrieved_at: datetime
    content_hash: str
    text: str = Field(min_length=100)
    parser_version: Literal["1", "2"] = "2"
