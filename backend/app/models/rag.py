from dataclasses import dataclass
from typing import Literal


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
