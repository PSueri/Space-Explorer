from typing import Protocol

from backend.app.models.retrieval import SearchResult


class DocumentRetriever(Protocol):
    def search(
        self,
        question: str,
        limit: int = 2,
    ) -> list[SearchResult]: ...
