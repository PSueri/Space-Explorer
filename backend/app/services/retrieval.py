from typing import Protocol

from backend.app.models.retrieval import RetrievalBatch


class DocumentRetriever(Protocol):
    def search(
        self,
        question: str,
        limit: int = 2,
    ) -> RetrievalBatch: ...
