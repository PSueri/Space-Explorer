from datetime import UTC, datetime

import pytest
from qdrant_client import QdrantClient

from backend.app.models.chunk import DocumentChunk
from backend.app.models.retrieval import IndexedChunk
from backend.app.models.source import SourceDefinition
from backend.app.repositories.qdrant import QdrantRepository


def make_record(identifier: str, text: str) -> IndexedChunk:
    return IndexedChunk(
        chunk=DocumentChunk(
            chunk_id=identifier,
            index=0,
            start=0,
            end=len(text),
            text=text,
        ),
        source=SourceDefinition(
            source_id="example-mission",
            title="Example mission",
            publisher="Example publisher",
            url="https://example.com/mission/",
            usage_guidelines_url="https://example.com/usage/",
        ),
        fetched_url="https://example.com/mission/",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        content_hash="example-hash",
        parser_version="2",
        chunker_version="2",
    )


def test_repository_persists_vectors_and_returns_matching_payload(tmp_path) -> None:
    first = make_record("a" * 64, "First example passage.")
    second = make_record("b" * 64, "Second example passage.")

    client = QdrantClient(path=str(tmp_path / "database"))

    try:
        repository = QdrantRepository(client, "test_documents")
        repository.create(dimension=3)
        repository.upsert(
            [first, second],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )

        repository.upsert([first], [[1.0, 0.0, 0.0]])

        assert (
            client.count(
                collection_name="test_documents",
                exact=True,
            ).count
            == 2
        )
    finally:
        client.close()

    reopened = QdrantClient(path=str(tmp_path / "database"))

    try:
        repository = QdrantRepository(reopened, "test_documents")
        results = repository.search([1.0, 0.0, 0.0], limit=1)

        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0)
        assert results[0].record == first
    finally:
        reopened.close()
