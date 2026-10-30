from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from backend.app.core.dependencies import get_rag_service
from backend.app.core.exceptions import (
    LLMTimeoutError,
    RetrievalUnavailableError,
)
from backend.app.core.settings import Settings
from backend.app.integrations.local_retrieval import LocalDocumentRetriever
from backend.app.main import create_app
from backend.app.models.chunk import DocumentChunk
from backend.app.models.retrieval import IndexedChunk, SearchResult
from backend.app.models.source import SourceDefinition
from backend.app.services.rag import RagService


def make_result() -> SearchResult:
    text = "The fictional probe landed on Example Moon."

    return SearchResult(
        score=0.8,
        record=IndexedChunk(
            chunk=DocumentChunk(
                chunk_id="example-chunk",
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
        ),
    )


class FakeRetriever:
    def __init__(self) -> None:
        self.results = [make_result()]
        self.error = None
        self.calls = 0

    def search(self, question: str, limit: int = 2) -> list[SearchResult]:
        self.calls += 1

        if self.error is not None:
            raise self.error

        return self.results[:limit]


class FakeLLM:
    def __init__(self) -> None:
        self.response = "The probe landed on Example Moon [1]."
        self.error = None
        self.calls = 0
        self.last_input = ""

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        history=(),
    ) -> str:
        self.calls += 1
        self.last_input = user_message

        if self.error is not None:
            raise self.error

        return self.response


@pytest.fixture
def rag_context():
    retriever = FakeRetriever()
    llm = FakeLLM()
    service = RagService(llm, retriever)

    application = create_app(
        Settings(
            _env_file=None,
            name="Space Explorer Test",
            debug=False,
        )
    )
    application.dependency_overrides[get_rag_service] = lambda: service

    with TestClient(application) as client:
        yield client, retriever, llm


def test_answer_includes_retrieved_citation(rag_context) -> None:
    client, _, llm = rag_context

    response = client.post(
        "/ask-sources",
        json={"question": "Where did the fictional probe land?"},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "answered"
    assert body["citations"][0]["number"] == 1
    assert body["citations"][0]["chunk_id"] == "example-chunk"
    assert body["citations"][0]["excerpt"] == make_result().record.chunk.text
    assert make_result().record.chunk.text in llm.last_input


@pytest.mark.parametrize(
    "answer",
    [
        "An answer without citations.",
        "An answer with an unknown citation [99].",
        "An answer with both valid and unknown citations [1] [99].",
        "INSUFFICIENT_EVIDENCE",
    ],
)
def test_unsupported_reference_format_causes_abstention(
    rag_context,
    answer: str,
) -> None:
    client, _, llm = rag_context
    llm.response = answer

    response = client.post(
        "/ask-sources",
        json={"question": "Where did the fictional probe land?"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert response.json()["citations"] == []


def test_empty_retrieval_skips_generation(rag_context) -> None:
    client, retriever, llm = rag_context
    retriever.results = []

    response = client.post(
        "/ask-sources",
        json={"question": "Where did the fictional probe land?"},
    )

    assert response.json()["status"] == "insufficient_evidence"
    assert llm.calls == 0


def test_retrieval_failure_returns_503(rag_context) -> None:
    client, retriever, llm = rag_context
    retriever.error = RetrievalUnavailableError()

    response = client.post(
        "/ask-sources",
        json={"question": "Where did the fictional probe land?"},
    )

    assert response.status_code == 503
    assert llm.calls == 0


def test_generation_timeout_returns_504(rag_context) -> None:
    client, _, llm = rag_context
    llm.error = LLMTimeoutError()

    response = client.post(
        "/ask-sources",
        json={"question": "Where did the fictional probe land?"},
    )

    assert response.status_code == 504


def test_empty_question_is_rejected(rag_context) -> None:
    client, retriever, llm = rag_context

    response = client.post("/ask-sources", json={"question": "   "})

    assert response.status_code == 422
    assert retriever.calls == 0
    assert llm.calls == 0


def test_missing_index_is_reported_without_contacting_ollama(tmp_path) -> None:
    retriever = LocalDocumentRetriever(
        Settings(_env_file=None),
        index_directory=tmp_path / "missing-index",
    )

    with pytest.raises(RetrievalUnavailableError):
        retriever.search("Example question")
