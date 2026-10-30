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
from backend.app.models.generation import GenerationResult
from backend.app.models.retrieval import (
    IndexedChunk,
    IndexManifest,
    RetrievalBatch,
    SearchResult,
)
from backend.app.models.source import SourceDefinition
from backend.app.services.rag import RagService


def make_result(
    chunk_id: str = "example-chunk",
    text: str = "The fictional probe landed on Example Moon.",
) -> SearchResult:
    return SearchResult(
        score=0.8,
        record=IndexedChunk(
            chunk=DocumentChunk(
                chunk_id=chunk_id,
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
        self.results = [
            make_result(),
            make_result("unused-chunk", "Another retrieved passage."),
        ]
        self.error = None
        self.calls = 0

    def search(self, question: str, limit: int = 2) -> RetrievalBatch:
        self.calls += 1

        if self.error is not None:
            raise self.error

        return RetrievalBatch(
            results=self.results[:limit],
            index_manifest=IndexManifest(
                collection_name="test_collection",
                embedding_model="test-embedding",
                embedding_digest="test-digest",
                vector_dimension=3,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        )


class FakeLLM:
    def __init__(self) -> None:
        self.response = "The probe landed on Example Moon [1]."
        self.done_reason = "stop"
        self.error = None
        self.calls = 0
        self.last_input = ""

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        history=(),
    ) -> str:
        result = await self.generate_with_details(
            system_prompt,
            user_message,
            history,
        )
        return result.answer.strip()

    async def generate_with_details(
        self,
        system_prompt: str,
        user_message: str,
        history=(),
    ) -> GenerationResult:
        self.calls += 1
        self.last_input = user_message

        if self.error is not None:
            raise self.error

        return GenerationResult(
            answer=self.response,
            model="fake-model",
            done_reason=self.done_reason,
            prompt_eval_count=100,
            eval_count=20,
        )


@pytest.fixture
def rag_context():
    retriever = FakeRetriever()
    llm = FakeLLM()
    settings = Settings(
        _env_file=None,
        name="Space Explorer Test",
        debug=False,
        rag_diagnostics_enabled=True,
    )
    service = RagService(llm, retriever, settings)
    application = create_app(settings)
    application.dependency_overrides[get_rag_service] = lambda: service

    with TestClient(application) as client:
        yield client, retriever, llm, settings


def test_answer_includes_retrieved_citation(rag_context) -> None:
    client, _, llm, _ = rag_context
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


def test_diagnostics_keep_uncited_retrieval_and_index_metadata(rag_context) -> None:
    client, _, _, _ = rag_context
    body = client.post(
        "/ask-sources",
        json={"question": "Where did the fictional probe land?"},
    ).json()

    assert len(body["citations"]) == 1
    trace = body["diagnostics"]
    assert len(trace["retrieval"]["results"]) == 2
    assert trace["retrieval"]["results"][1]["record"]["chunk"]["chunk_id"] == (
        "unused-chunk"
    )
    manifest = trace["retrieval"]["index_manifest"]
    assert manifest["collection_name"] == "test_collection"
    assert manifest["embedding_digest"] == "test-digest"
    assert trace["configuration"]["requested_model"] == "qwen3:0.6b"
    assert trace["generation"]["reported_model"] == "fake-model"


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ("An answer without citations.", "missing_citations"),
        ("An answer with an unknown citation [99].", "unknown_citations"),
        ("An answer with both citations [1] [99].", "unknown_citations"),
        ("INSUFFICIENT_EVIDENCE", "model_abstained"),
    ],
)
def test_abstention_records_reason_and_original_answer(
    rag_context,
    answer: str,
    reason: str,
) -> None:
    client, _, llm, _ = rag_context
    llm.response = f"  {answer}\n"

    response = client.post(
        "/ask-sources",
        json={"question": "Where did the fictional probe land?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_evidence"
    assert body["citations"] == []
    assert body["diagnostics"]["abstention_reason"] == reason
    assert body["diagnostics"]["generation"]["raw_answer"] == llm.response


def test_empty_retrieval_skips_generation(rag_context) -> None:
    client, retriever, llm, _ = rag_context
    retriever.results = []

    body = client.post(
        "/ask-sources",
        json={"question": "Where did the fictional probe land?"},
    ).json()

    assert body["status"] == "insufficient_evidence"
    assert llm.calls == 0
    assert body["diagnostics"]["abstention_reason"] == "no_retrieval_results"
    assert body["diagnostics"]["generation"]["attempted"] is False
    assert body["diagnostics"]["generation"].get("raw_answer") is None


@pytest.mark.parametrize(
    ("done_reason", "expected"),
    [("length", True), ("stop", False), (None, None), ("other", None)],
)
def test_generation_finish_reason_is_recorded_without_changing_decision(
    rag_context,
    done_reason,
    expected,
) -> None:
    client, _, llm, _ = rag_context
    llm.done_reason = done_reason

    body = client.post(
        "/ask-sources",
        json={"question": "Where did the fictional probe land?"},
    ).json()

    assert body["status"] == "answered"
    assert body["diagnostics"]["generation"].get("output_truncated") is expected
    assert body["diagnostics"]["generation"]["eval_count"] == 20


def test_disabled_diagnostics_preserve_public_response(rag_context) -> None:
    client, _, _, settings = rag_context
    settings.rag_diagnostics_enabled = False

    body = client.post(
        "/ask-sources",
        json={"question": "Where did the fictional probe land?"},
    ).json()

    assert set(body) == {"status", "answer", "citations"}


def test_diagnostics_do_not_change_answer_or_citations(rag_context) -> None:
    client, _, _, settings = rag_context
    request = {"question": "Where did the fictional probe land?"}

    enabled = client.post("/ask-sources", json=request).json()
    settings.rag_diagnostics_enabled = False
    disabled = client.post("/ask-sources", json=request).json()

    enabled.pop("diagnostics")
    assert enabled == disabled


def test_retrieval_failure_returns_503(rag_context) -> None:
    client, retriever, llm, _ = rag_context
    retriever.error = RetrievalUnavailableError()

    response = client.post(
        "/ask-sources",
        json={"question": "Where did the fictional probe land?"},
    )

    assert response.status_code == 503
    assert llm.calls == 0


def test_generation_timeout_returns_504(rag_context) -> None:
    client, _, llm, _ = rag_context
    llm.error = LLMTimeoutError()

    response = client.post(
        "/ask-sources",
        json={"question": "Where did the fictional probe land?"},
    )

    assert response.status_code == 504


def test_empty_question_is_rejected(rag_context) -> None:
    client, retriever, llm, _ = rag_context
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
