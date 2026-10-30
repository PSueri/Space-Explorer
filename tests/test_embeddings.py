import json

import httpx2
import pytest

from backend.app.core.settings import Settings
from backend.app.integrations.ollama_embeddings import OllamaEmbeddingClient


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        embedding_model="test-embedding",
    )


def test_embedding_request_preserves_input_and_disables_truncation() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)

        assert request.url.path == "/api/embed"
        assert body["input"] == ["First text", "Second text"]
        assert body["truncate"] is False

        return httpx2.Response(
            200,
            json={"embeddings": [[1.0, 0.0], [0.0, 1.0]]},
        )

    with httpx2.Client(
        transport=httpx2.MockTransport(handler),
        base_url="http://localhost:11434",
    ) as client:
        embedder = OllamaEmbeddingClient(client, make_settings())
        vectors = embedder.embed(["First text", "Second text"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [[]],
        [[0.0, 0.0]],
        [[1.0, 0.0], [1.0]],
    ],
)
def test_invalid_embedding_responses_are_rejected(vectors: list) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"embeddings": vectors})

    with httpx2.Client(
        transport=httpx2.MockTransport(handler),
        base_url="http://localhost:11434",
    ) as client:
        embedder = OllamaEmbeddingClient(client, make_settings())

        with pytest.raises(ValueError):
            embedder.embed(["Example text"])


def test_model_digest_is_resolved_from_installed_models() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={"models": [{"name": "test-embedding", "digest": "example-digest"}]},
        )

    with httpx2.Client(
        transport=httpx2.MockTransport(handler),
        base_url="http://localhost:11434",
    ) as client:
        embedder = OllamaEmbeddingClient(client, make_settings())

        assert embedder.model_digest() == "example-digest"
