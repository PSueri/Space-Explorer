import math

import httpx2
from pydantic import BaseModel, Field, FiniteFloat

from backend.app.core.settings import Settings


class _EmbeddingResponse(BaseModel):
    embeddings: list[list[FiniteFloat]] = Field(min_length=1)


class _ModelTag(BaseModel):
    name: str
    digest: str = Field(min_length=1)


class _ModelList(BaseModel):
    models: list[_ModelTag]


class OllamaEmbeddingClient:
    def __init__(
        self,
        http_client: httpx2.Client,
        settings: Settings,
    ) -> None:
        self._http_client = http_client
        self._model = settings.embedding_model
        self._document_prefix = settings.embedding_document_prefix
        self._query_prefix = settings.embedding_query_prefix
        self._context_size = settings.embedding_context_size

    def model_digest(self) -> str:
        response = self._http_client.get("/api/tags")
        response.raise_for_status()
        available = _ModelList.model_validate(response.json())

        for model in available.models:
            if model.name == self._model:
                return model.digest

        raise ValueError(
            f"Embedding model is not installed: {self._model}. "
            "Download it with ollama pull."
        )

    @staticmethod
    def _validate_inputs(texts: list[str]) -> None:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Embedding inputs must contain non-empty text.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._validate_inputs(texts)

        return self.embed([self._document_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        self._validate_inputs([text])

        return self.embed([self._query_prefix + text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._validate_inputs(texts)

        response = self._http_client.post(
            "/api/embed",
            json={
                "model": self._model,
                "input": texts,
                "truncate": False,
                "keep_alive": "0",
                "options": {"num_ctx": self._context_size},
            },
        )
        response.raise_for_status()

        payload = _EmbeddingResponse.model_validate(response.json())
        vectors = payload.embeddings

        if len(vectors) != len(texts):
            raise ValueError("The number of embeddings does not match the inputs.")

        dimension = len(vectors[0])

        if dimension == 0 or any(len(vector) != dimension for vector in vectors):
            raise ValueError("Embedding dimensions are empty or inconsistent.")

        if any(math.hypot(*vector) == 0 for vector in vectors):
            raise ValueError("The provider returned a zero embedding vector.")

        return vectors
