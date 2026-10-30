from pathlib import Path
from threading import Lock

import httpx2
from qdrant_client import QdrantClient

from backend.app.core.exceptions import RetrievalUnavailableError
from backend.app.core.settings import PROJECT_ROOT, Settings
from backend.app.integrations.ollama_embeddings import OllamaEmbeddingClient
from backend.app.models.retrieval import IndexManifest, SearchResult
from backend.app.repositories.qdrant import QdrantRepository


class LocalDocumentRetriever:
    def __init__(
        self,
        settings: Settings,
        index_directory: Path | None = None,
    ) -> None:
        self._settings = settings
        self._index_directory = (
            index_directory
            if index_directory is not None
            else PROJECT_ROOT / "data" / "vector-index"
        )
        self._lock = Lock()

    def search(
        self,
        question: str,
        limit: int = 2,
    ) -> list[SearchResult]:
        try:
            with self._lock:
                return self._search(question, limit)
        except (httpx2.HTTPError, ValueError, OSError, RuntimeError) as exc:
            raise RetrievalUnavailableError(
                "The document index or embedding provider is unavailable."
            ) from exc

    def _search(
        self,
        question: str,
        limit: int,
    ) -> list[SearchResult]:
        manifest_path = self._index_directory / "current.json"
        database_path = self._index_directory / "database"

        manifest = IndexManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )

        if not database_path.is_dir():
            raise ValueError("The vector database directory is missing.")

        with httpx2.Client(
            base_url=str(self._settings.embedding_base_url),
            timeout=self._settings.embedding_timeout_seconds,
            trust_env=False,
        ) as http_client:
            embedder = OllamaEmbeddingClient(http_client, self._settings)

            if (
                manifest.embedding_model != self._settings.embedding_model
                or manifest.embedding_digest != embedder.model_digest()
            ):
                raise ValueError(
                    "The configured embedding model differs from the index."
                )

            vector = embedder.embed([question])[0]

        if len(vector) != manifest.vector_dimension:
            raise ValueError("The query vector has incompatible dimensions.")

        client = QdrantClient(path=str(database_path))

        try:
            repository = QdrantRepository(client, manifest.collection_name)
            return repository.search(vector, limit=limit)
        finally:
            client.close()
