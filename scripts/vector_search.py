import argparse
import sys
from datetime import UTC, datetime
from uuid import uuid4

import httpx2
from qdrant_client import QdrantClient

from backend.app.core.settings import PROJECT_ROOT, Settings
from backend.app.integrations.ollama_embeddings import OllamaEmbeddingClient
from backend.app.models.chunk import ChunkedDocument
from backend.app.models.retrieval import IndexedChunk, IndexManifest
from backend.app.models.source import SourceDocument
from backend.app.repositories.qdrant import QdrantRepository
from scripts.ingest_sources import load_sources

INDEX_DIRECTORY = PROJECT_ROOT / "data" / "vector-index"
DATABASE_DIRECTORY = INDEX_DIRECTORY / "database"
MANIFEST_FILE = INDEX_DIRECTORY / "current.json"


def load_records() -> list[IndexedChunk]:
    sources = load_sources(PROJECT_ROOT / "data" / "sources.json")
    records: list[IndexedChunk] = []

    for source in sources:
        corpus_path = PROJECT_ROOT / "data" / "corpus" / f"{source.source_id}.json"
        chunks_path = PROJECT_ROOT / "data" / "chunks" / f"{source.source_id}.json"

        document = SourceDocument.model_validate_json(
            corpus_path.read_text(encoding="utf-8")
        )
        chunked = ChunkedDocument.model_validate_json(
            chunks_path.read_text(encoding="utf-8")
        )

        if (
            document.source != source
            or chunked.source != source
            or chunked.content_hash != document.content_hash
            or chunked.parser_version != document.parser_version
        ):
            raise ValueError(
                f"Source snapshots are inconsistent: {source.source_id}. "
                "Run ingestion and chunking again."
            )

        for chunk in chunked.chunks:
            if chunk.text != document.text[chunk.start : chunk.end]:
                raise ValueError(f"Chunk offsets are invalid: {chunk.chunk_id}")

            records.append(
                IndexedChunk(
                    chunk=chunk,
                    source=chunked.source,
                    fetched_url=chunked.fetched_url,
                    retrieved_at=chunked.retrieved_at,
                    content_hash=chunked.content_hash,
                    parser_version=chunked.parser_version,
                    chunker_version=chunked.chunker_version,
                )
            )

    if not records:
        raise ValueError("No chunks are available for indexing.")

    return records


def build_index(
    settings: Settings,
    embedder: OllamaEmbeddingClient,
) -> None:
    records = load_records()
    digest = embedder.model_digest()
    vectors = embedder.embed([record.chunk.text for record in records])

    if embedder.model_digest() != digest:
        raise ValueError("The embedding model changed during indexing. Try again.")

    collection_name = f"documents_{uuid4().hex}"
    INDEX_DIRECTORY.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(DATABASE_DIRECTORY))

    try:
        repository = QdrantRepository(client, collection_name)
        repository.create(dimension=len(vectors[0]))
        repository.upsert(records, vectors)
    finally:
        client.close()

    manifest = IndexManifest(
        collection_name=collection_name,
        embedding_model=settings.embedding_model,
        embedding_digest=digest,
        vector_dimension=len(vectors[0]),
        created_at=datetime.now(UTC),
    )

    temporary_path = MANIFEST_FILE.with_suffix(".json.tmp")
    temporary_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(MANIFEST_FILE)

    print(f"Indexed {len(records)} chunks.")
    print(f"Embedding dimensions: {manifest.vector_dimension}")
    print(f"Active collection: {collection_name}")


def search_index(
    settings: Settings,
    embedder: OllamaEmbeddingClient,
    query: str,
) -> None:
    manifest = IndexManifest.model_validate_json(
        MANIFEST_FILE.read_text(encoding="utf-8")
    )

    if (
        manifest.embedding_model != settings.embedding_model
        or manifest.embedding_digest != embedder.model_digest()
    ):
        raise ValueError(
            "The embedding model differs from the indexed model. "
            "Rebuild the index before searching."
        )

    vector = embedder.embed([query])[0]

    if len(vector) != manifest.vector_dimension:
        raise ValueError("Query vector dimensions do not match the index.")

    client = QdrantClient(path=str(DATABASE_DIRECTORY))

    try:
        repository = QdrantRepository(client, manifest.collection_name)
        results = repository.search(vector, limit=3)
    finally:
        client.close()

    for position, result in enumerate(results, start=1):
        record = result.record

        print(f"\nResult {position} | cosine score: {result.score:.4f}")
        print(f"Source: {record.source.title}")
        print(f"URL: {record.fetched_url}")
        print(f"Chunk: {record.chunk.chunk_id}")
        print(f"Offsets: {record.chunk.start}:{record.chunk.end}")
        print(record.chunk.text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and search the local document vector index."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index")

    search_command = commands.add_parser("search")
    search_command.add_argument("query")

    arguments = parser.parse_args()

    try:
        settings = Settings()

        with httpx2.Client(
            base_url=str(settings.embedding_base_url),
            timeout=settings.embedding_timeout_seconds,
            trust_env=False,
        ) as http_client:
            embedder = OllamaEmbeddingClient(http_client, settings)

            if arguments.command == "index":
                build_index(settings, embedder)
            else:
                search_index(settings, embedder, arguments.query)
    except (httpx2.HTTPError, ValueError, OSError, RuntimeError) as exc:
        print(f"Vector operation failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
