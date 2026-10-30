from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from backend.app.models.retrieval import IndexedChunk, SearchResult


class QdrantRepository:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
    ) -> None:
        self._client = client
        self._collection_name = collection_name

    def create(self, dimension: int) -> None:
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert(
        self,
        records: list[IndexedChunk],
        vectors: list[list[float]],
    ) -> None:
        if not records or len(records) != len(vectors):
            raise ValueError("Records and vectors must have matching non-zero counts.")

        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, record.chunk.chunk_id)),
                vector=vector,
                payload=record.model_dump(mode="json"),
            )
            for record, vector in zip(records, vectors, strict=True)
        ]

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        vector: list[float],
        limit: int = 3,
    ) -> list[SearchResult]:
        if limit < 1:
            raise ValueError("Search limit must be positive.")

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [
            SearchResult(
                score=point.score,
                record=IndexedChunk.model_validate(point.payload or {}),
            )
            for point in response.points
        ]
