import hashlib

from backend.app.models.chunk import ChunkedDocument, DocumentChunk
from backend.app.models.source import SourceDocument

CHUNKER_VERSION = "1"


def chunk_document(
    document: SourceDocument,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> ChunkedDocument:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    text = document.text
    actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    if actual_hash != document.content_hash:
        raise ValueError("Document content does not match its stored hash.")

    chunks: list[DocumentChunk] = []
    start = 0

    while start < len(text):
        limit = min(start + chunk_size, len(text))
        end = limit

        if limit < len(text):
            minimum = start + chunk_size // 2
            boundary = text.rfind("\n\n", minimum, limit)

            if boundary != -1:
                candidate = boundary + 2
            else:
                boundary = text.rfind(" ", minimum, limit)
                candidate = boundary + 1 if boundary != -1 else limit

            if candidate > start + overlap:
                end = candidate

        identity = (
            f"{document.source.source_id}:"
            f"{document.content_hash}:"
            f"{document.parser_version}:"
            f"{CHUNKER_VERSION}:"
            f"{chunk_size}:{overlap}:{start}:{end}"
        )
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                index=len(chunks),
                start=start,
                end=end,
                text=text[start:end],
            )
        )

        if end == len(text):
            break

        start = end - overlap

    return ChunkedDocument(
        source=document.source,
        fetched_url=document.fetched_url,
        retrieved_at=document.retrieved_at,
        content_hash=document.content_hash,
        parser_version=document.parser_version,
        chunker_version=CHUNKER_VERSION,
        chunk_size=chunk_size,
        overlap=overlap,
        chunks=chunks,
    )
