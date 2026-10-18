import hashlib
from datetime import UTC, datetime

import pytest

from backend.app.ingestion.chunking import chunk_document
from backend.app.models.source import SourceDefinition, SourceDocument


def make_document(text: str) -> SourceDocument:
    return SourceDocument(
        source=SourceDefinition(
            source_id="example-mission",
            title="Example mission",
            publisher="Example publisher",
            url="https://example.com/mission/",
            usage_guidelines_url="https://example.com/usage/",
        ),
        fetched_url="https://example.com/mission/",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        text=text,
    )


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(1000, 150), (300, 0), (10, 9)],
)
def test_chunks_preserve_complete_text_and_exact_offsets(
    chunk_size: int,
    overlap: int,
) -> None:
    document = make_document(
        ("Example mission text with several words. " * 25 + "\n\n") * 3
    )

    result = chunk_document(document, chunk_size, overlap)

    assert result.chunks[0].start == 0
    assert result.chunks[-1].end == len(document.text)

    for chunk in result.chunks:
        assert chunk.text == document.text[chunk.start : chunk.end]
        assert 0 < len(chunk.text) <= chunk_size

    reconstructed = result.chunks[0].text

    for previous, current in zip(result.chunks, result.chunks[1:]):
        assert current.start == previous.end - overlap
        assert current.start > previous.start
        reconstructed += current.text[overlap:]

    assert reconstructed == document.text


def test_chunking_prefers_paragraph_boundaries() -> None:
    document = make_document("A" * 600 + "\n\n" + "B" * 600)

    result = chunk_document(document, chunk_size=1000, overlap=150)

    assert result.chunks[0].end == 602


def test_chunk_identifiers_are_stable_and_configuration_sensitive() -> None:
    document = make_document("Example mission content. " * 100)

    first = chunk_document(document, chunk_size=1000, overlap=150)
    repeated = chunk_document(document, chunk_size=1000, overlap=150)
    changed = chunk_document(document, chunk_size=800, overlap=100)

    first_ids = [chunk.chunk_id for chunk in first.chunks]
    repeated_ids = [chunk.chunk_id for chunk in repeated.chunks]
    changed_ids = [chunk.chunk_id for chunk in changed.chunks]

    assert first_ids == repeated_ids
    assert first_ids != changed_ids
    assert len(first_ids) == len(set(first_ids))


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (100, -1), (100, 100), (100, 101)],
)
def test_chunking_rejects_invalid_parameters(
    chunk_size: int,
    overlap: int,
) -> None:
    document = make_document("Example mission content. " * 10)

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size, overlap)


def test_chunking_rejects_content_with_an_incorrect_hash() -> None:
    document = make_document("Example mission content. " * 10)
    modified = document.model_copy(
        update={"text": document.text + " Unexpected change."}
    )

    with pytest.raises(ValueError, match="hash"):
        chunk_document(modified)
