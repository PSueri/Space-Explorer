import argparse
import sys
from pathlib import Path

from backend.app.core.settings import PROJECT_ROOT
from backend.app.ingestion.chunking import chunk_document
from backend.app.models.chunk import ChunkedDocument
from backend.app.models.source import SourceDocument
from scripts.ingest_sources import load_sources

SOURCES_FILE = PROJECT_ROOT / "data" / "sources.json"
CORPUS_DIRECTORY = PROJECT_ROOT / "data" / "corpus"
CHUNKS_DIRECTORY = PROJECT_ROOT / "data" / "chunks"


def save_chunks(document: ChunkedDocument, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)

    destination = directory / f"{document.source.source_id}.json"
    temporary_path = destination.with_suffix(".json.tmp")

    temporary_path.write_text(
        document.model_dump_json(indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(destination)

    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split acquired source documents into traceable chunks."
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    arguments = parser.parse_args()

    try:
        sources = load_sources(SOURCES_FILE)

        for source in sources:
            document_path = CORPUS_DIRECTORY / f"{source.source_id}.json"
            document = SourceDocument.model_validate_json(
                document_path.read_text(encoding="utf-8")
            )

            if document.source != source:
                raise ValueError(
                    f"Stored metadata differs for {source.source_id}. "
                    "Run source ingestion again."
                )

            chunked_document = chunk_document(
                document,
                chunk_size=arguments.chunk_size,
                overlap=arguments.overlap,
            )
            destination = save_chunks(chunked_document, CHUNKS_DIRECTORY)

            print(
                f"Saved {source.source_id}: "
                f"{len(chunked_document.chunks)} chunks -> {destination}"
            )
    except (ValueError, OSError) as exc:
        print(f"Chunking failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
