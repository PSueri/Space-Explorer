import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx2
from pydantic import TypeAdapter

from backend.app.core.settings import PROJECT_ROOT
from backend.app.ingestion.html import extract_main_text
from backend.app.models.source import SourceDefinition, SourceDocument

SOURCES_FILE = PROJECT_ROOT / "data" / "sources.json"
CORPUS_DIRECTORY = PROJECT_ROOT / "data" / "corpus"


def load_sources(path: Path) -> list[SourceDefinition]:
    sources = TypeAdapter(list[SourceDefinition]).validate_json(
        path.read_text(encoding="utf-8")
    )

    if not sources:
        raise ValueError("The source registry is empty.")

    identifiers = [source.source_id for source in sources]

    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Source identifiers must be unique.")

    return sources


def ingest_source(
    client: httpx2.Client,
    source: SourceDefinition,
) -> SourceDocument:
    response = client.get(str(source.url))
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    if not content_type.startswith("text/html"):
        raise ValueError(f"Expected HTML for source: {source.source_id}")

    text = extract_main_text(response.text)
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return SourceDocument(
        source=source,
        fetched_url=str(response.url),
        retrieved_at=datetime.now(UTC),
        content_hash=content_hash,
        text=text,
    )


def save_document(
    document: SourceDocument,
    directory: Path,
) -> Path:
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
    try:
        sources = load_sources(SOURCES_FILE)

        with httpx2.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "SpaceExplorerPortfolio/0.1"},
        ) as client:
            for source in sources:
                document = ingest_source(client, source)
                destination = save_document(document, CORPUS_DIRECTORY)

                print(
                    f"Saved {source.source_id}: "
                    f"{len(document.text)} characters -> {destination}"
                )
    except (httpx2.HTTPError, ValueError, OSError) as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
