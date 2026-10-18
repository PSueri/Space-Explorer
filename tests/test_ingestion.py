import hashlib
import json

import httpx2
import pytest

from backend.app.ingestion.html import extract_main_text
from backend.app.models.source import SourceDefinition
from scripts.ingest_sources import ingest_source, load_sources, save_document

ARTICLE_TEXT = (
    "Example mission content for testing document ingestion. "
    "This paragraph describes a fictional spacecraft and its instruments. "
    "It is long enough to pass the minimum extracted content check."
)

HTML = f"""
<html>
    <body>
        <nav>Global navigation</nav>
        <main>
            <h1>Example mission</h1>
            <nav>Local navigation</nav>
            <p>{ARTICLE_TEXT}</p>
            <script>unwanted_script()</script>
            <footer>Footer content</footer>
        </main>
    </body>
</html>
"""


def make_source() -> SourceDefinition:
    return SourceDefinition(
        source_id="example-mission",
        title="Example mission",
        publisher="Example publisher",
        url="https://example.com/mission/",
        usage_guidelines_url="https://example.com/usage/",
    )


def test_extraction_keeps_article_and_removes_navigation() -> None:
    text = extract_main_text(HTML)

    assert "Example mission" in text
    assert ARTICLE_TEXT in text
    assert "Global navigation" not in text
    assert "Local navigation" not in text
    assert "unwanted_script" not in text
    assert "Footer content" not in text


@pytest.mark.parametrize(
    "html",
    [
        "<html><body><p>No main element.</p></body></html>",
        "<main><p>Too short.</p></main>",
    ],
)
def test_extraction_rejects_missing_content(html: str) -> None:
    with pytest.raises(ValueError):
        extract_main_text(html)


def test_ingestion_preserves_provenance_and_saved_content(tmp_path) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            text=HTML,
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    source = make_source()

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        document = ingest_source(client, source)

    path = save_document(document, tmp_path)
    stored = json.loads(path.read_text(encoding="utf-8"))

    assert stored["source"]["source_id"] == source.source_id
    assert stored["fetched_url"] == str(source.url)
    assert stored["retrieved_at"]
    assert stored["text"] == extract_main_text(HTML)
    assert (
        stored["content_hash"]
        == hashlib.sha256(stored["text"].encode("utf-8")).hexdigest()
    )


def test_source_registry_rejects_duplicate_identifiers(tmp_path) -> None:
    source = make_source().model_dump(mode="json")
    path = tmp_path / "sources.json"
    path.write_text(json.dumps([source, source]), encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        load_sources(path)
