from datetime import UTC, datetime

import pytest

from backend.app.core.index_configuration import validate_index_configuration
from backend.app.core.settings import Settings
from backend.app.models.retrieval import IndexManifest


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        embedding_model="test-embedding",
        embedding_document_prefix="search_document: ",
        embedding_query_prefix="search_query: ",
        embedding_context_size=512,
    )


def make_manifest() -> IndexManifest:
    return IndexManifest(
        collection_name="test_collection",
        embedding_model="test-embedding",
        embedding_digest="test-digest",
        vector_dimension=3,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        document_prefix="search_document: ",
        query_prefix="search_query: ",
        embedding_context_size=512,
    )


def test_matching_configuration_is_accepted() -> None:
    validate_index_configuration(
        make_manifest(),
        make_settings(),
        "test-digest",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_model", "another-model"),
        ("embedding_digest", "another-digest"),
        ("document_prefix", ""),
        ("query_prefix", ""),
        ("embedding_context_size", 1024),
    ],
)
def test_incompatible_index_configuration_is_rejected(
    field: str,
    value,
) -> None:
    manifest = make_manifest().model_copy(update={field: value})

    with pytest.raises(ValueError, match=field):
        validate_index_configuration(
            manifest,
            make_settings(),
            "test-digest",
        )
