from backend.app.core.settings import Settings
from backend.app.models.retrieval import IndexManifest


def validate_index_configuration(
    manifest: IndexManifest,
    settings: Settings,
    model_digest: str,
) -> None:
    expected = {
        "embedding_model": settings.embedding_model,
        "embedding_digest": model_digest,
        "document_prefix": settings.embedding_document_prefix,
        "query_prefix": settings.embedding_query_prefix,
        "embedding_context_size": settings.embedding_context_size,
    }

    differences = [
        name for name, value in expected.items() if getattr(manifest, name) != value
    ]

    if differences:
        fields = ", ".join(differences)

        raise ValueError(
            f"Embedding configuration differs from the index: {fields}. "
            "Rebuild the index before searching."
        )
