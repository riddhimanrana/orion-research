"""Graph database, embeddings, and retrieval layer."""

from .embeddings import EmbeddingModel, create_embedding_model
# Removed: indexing and database modules (obsolete - we use Memgraph now)


__all__ = [
    "EmbeddingModel",
    "create_embedding_model",
]

