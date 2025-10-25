"""Graph database, embeddings, and retrieval layer."""

from .embeddings import EmbeddingModel, create_embedding_model
from .indexing import (
    backfill_embeddings,
    ensure_vector_indexes,
    query_similar_entities,
    query_similar_scenes,
)

__all__ = [
    "EmbeddingModel",
    "create_embedding_model",
    "backfill_embeddings",
    "ensure_vector_indexes",
    "query_similar_entities",
    "query_similar_scenes",
]
