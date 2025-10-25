"""
Vector Indexing Utilities for Neo4j
===================================

Creates vector indexes and backfills embeddings for Entities and Scenes
so that semantic search (retrieval augmented generation) works well in
the unified VideoQASystem.

This uses the EmbeddingModel abstraction (Ollama or sentence-transformers)
to embed rich context text for nodes and stores the vectors directly on
the nodes, then creates native Neo4j vector indexes that power the
unified :class:`orion.video_qa.VideoQASystem`.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from neo4j import GraphDatabase, Query

from .embeddings import create_embedding_model, EmbeddingModel

logger = logging.getLogger("orion.vector_indexing")


ENTITY_INDEX = "entity_context_index"
SCENE_INDEX = "scene_context_index"
ENTITY_VECTOR_PROP = "context_embedding"
SCENE_VECTOR_PROP = "context_embedding"


def _safe_len(vec: List[float] | np.ndarray | None) -> int:
    if vec is None:
        return 0
    try:
        return len(vec)  # type: ignore[arg-type]
    except Exception:
        return 0


def ensure_vector_indexes(uri: str, user: str, password: str, entity_dim: int, scene_dim: int) -> None:
    """Create Neo4j native vector indexes if they don't exist."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        try:
            # Entity vector index (Neo4j 5+ native vector index)
            session.run(
                Query(
                    """
                    CREATE VECTOR INDEX entity_context_index IF NOT EXISTS FOR (e:Entity) ON (e.context_embedding)
                    OPTIONS { indexConfig: { `vector.dimensions`: $dim, `vector.similarity_function`: 'cosine' } }
                    """
                ), {"dim": int(entity_dim)}
            )
        except Exception as e:
            # Likely already exists; log at debug
            logger.debug(f"Entity vector index create skipped: {e}")

        try:
            # Scene vector index
            session.run(
                Query(
                    """
                    CREATE VECTOR INDEX scene_context_index IF NOT EXISTS FOR (s:Scene) ON (s.context_embedding)
                    OPTIONS { indexConfig: { `vector.dimensions`: $dim, `vector.similarity_function`: 'cosine' } }
                    """
                ), {"dim": int(scene_dim)}
            )
        except Exception as e:
            logger.debug(f"Scene vector index create skipped: {e}")
    driver.close()


def _iter_entities(session) -> List[dict]:
    result = session.run(
        """
        MATCH (e:Entity)
        RETURN e.id AS id,
               e.class AS class,
               e.canonical_label AS canonical,
               e.description AS description,
               coalesce(e.scene_types, []) AS scene_types
        """
    )
    return result.data()


def _iter_scenes(session) -> List[dict]:
    result = session.run(
        """
        MATCH (s:Scene)
        RETURN s.id AS id,
               s.scene_type AS type,
               s.description AS description,
               coalesce(s.dominant_objects, []) AS objects
        """
    )
    return result.data()


def _entity_context_text(rec: dict) -> str:
    parts: List[str] = []
    cls = (rec.get("class") or "").strip()
    can = (rec.get("canonical") or "").strip()
    desc = (rec.get("description") or "").strip()
    scene_types = rec.get("scene_types") or []

    if can:
        parts.append(f"Canonical: {can}.")
    if cls:
        parts.append(f"Class: {cls}.")
    if scene_types:
        parts.append("Scenes: " + ", ".join(scene_types[:4]) + ".")
    if desc:
        parts.append("Description: " + desc[:300])
    return " ".join(parts) or f"Entity {rec.get('id')}"


def _scene_context_text(rec: dict) -> str:
    parts: List[str] = []
    st = (rec.get("type") or "").strip()
    desc = (rec.get("description") or "").strip()
    objs = rec.get("objects") or []
    if st:
        parts.append(f"Scene type: {st}.")
    if objs:
        parts.append("Objects: " + ", ".join([str(o) for o in objs[:6]]) + ".")
    if desc:
        parts.append("Description: " + desc[:300])
    return " ".join(parts) or f"Scene {rec.get('id')}"


def backfill_embeddings(
    uri: str,
    user: str,
    password: str,
    prefer_ollama: bool = True,
    backend: str | None = None,
    model_name: str | None = None,
) -> Tuple[int, int, int, int]:
    """
    Compute and store context embeddings for Entities and Scenes.

    Returns:
        (entities_processed, entities_embedded, scenes_processed, scenes_embedded)
    """
    emb_model: EmbeddingModel = create_embedding_model(
        prefer_ollama=prefer_ollama,
        backend=backend,
        model=model_name,
    )
    
    entity_dim = emb_model.get_embedding_dimension()
    scene_dim = entity_dim  # same model for both

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        entities = _iter_entities(session)
        scenes = _iter_scenes(session)

        # Ensure indexes
        ensure_vector_indexes(uri, user, password, entity_dim, scene_dim)

        # Embed entities
        e_texts = [_entity_context_text(r) for r in entities]
        e_vecs = emb_model.encode(e_texts)
        e_count = 0
        for rec, vec in zip(entities, e_vecs):
            if _safe_len(vec) == 0:
                continue
            session.run(
                """
                MATCH (e:Entity {id: $id})
                SET e.context_embedding = $vec
                """,
                {"id": rec["id"], "vec": vec.tolist()},
            )
            e_count += 1

        # Embed scenes
        s_texts = [_scene_context_text(r) for r in scenes]
        s_vecs = emb_model.encode(s_texts)
        s_count = 0
        for rec, vec in zip(scenes, s_vecs):
            if _safe_len(vec) == 0:
                continue
            session.run(
                """
                MATCH (s:Scene {id: $id})
                SET s.context_embedding = $vec
                """,
                {"id": rec["id"], "vec": vec.tolist()},
            )
            s_count += 1

    driver.close()
    logger.info(
        "Backfilled embeddings: %d/%d Entities, %d/%d Scenes",
        e_count,
        len(entities),
        s_count,
        len(scenes),
    )
    return len(entities), e_count, len(scenes), s_count


def query_similar_entities(session, embedding: List[float], k: int = 5) -> List[dict]:
    """Query top-k similar entities using the vector index."""
    result = session.run(
        """
        CALL db.index.vector.queryNodes($index, $k, $embedding)
        YIELD node, score
        RETURN node.id AS id, node.class AS class, node.canonical_label AS canonical, score
        """,
        {"index": ENTITY_INDEX, "k": int(k), "embedding": embedding},
    )
    return result.data()


def query_similar_scenes(session, embedding: List[float], k: int = 3) -> List[dict]:
    """Query top-k similar scenes using the vector index."""
    result = session.run(
        """
        CALL db.index.vector.queryNodes($index, $k, $embedding)
        YIELD node, score
        RETURN node.id AS id, node.scene_type AS type, score
        """,
        {"index": SCENE_INDEX, "k": int(k), "embedding": embedding},
    )
    return result.data()
