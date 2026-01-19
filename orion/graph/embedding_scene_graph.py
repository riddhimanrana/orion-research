"""
DINOv3-aware scene graph generation with embedding-based relationship verification.

This module extends the base scene graph builder to:
1. Load DINOv3 embeddings from appearance_history
2. Use embedding similarity to verify object identity across frames
3. Weight geometric relationships with semantic similarity
4. Improve relationship confidence using embedding cosine similarity
"""

import json
import logging
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRelationConfig:
    """Configuration for embedding-aware relationship inference."""
    
    # Embedding similarity weights
    use_embedding_similarity: bool = True
    """If True, use DINOv3 embeddings to verify relationships."""
    
    embedding_weight: float = 0.3
    """Weight for embedding similarity (0-1). Geometry gets 1.0 - embedding_weight."""
    
    # Similarity thresholds
    same_object_similarity: float = 0.75
    """Cosine similarity threshold to consider two detections as same object."""
    
    near_embedding_threshold: float = 0.6
    """Minimum embedding similarity for 'near' relationships."""
    
    on_embedding_threshold: float = 0.5
    """Minimum embedding similarity for 'on' relationships."""
    
    held_by_embedding_threshold: float = 0.55
    """Minimum embedding similarity for 'held_by' relationships."""


def load_embeddings_from_memory(memory_input) -> Dict[str, np.ndarray]:
    """
    Load DINOv3 embeddings from memory dict or memory.json file.
    
    NOTE: Current memory.json format doesn't contain embedding vectors.
    This function is prepared for when embeddings are stored.
    
    Args:
        memory_input: Either a file path (Path or str) or memory dict
    
    Returns:
        Dict mapping memory_id → embedding vector (normalized to unit norm)
    """
    embeddings = {}
    
    # Load memory dict
    try:
        if isinstance(memory_input, dict):
            memory = memory_input
        else:
            with open(memory_input) as f:
                memory = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load memory: {e}")
        return embeddings
    
    # Try to load from various possible locations in memory
    for obj in memory.get("objects", []):
        mem_id = obj.get("memory_id")
        if not mem_id:
            continue
        
        # Try location 1: prototype_embedding_vector
        if isinstance(obj.get("prototype_embedding_vector"), list):
            try:
                emb_arr = np.array(obj["prototype_embedding_vector"], dtype=np.float32)
                emb_arr = emb_arr / (np.linalg.norm(emb_arr) + 1e-8)
                embeddings[mem_id] = emb_arr
                continue
            except Exception:
                pass
        
        # Try location 2: embedding field in object
        if isinstance(obj.get("embedding"), list):
            try:
                emb_arr = np.array(obj["embedding"], dtype=np.float32)
                emb_arr = emb_arr / (np.linalg.norm(emb_arr) + 1e-8)
                embeddings[mem_id] = emb_arr
                continue
            except Exception:
                pass
        
        # Try location 3: appearance_history with embedding lists
        for obs in obj.get("appearance_history", []):
            if not isinstance(obs, dict):
                continue
            
            # Try observations as list
            obs_list = obs.get("observations")
            if isinstance(obs_list, list):
                for det in obs_list:
                    if isinstance(det.get("embedding"), list):
                        try:
                            emb_arr = np.array(det["embedding"], dtype=np.float32)
                            emb_arr = emb_arr / (np.linalg.norm(emb_arr) + 1e-8)
                            embeddings[mem_id] = emb_arr
                            break
                        except Exception:
                            pass
            
            if mem_id in embeddings:
                break
    
    if embeddings:
        logger.info(f"Loaded {len(embeddings)} DINOv3 embeddings for scene graph verification")
    else:
        logger.debug("No DINOv3 embeddings found in memory. Falling back to geometry-only scene graphs.")
    
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if a is None or b is None:
        return 0.0
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def get_embedding_for_object(
    mem_id: str,
    frame_id: int,
    embeddings: Dict[Tuple[str, int], np.ndarray],
    memory: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Get the most recent embedding for an object in a frame."""
    
    # First try exact frame match
    if (mem_id, frame_id) in embeddings:
        return embeddings[(mem_id, frame_id)]
    
    # Fall back to most recent embedding for this object
    best_embedding = None
    best_frame = -1
    
    for (obj_id, f), emb in embeddings.items():
        if obj_id == mem_id and f <= frame_id and f > best_frame:
            best_embedding = emb
            best_frame = f
    
    return best_embedding


def build_embedding_aware_scene_graph(
    tracks: List[Dict[str, Any]],
    memory: Dict[str, Any],
    memory_path: Optional[Path] = None,
    config: Optional[EmbeddingRelationConfig] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Build scene graph with DINOv3 embedding-based relationship verification.
    
    Args:
        tracks: List of track observations (same format as before)
        memory: Memory object data
        memory_path: Path to memory.json (for loading embedding history)
        config: Embedding-aware relationship config
        **kwargs: Forwarded to base scene graph builder
    
    Returns:
        List of per-frame scene graphs with embedding-verified edges
    """
    
    if config is None:
        config = EmbeddingRelationConfig()
    
    # Load embeddings if available
    embeddings = {}
    if memory_path and config.use_embedding_similarity:
        embeddings = load_embeddings_from_memory(Path(memory_path))
        logger.info(f"Loaded {len(embeddings)} embeddings from memory")
    
    # Build base scene graphs using geometry
    from orion.graph.scene_graph import build_research_scene_graph
    
    graphs = build_research_scene_graph(tracks=tracks, memory=memory, **kwargs)
    
    if not embeddings or not config.use_embedding_similarity:
        return graphs  # Return geometric graphs as-is
    
    # Post-process edges with embedding verification
    for graph in graphs:
        frame_id = graph.get("frame")
        edges = graph.get("edges", [])
        
        verified_edges = []
        for edge in edges:
            subject_id = edge.get("subject")
            object_id = edge.get("object")
            relation = edge.get("relation")
            
            # Get embeddings for this frame
            subj_emb = get_embedding_for_object(subject_id, frame_id, embeddings, memory)
            obj_emb = get_embedding_for_object(object_id, frame_id, embeddings, memory)
            
            if subj_emb is None or obj_emb is None:
                # No embedding available, keep geometric edge
                verified_edges.append(edge)
                continue
            
            # Compute embedding similarity
            similarity = cosine_similarity(subj_emb, obj_emb)
            
            # Apply relation-specific threshold
            threshold = 0.0
            if relation == "near":
                threshold = config.near_embedding_threshold
            elif relation == "on":
                threshold = config.on_embedding_threshold
            elif relation == "held_by":
                threshold = config.held_by_embedding_threshold
            
            # Weighted confidence: geometry + embedding
            if similarity >= threshold:
                # High confidence: both geometry and semantics agree
                edge["embedding_similarity"] = float(similarity)
                edge["confidence"] = min(
                    1.0,
                    (1.0 - config.embedding_weight) * 0.8 +  # Assume geometry had 0.8 confidence
                    config.embedding_weight * similarity
                )
                verified_edges.append(edge)
            # else: drop edge if embedding similarity is too low
        
        graph["edges"] = verified_edges
    
    return graphs


if __name__ == "__main__":
    # Test: load embeddings from a result directory
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python embedding_scene_graph.py <results_dir>")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    memory_path = results_dir / "memory.json"
    
    if not memory_path.exists():
        print(f"memory.json not found in {results_dir}")
        sys.exit(1)
    
    embeddings = load_embeddings_from_memory(memory_path)
    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Embedding dimension: {next(iter(embeddings.values())).shape if embeddings else 'N/A'}")
