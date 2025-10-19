"""
Smart Perception Engine (Tracking-Based)
=========================================

Uses the "Track First, Describe Once" approach:
1. Detect all objects across video (436 detections)
2. Cluster into unique entities using HDBSCAN (~10-50 entities)
3. Describe each entity ONCE from best frame
4. Build perception log with all observations

This is 10-100x more efficient than describing every detection individually.

Author: Orion Research Team
Date: October 2025
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

from orion.config import OrionConfig

from .tracking_engine import run_tracking_engine, Entity, Observation

logger = logging.getLogger(__name__)


def run_smart_perception(
    video_path: str,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    config: Optional[OrionConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Run smart tracking-based perception.
    
    This replaces the old perception_engine with a much more efficient approach:
    - Old: Describe all 436 detections individually
    - New: Cluster into ~10-50 entities, describe each once
    
    Args:
        video_path: Path to video file
        progress_callback: Optional callback for progress updates
        config: Optional configuration
        
    Returns:
        List of perception objects (compatible with semantic_uplift)
    """
    logger.info("="*80)
    logger.info("SMART PERCEPTION ENGINE")
    logger.info("Using tracking-based 'describe once' approach")
    logger.info("="*80)
    
    # Notify start
    if progress_callback:
        progress_callback("smart_perception.start", {
            "mode": "tracking-based",
            "video": video_path
        })
    
    # Phase 1: Detection & Embedding
    if progress_callback:
        progress_callback("smart_perception.phase1.start", {
            "phase": "Detection & CLIP Embeddings"
        })
    
    entities, observations = run_tracking_engine(video_path, config)
    
    if progress_callback:
        progress_callback("smart_perception.phase1.complete", {
            "observations": len(observations),
            "message": f"{len(observations)} observations with CLIP embeddings"
        })
    
    # Phase 2: Clustering
    if progress_callback:
        progress_callback("smart_perception.phase2.complete", {
            "entities": len(entities),
            "observations": len(observations),
            "efficiency": f"{len(observations) / max(len(entities), 1):.1f}x",
            "message": f"Clustered into {len(entities)} unique entities"
        })
    
    logger.info(f"\n✓ Tracking complete:")
    logger.info(f"  Total detections: {len(observations)}")
    logger.info(f"  Unique entities: {len(entities)}")
    logger.info(f"  Efficiency: {len(observations) / max(len(entities), 1):.1f}x")
    logger.info(f"  (described {len(entities)} entities instead of {len(observations)} objects)")
    
    # Build entity lookup
    entity_map = {e.id: e for e in entities}
    
    # Convert to perception log format
    perception_log = []
    
    for obs in observations:
        # Find the entity this observation belongs to
        entity = None
        for e in entities:
            if any(o.frame_number == obs.frame_number and 
                   _bboxes_match(o.bbox, obs.bbox) 
                   for o in e.observations):
                entity = e
                break
        
        if not entity:
            logger.warning(f"Observation at frame {obs.frame_number} has no entity - skipping")
            continue
        
        # Create perception object
        perception_obj = {
            # Identity
            'entity_id': entity.id,
            'temp_id': entity.id,  # Use entity ID as temp_id
            
            # Classification
            'object_class': entity.class_name,
            'detection_confidence': obs.confidence,
            
            # Description (from entity, not individual observation)
            'rich_description': entity.description or f"a {entity.class_name}",
            
            # Temporal
            'timestamp': obs.timestamp,
            'frame_number': obs.frame_number,
            
            # Spatial
            'bounding_box': obs.bbox,
            'centroid': ((obs.bbox[0] + obs.bbox[2]) / 2, (obs.bbox[1] + obs.bbox[3]) / 2),
            
            # Visual
            'visual_embedding': obs.embedding.tolist() if hasattr(obs.embedding, 'tolist') else obs.embedding,
            'crop_size': (obs.bbox[2] - obs.bbox[0], obs.bbox[3] - obs.bbox[1]),
            
            # Tracking info
            'appearance_count': entity.appearance_count,
            'first_seen': entity.first_seen,
            'last_seen': entity.last_seen,
            'duration': entity.duration,
            
            # State changes
            'state_changes': len(entity.state_changes),
            'has_state_change': len(entity.state_changes) > 0,
        }
        
        perception_log.append(perception_obj)
    
    # Phase 3: Description Generation
    if progress_callback:
        progress_callback("smart_perception.phase3.complete", {
            "entities": len(entities),
            "message": f"{len(entities)} entity descriptions (reused for all appearances)"
        })
    
    # Complete
    if progress_callback:
        progress_callback("smart_perception.complete", {
            "total": len(perception_log),
            "entities": len(entities),
            "observations": len(observations),
            "efficiency": f"{len(observations) / max(len(entities), 1):.1f}x",
            "message": f"Smart perception complete: {len(entities)} entities, {len(observations)} observations"
        })
    
    logger.info(f"\n✓ Generated perception log with {len(perception_log)} observations")
    logger.info(f"✓ All {len(entities)} unique entities have descriptions")
    
    return perception_log


def _bboxes_match(bbox1: List[int], bbox2: List[int], threshold: float = 5.0) -> bool:
    """Check if two bounding boxes match (within threshold pixels)"""
    if len(bbox1) != 4 or len(bbox2) != 4:
        return False
    
    for i in range(4):
        if abs(bbox1[i] - bbox2[i]) > threshold:
            return False
    
    return True
