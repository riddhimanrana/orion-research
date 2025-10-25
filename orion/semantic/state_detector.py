"""
State Change Detector
=====================

Detects semantic state changes in entity descriptions.

Responsibilities:
- Compare entity descriptions across time
- Compute embedding similarity scores
- Identify significant state changes
- Track state transitions with metadata

Author: Orion Research Team
Date: October 2025
"""

import logging
from typing import List, Optional

import numpy as np

from orion.semantic.types import StateChange, SemanticEntity, ChangeType
from orion.semantic.config import StateChangeConfig

logger = logging.getLogger(__name__)


class StateChangeDetector:
    """
    Detects state changes in entities over time.
    
    Compares entity descriptions and visual embeddings to identify
    significant semantic state transitions.
    """
    
    def __init__(
        self,
        embedding_model,
        config: StateChangeConfig,
    ):
        """
        Initialize state change detector.
        
        Args:
            embedding_model: Model for text embeddings (CLIP or SentenceTransformer)
            config: State change configuration
        """
        self.embedding_model = embedding_model
        self.config = config
        self.state_changes: List[StateChange] = []
        
        logger.debug(
            f"StateChangeDetector initialized: threshold={config.embedding_similarity_threshold}"
        )
    
    def detect_changes(
        self,
        semantic_entities: List[SemanticEntity],
    ) -> List[StateChange]:
        """
        Detect state changes across all entities.
        
        Args:
            semantic_entities: List of semantic entities
            
        Returns:
            List of detected state changes
        """
        logger.info("="*80)
        logger.info("PHASE 2B: STATE CHANGE DETECTION")
        logger.info("="*80)
        
        logger.info(f"Analyzing {len(semantic_entities)} entities for state changes...")
        logger.info(f"  Similarity threshold: {self.config.embedding_similarity_threshold}")
        
        all_changes = []
        
        for entity in semantic_entities:
            changes = self._detect_entity_changes(entity)
            all_changes.extend(changes)
            
            if changes:
                entity.state_changes = changes
        
        logger.info(f"\n✓ Detected {len(all_changes)} state changes")
        
        if all_changes:
            logger.info("  Sample state changes:")
            for i, change in enumerate(all_changes[:5]):
                logger.info(
                    f"    {i+1}. Entity {change.entity_id}: "
                    f"change_magnitude={change.change_magnitude:.2f} "
                    f"at t={change.timestamp_after:.2f}s"
                )
        
        logger.info("="*80 + "\n")
        
        self.state_changes = all_changes
        return all_changes
    
    def _detect_entity_changes(self, entity: SemanticEntity) -> List[StateChange]:
        """
        Detect state changes for a single entity.
        
        Args:
            entity: Semantic entity
            
        Returns:
            List of state changes for this entity
        """
        changes = []
        
        # Need at least 2 descriptions to detect changes
        if len(entity.descriptions) < 2:
            return changes
        
        # Compare consecutive descriptions
        for i in range(len(entity.descriptions) - 1):
            desc_before = entity.descriptions[i]
            desc_after = entity.descriptions[i + 1]
            
            # Check time gap
            time_gap = desc_after["timestamp"] - desc_before["timestamp"]
            if time_gap < self.config.min_time_between_changes:
                continue
            
            # Compute similarity
            similarity = self._compute_description_similarity(
                desc_before["text"],
                desc_after["text"],
            )
            
            # Detect significant change
            if similarity < self.config.embedding_similarity_threshold:
                # Create state change
                change = StateChange(
                    entity_id=entity.entity_id,
                    timestamp_before=desc_before["timestamp"],
                    timestamp_after=desc_after["timestamp"],
                    frame_before=desc_before["frame"],
                    frame_after=desc_after["frame"],
                    description_before=desc_before["text"],
                    description_after=desc_after["text"],
                    similarity_score=similarity,
                )
                
                # Compute displacement if spatial info available
                if entity.perception_entity:
                    obs_before = self._find_closest_observation(
                        entity.perception_entity,
                        desc_before["frame"],
                    )
                    obs_after = self._find_closest_observation(
                        entity.perception_entity,
                        desc_after["frame"],
                    )
                    
                    if obs_before and obs_after:
                        change.centroid_before = obs_before.centroid
                        change.centroid_after = obs_after.centroid
                        change.bounding_box_before = obs_before.bounding_box.to_list()
                        change.bounding_box_after = obs_after.bounding_box.to_list()
                        
                        # Compute displacement
                        dx = obs_after.centroid[0] - obs_before.centroid[0]
                        dy = obs_after.centroid[1] - obs_before.centroid[1]
                        change.displacement = np.sqrt(dx**2 + dy**2)
                        
                        # Compute velocity
                        if time_gap > 0:
                            change.velocity = change.displacement / time_gap
                
                changes.append(change)
        
        return changes
    
    def _compute_description_similarity(
        self,
        desc1: str,
        desc2: str,
    ) -> float:
        """
        Compute similarity between two descriptions.
        
        Args:
            desc1: First description
            desc2: Second description
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Embed descriptions
        emb1 = self._embed_text(desc1)
        emb2 = self._embed_text(desc2)
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(similarity)
    
    def _embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self.config.embedding_model == "clip":
            # Use CLIP text encoder
            embedding = self.embedding_model.encode_text(text, normalize=True)
        else:
            # Use SentenceTransformer
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        
        return embedding
    
    def _find_closest_observation(self, perception_entity, target_frame: int):
        """
        Find observation closest to target frame.
        
        Args:
            perception_entity: PerceptionEntity
            target_frame: Target frame number
            
        Returns:
            Closest observation or None
        """
        if not perception_entity.observations:
            return None
        
        closest_obs = None
        min_distance = float('inf')
        
        for obs in perception_entity.observations:
            distance = abs(obs.frame_number - target_frame)
            if distance < min_distance:
                min_distance = distance
                closest_obs = obs
        
        return closest_obs
    
    def get_changes_for_entity(self, entity_id: str) -> List[StateChange]:
        """Get all state changes for a specific entity."""
        return [c for c in self.state_changes if c.entity_id == entity_id]
    
    def get_changes_in_window(
        self,
        start_time: float,
        end_time: float,
    ) -> List[StateChange]:
        """Get state changes within a time window."""
        return [
            c for c in self.state_changes
            if start_time <= c.timestamp_after <= end_time
        ]
