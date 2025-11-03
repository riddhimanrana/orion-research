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
from typing import Dict, List, Optional

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
        
        # Cache for text embeddings to avoid recomputation
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        logger.debug(
            f"StateChangeDetector initialized: threshold={config.embedding_similarity_threshold}, "
            f"embedding_model={'CLIP' if embedding_model else 'fallback'}"
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
        
        all_changes: List[StateChange] = []
        
        for entity in semantic_entities:
            changes = self._detect_entity_changes(entity)
            all_changes.extend(changes)
            
            if changes:
                entity.state_changes = changes
        
        logger.info(f"\n✓ Detected {len(all_changes)} state changes")
        
        if not all_changes:
            fallback_changes = self._generate_fallback_changes(semantic_entities)
            if fallback_changes:
                logger.warning(
                    "No high-confidence semantic state changes detected; "
                    "synthesizing fallback appearance/disappearance changes"
                )
                all_changes.extend(fallback_changes)
                logger.info(f"  ➜ Added {len(fallback_changes)} fallback changes")

        # Persist fallback changes on entities for downstream use
        if all_changes:
            changes_by_entity: Dict[str, List[StateChange]] = {}
            for change in all_changes:
                changes_by_entity.setdefault(change.entity_id, []).append(change)

            for entity in semantic_entities:
                if entity.entity_id in changes_by_entity:
                    entity.state_changes = changes_by_entity[entity.entity_id]

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
            
            # Debug logging for similarity scores
            logger.debug(
                f"Entity {entity.entity_id}: similarity={similarity:.3f} "
                f"(threshold={self.config.embedding_similarity_threshold})"
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
                
                logger.debug(
                    f"  → State change detected! Change magnitude: {change.change_magnitude:.3f}"
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
                
                # Classify change type heuristically based on displacement
                if change.displacement > self.config.motion_displacement_threshold:
                    change.change_type = ChangeType.POSITION
                else:
                    change.change_type = ChangeType.APPEARANCE

                changes.append(change)
        
        return changes

    def _generate_fallback_changes(
        self,
        semantic_entities: List[SemanticEntity],
    ) -> List[StateChange]:
        """Generate fallback state changes using appearance/disappearance heuristics."""
        fallback_changes: List[StateChange] = []

        for entity in semantic_entities:
            observations = []
            if entity.perception_entity and entity.perception_entity.observations:
                observations = list(entity.perception_entity.observations)
            elif entity.observations:
                observations = list(entity.observations)

            if not observations:
                continue

            observations.sort(key=lambda obs: obs.timestamp)
            first_obs = observations[0]
            last_obs = observations[-1]

            base_description = entity.description or f"{entity.object_class}"

            appearance_change = StateChange(
                entity_id=entity.entity_id,
                timestamp_before=max(0.0, first_obs.timestamp - 0.1),
                timestamp_after=first_obs.timestamp + 0.1,
                frame_before=first_obs.frame_number,
                frame_after=first_obs.frame_number,
                description_before=f"{base_description} not visible",
                description_after=f"{base_description} becomes visible",
                similarity_score=0.0,
            )
            appearance_change.change_type = ChangeType.APPEARANCE
            appearance_change.is_fallback = True
            appearance_change.centroid_after = first_obs.centroid
            appearance_change.centroid_before = first_obs.centroid
            appearance_change.bounding_box_after = first_obs.bounding_box.to_list()
            appearance_change.bounding_box_before = first_obs.bounding_box.to_list()

            fallback_changes.append(appearance_change)

            # Only create disappearance change if entity exists beyond 0.2s span
            if last_obs.timestamp - first_obs.timestamp >= 0.2:
                disappearance_change = StateChange(
                    entity_id=entity.entity_id,
                    timestamp_before=max(first_obs.timestamp, last_obs.timestamp - 0.1),
                    timestamp_after=last_obs.timestamp + 0.1,
                    frame_before=last_obs.frame_number,
                    frame_after=last_obs.frame_number,
                    description_before=f"{base_description} visible",
                    description_after=f"{base_description} no longer visible",
                    similarity_score=0.0,
                )
                disappearance_change.change_type = ChangeType.DISAPPEARANCE
                disappearance_change.is_fallback = True
                disappearance_change.centroid_before = last_obs.centroid
                disappearance_change.centroid_after = last_obs.centroid
                disappearance_change.bounding_box_before = last_obs.bounding_box.to_list()
                disappearance_change.bounding_box_after = last_obs.bounding_box.to_list()

                fallback_changes.append(disappearance_change)

        fallback_changes.sort(key=lambda change: change.timestamp_after)
        return fallback_changes
    
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
            Cosine similarity score (0-1), clamped from [-1, 1] range
        """
        # Embed descriptions
        emb1 = self._embed_text(desc1)
        emb2 = self._embed_text(desc2)
        
        # Compute cosine similarity (ranges from -1 to 1)
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Clamp to [0, 1] range for StateChange validation
        # Map [-1, 1] to [0, 1]: similarity_normalized = (similarity + 1) / 2
        similarity_normalized = (float(similarity) + 1.0) / 2.0
        
        # Ensure it's strictly in [0, 1] even with floating point errors
        emb1 = self._embed_text(desc1)
        emb2 = self._embed_text(desc2)

        # Compute cosine similarity in [-1, 1]
        denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        if denom == 0:
            logger.debug("Zero-norm embedding encountered when computing similarity")
            raw_sim = 0.0
        else:
            raw_sim = float(np.dot(emb1, emb2) / denom)

        # Map from [-1, 1] to [0, 1] for downstream code expectations
        sim01 = max(0.0, min(1.0, (raw_sim + 1.0) / 2.0))

        logger.debug(
            f"Description similarity: raw={raw_sim:.4f} mapped={sim01:.4f} '\n'"
            f"  desc1='{desc1[:60]}...' desc2='{desc2[:60]}...'"
        )

        return float(sim01)

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check cache first
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        # Use CLIP text encoder if available
        if self.embedding_model is not None:
            try:
                # CLIP has encode_text method
                if hasattr(self.embedding_model, 'encode_text'):
                    embedding = self.embedding_model.encode_text(text, normalize=True)
                # SentenceTransformer has encode method
                elif hasattr(self.embedding_model, 'encode'):
                    embedding = self.embedding_model.encode(text, normalize_embeddings=True)
                else:
                    # Fallback to simple hash-based embedding
                    logger.debug(f"Embedding model has no encode method, using fallback")
                    embedding = self._fallback_embedding(text)

                # Cache the result
                self._embedding_cache[text] = embedding
                return embedding

            except Exception as e:
                logger.debug(f"Failed to embed text with model: {e}, using fallback")

        # Fallback: Use deterministic hash-based embedding
        embedding = self._fallback_embedding(text)
        self._embedding_cache[text] = embedding
        return embedding
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """
        Generate fallback embedding from text hash.
        
        This is a deterministic fallback when no embedding model is available.
        
        Args:
            text: Text to embed
            
        Returns:
            Deterministic embedding vector
        """
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Create deterministic embedding from hash
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(512)
        embedding /= np.linalg.norm(embedding)
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
