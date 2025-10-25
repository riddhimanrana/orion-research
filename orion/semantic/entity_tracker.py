"""
Semantic Entity Tracker
=======================

Consolidates perception entities across time for semantic analysis.

Responsibilities:
- Track entities temporally across Phase 1 observations
- Detect entity appearances and disappearances
- Maintain entity state history
- Prepare entities for state change detection

Author: Orion Research Team
Date: October 2025
"""

import logging
from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np

from orion.perception.types import PerceptionEntity, PerceptionResult, BoundingBox
from orion.semantic.types import SemanticEntity
from orion.semantic.config import StateChangeConfig

logger = logging.getLogger(__name__)


class SemanticEntityTracker:
    """
    Tracks entities for semantic analysis.
    
    Consolidates perception entities and tracks their temporal evolution
    for state change detection.
    """
    
    def __init__(self, config: StateChangeConfig):
        """
        Initialize semantic entity tracker.
        
        Args:
            config: State change configuration
        """
        self.config = config
        self.semantic_entities: Dict[str, SemanticEntity] = {}
        
        logger.debug("SemanticEntityTracker initialized")
    
    def consolidate_entities(
        self,
        perception_result: PerceptionResult,
    ) -> List[SemanticEntity]:
        """
        Consolidate perception entities into semantic entities.
        
        Args:
            perception_result: Output from Phase 1 (Perception)
            
        Returns:
            List of semantic entities ready for state change detection
        """
        logger.info("="*80)
        logger.info("PHASE 2A: SEMANTIC ENTITY CONSOLIDATION")
        logger.info("="*80)
        
        perception_entities = perception_result.entities
        
        logger.info(f"Consolidating {len(perception_entities)} perception entities...")
        
        semantic_entities = []
        
        for perc_entity in perception_entities:
            # Create semantic entity from perception entity
            sem_entity = SemanticEntity(
                entity_id=perc_entity.entity_id,
                object_class=perc_entity.object_class.value,
                perception_entity=perc_entity,
            )
            sem_entity.observations = list(perc_entity.observations)
            if perc_entity.average_embedding is None:
                try:
                    perc_entity.compute_average_embedding()
                except Exception as exc:
                    logger.debug("Failed to compute average embedding for %s: %s", perc_entity.entity_id, exc)

            sem_entity.average_embedding = perc_entity.average_embedding
            
            # Track first/last appearance
            sem_entity.first_appearance_time = perc_entity.observations[0].timestamp if perc_entity.observations else 0.0
            sem_entity.last_appearance_time = perc_entity.observations[-1].timestamp if perc_entity.observations else 0.0
            sem_entity.first_timestamp = sem_entity.first_appearance_time
            sem_entity.last_timestamp = sem_entity.last_appearance_time

            if perc_entity.observations:
                centroids = np.array([obs.centroid for obs in perc_entity.observations], dtype=float)
                avg_cx = float(np.mean(centroids[:, 0]))
                avg_cy = float(np.mean(centroids[:, 1]))
                sem_entity.average_centroid = (avg_cx, avg_cy)

                frame_width = perc_entity.observations[0].frame_width or 1920.0
                frame_height = perc_entity.observations[0].frame_height or 1080.0
                sem_entity.frame_width = frame_width
                sem_entity.frame_height = frame_height

                x1 = min(obs.bounding_box.x1 for obs in perc_entity.observations)
                y1 = min(obs.bounding_box.y1 for obs in perc_entity.observations)
                x2 = max(obs.bounding_box.x2 for obs in perc_entity.observations)
                y2 = max(obs.bounding_box.y2 for obs in perc_entity.observations)
                sem_entity.average_bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            
            # Add initial description if available
            if perc_entity.description:
                sem_entity.descriptions.append({
                    "timestamp": sem_entity.first_appearance_time,
                    "frame": perc_entity.description_frame or 0,
                    "text": perc_entity.description,
                })
                sem_entity.description = perc_entity.description
            
            semantic_entities.append(sem_entity)
            self.semantic_entities[sem_entity.entity_id] = sem_entity
        
        # Sort by first appearance
        semantic_entities.sort(key=lambda e: e.first_appearance_time)
        
        logger.info(f"✓ Consolidated {len(semantic_entities)} semantic entities")
        logger.info(f"  Time range: {semantic_entities[0].first_appearance_time:.2f}s - {semantic_entities[-1].last_appearance_time:.2f}s")
        logger.info("="*80 + "\n")
        
        return semantic_entities
    
    def get_entity(self, entity_id: str) -> Optional[SemanticEntity]:
        """Get semantic entity by ID."""
        return self.semantic_entities.get(entity_id)
    
    def get_all_entities(self) -> List[SemanticEntity]:
        """Get all semantic entities."""
        return list(self.semantic_entities.values())
    
    def get_active_entities(self, timestamp: float, window: float = 1.0) -> List[SemanticEntity]:
        """
        Get entities active at a given timestamp.
        
        Args:
            timestamp: Target timestamp
            window: Time window (seconds) around timestamp
            
        Returns:
            List of active entities
        """
        active = []
        
        for entity in self.semantic_entities.values():
            if (entity.first_appearance_time - window <= timestamp <= 
                entity.last_appearance_time + window):
                active.append(entity)
        
        return active
    
    def get_entity_timeline(self, entity_id: str) -> List[Dict]:
        """
        Get temporal timeline for an entity.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            List of timeline events (appearances, descriptions, state changes)
        """
        entity = self.get_entity(entity_id)
        if not entity:
            return []
        
        timeline = []
        
        # Add appearance events
        if entity.perception_entity:
            for obs in entity.perception_entity.observations:
                timeline.append({
                    "type": "appearance",
                    "timestamp": obs.timestamp,
                    "frame": obs.frame_number,
                    "confidence": obs.confidence,
                    "bbox": obs.bounding_box.to_list(),
                })
        
        # Add description events
        for desc in entity.descriptions:
            timeline.append({
                "type": "description",
                "timestamp": desc["timestamp"],
                "frame": desc["frame"],
                "text": desc["text"],
            })
        
        # Add state change events
        for change in entity.state_changes:
            timeline.append({
                "type": "state_change",
                "timestamp": change.timestamp_after,
                "frame_before": change.frame_before,
                "frame_after": change.frame_after,
                "change_magnitude": change.change_magnitude,
            })
        
        # Sort by timestamp
        timeline.sort(key=lambda e: e["timestamp"])
        
        return timeline
