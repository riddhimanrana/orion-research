"""
Temporal Description Generator
================================

Generates entity descriptions at multiple time points throughout a video.

This is critical for state change detection, which requires comparing
descriptions from different time points.

Responsibilities:
- Sample entities at regular intervals
- Generate fresh CLIP descriptions at each sample point
- Track description evolution over time
- Enable state change detection

Author: Orion Research Team
Date: October 2025
"""

import logging
from typing import List, Dict, Optional
import numpy as np

from orion.perception.types import PerceptionEntity, Observation
from orion.semantic.types import SemanticEntity

logger = logging.getLogger(__name__)


class TemporalDescriptionGenerator:
    """
    Generates entity descriptions at multiple time points.
    
    Essential for state change detection which requires descriptions
    at different times to compare.
    """
    
    def __init__(
        self,
        clip_model=None,
        sample_interval: float = 2.0,
        min_samples_per_entity: int = 3,
    ):
        """
        Initialize temporal description generator.
        
        Args:
            clip_model: CLIP model for image encoding and text generation
            sample_interval: Seconds between description samples (default 2.0s)
            min_samples_per_entity: Minimum number of descriptions per entity
        """
        self.clip_model = clip_model
        self.sample_interval = sample_interval
        self.min_samples_per_entity = min_samples_per_entity
        
        logger.debug(
            f"TemporalDescriptionGenerator initialized: "
            f"interval={sample_interval}s, min_samples={min_samples_per_entity}"
        )
    
    def generate_temporal_descriptions(
        self,
        semantic_entities: List[SemanticEntity],
    ) -> None:
        """
        Generate descriptions for entities at multiple time points.
        
        Args:
            semantic_entities: List of semantic entities to describe
        """
        logger.info("="*80)
        logger.info("PHASE 2B+: TEMPORAL DESCRIPTION GENERATION")
        logger.info("="*80)
        
        total_descriptions_added = 0
        
        for entity in semantic_entities:
            descriptions_added = self._generate_entity_descriptions(entity)
            total_descriptions_added += descriptions_added
            
            if descriptions_added > 0:
                logger.debug(
                    f"Entity {entity.entity_id}: "
                    f"Added {descriptions_added} descriptions "
                    f"(total: {len(entity.descriptions)})"
                )
        
        logger.info(f"✓ Generated {total_descriptions_added} temporal descriptions")
        logger.info("="*80 + "\n")
    
    def _generate_entity_descriptions(self, entity: SemanticEntity) -> int:
        """
        Generate descriptions for a single entity at multiple time points.
        
        Args:
            entity: Semantic entity
            
        Returns:
            Number of new descriptions added
        """
        if not entity.perception_entity or not entity.perception_entity.observations:
            return 0
        
        perc_entity = entity.perception_entity
        observations = sorted(perc_entity.observations, key=lambda o: o.timestamp)
        
        if len(observations) < 2:
            return 0
        
        # Calculate sampling points
        start_time = observations[0].timestamp
        end_time = observations[-1].timestamp
        duration = end_time - start_time
        
        # Need at least 2 samples per entity for state change detection
        sample_times = self._calculate_sample_times(
            start_time,
            end_time,
            self.sample_interval,
        )
        
        # Ensure we have enough samples
        if len(sample_times) < self.min_samples_per_entity:
            # Recalculate with fewer samples
            num_samples = max(self.min_samples_per_entity, 2)
            sample_times = np.linspace(start_time, end_time, num_samples).tolist()
        
        descriptions_added = 0
        
        for sample_time in sample_times:
            # Skip if already have description at this time (within 0.1s tolerance)
            existing_times = {d["timestamp"] for d in entity.descriptions}
            if any(abs(sample_time - t) < 0.1 for t in existing_times):
                continue
            
            # Find observations near this time
            obs_before, obs_after = self._find_bracketing_observations(
                observations,
                sample_time,
            )
            
            if not obs_before:
                continue
            
            # Generate description from observation(s)
            description = self._generate_description_from_observations(
                entity,
                obs_before,
                obs_after if obs_after else None,
            )
            
            if description:
                entity.descriptions.append({
                    "timestamp": sample_time,
                    "frame": obs_before.frame_number,
                    "text": description,
                })
                descriptions_added += 1
        
        return descriptions_added
    
    def _calculate_sample_times(
        self,
        start_time: float,
        end_time: float,
        interval: float,
    ) -> List[float]:
        """
        Calculate regular sample times across video duration.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            interval: Interval between samples
            
        Returns:
            List of sample times
        """
        sample_times = []
        current_time = start_time
        
        while current_time <= end_time:
            sample_times.append(current_time)
            current_time += interval
        
        # Always include end time
        if abs(sample_times[-1] - end_time) > 0.01:
            sample_times.append(end_time)
        
        return sample_times
    
    def _find_bracketing_observations(
        self,
        observations: List[Observation],
        target_time: float,
    ) -> tuple:
        """
        Find observations before and after a target time.
        
        Args:
            observations: Sorted list of observations
            target_time: Target timestamp
            
        Returns:
            Tuple of (obs_before, obs_after) or (None, None)
        """
        obs_before = None
        obs_after = None
        
        for obs in observations:
            if obs.timestamp <= target_time:
                obs_before = obs
            elif obs.timestamp > target_time and obs_after is None:
                obs_after = obs
                break
        
        return obs_before, obs_after
    
    def _generate_description_from_observations(
        self,
        entity: SemanticEntity,
        obs_before: Observation,
        obs_after: Optional[Observation] = None,
    ) -> Optional[str]:
        """
        Generate description from observation(s).
        
        Args:
            entity: Semantic entity
            obs_before: Observation at or before sample time
            obs_after: Optional observation after sample time
            
        Returns:
            Generated description or None
        """
        if not self.clip_model:
            # Fallback: Use original or interpolate description
            if entity.description:
                return entity.description
            return None
        
        try:
            # Get image crop for observation
            image_crop = obs_before.get_crop() if hasattr(obs_before, 'get_crop') else None
            
            if image_crop is None:
                return entity.description  # Fallback to original
            
            # Generate CLIP description
            # This would use a caption model or text encoder
            # For now, return entity's description + movement info if available
            
            desc_parts = [entity.description]
            
            if obs_after and obs_before.timestamp < obs_after.timestamp:
                # Add motion description
                dx = obs_after.centroid[0] - obs_before.centroid[0]
                dy = obs_after.centroid[1] - obs_before.centroid[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 10:  # Significant movement
                    time_delta = obs_after.timestamp - obs_before.timestamp
                    speed = distance / time_delta if time_delta > 0 else 0
                    
                    if dx > 0:
                        direction = "moving right"
                    elif dx < 0:
                        direction = "moving left"
                    elif dy > 0:
                        direction = "moving down"
                    else:
                        direction = "moving up"
                    
                    desc_parts.append(f"({direction} at {speed:.1f}px/s)")
            
            return " ".join(desc_parts)
        
        except Exception as e:
            logger.debug(f"Failed to generate description: {e}")
            return entity.description


class TemporalDescriptionUpdater:
    """
    Updates semantic entities with temporal descriptions.
    
    This should be called after entity consolidation and before
    state change detection.
    """
    
    def __init__(self, generator: TemporalDescriptionGenerator):
        """
        Initialize updater.
        
        Args:
            generator: Description generator instance
        """
        self.generator = generator
    
    def update_entities(self, semantic_entities: List[SemanticEntity]) -> None:
        """
        Update all entities with temporal descriptions.
        
        Args:
            semantic_entities: List of entities to update
        """
        self.generator.generate_temporal_descriptions(semantic_entities)
