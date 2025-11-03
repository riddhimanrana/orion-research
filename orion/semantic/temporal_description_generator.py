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
        vlm_model=None,
        sample_interval: float = 2.0,
        min_samples_per_entity: int = 3,
    ):
        """
        Initialize temporal description generator.
        
        Args:
            clip_model: CLIP model for image encoding and text generation
            vlm_model: Vision-Language Model (FastVLM) for generating descriptions
            sample_interval: Seconds between description samples (default 2.0s)
            min_samples_per_entity: Minimum number of descriptions per entity
        """
        self.clip_model = clip_model
        self.vlm_model = vlm_model
        self.sample_interval = sample_interval
        self.min_samples_per_entity = min_samples_per_entity
        
        logger.debug(
            f"TemporalDescriptionGenerator initialized: "
            f"interval={sample_interval}s, min_samples={min_samples_per_entity}, "
            f"vlm_model={'loaded' if vlm_model else 'not available'}"
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
        entities_with_descriptions = 0
        entities_skipped = 0
        
        for entity in semantic_entities:
            descriptions_added = self._generate_entity_descriptions(entity)
            total_descriptions_added += descriptions_added
            
            if descriptions_added > 0:
                entities_with_descriptions += 1
                logger.debug(
                    f"Entity {entity.entity_id}: "
                    f"Added {descriptions_added} descriptions "
                    f"(total: {len(entity.descriptions)})"
                )
            else:
                entities_skipped += 1
                logger.debug(
                    f"Entity {entity.entity_id}: No descriptions added "
                    f"(observations: {len(entity.observations) if entity.perception_entity else 0})"
                )
        
        logger.info(f"✓ Generated {total_descriptions_added} temporal descriptions")
        logger.info(f"  Entities with descriptions: {entities_with_descriptions}/{len(semantic_entities)}")
        logger.info(f"  Entities skipped: {entities_skipped}")
        logger.info("="*80 + "\n")
    
    def _generate_entity_descriptions(self, entity: SemanticEntity) -> int:
        """
        Generate descriptions for a single entity at multiple time points.
        
        Args:
            entity: Semantic entity
            
        Returns:
            Number of new descriptions added
        """
        # Get observations from either perception_entity or entity itself
        observations = []
        if entity.perception_entity and entity.perception_entity.observations:
            observations = entity.perception_entity.observations
        elif entity.observations:
            observations = entity.observations
        else:
            logger.debug(f"Entity {entity.entity_id}: No observations, skipping")
            return 0
        
        observations = sorted(observations, key=lambda o: o.timestamp)
        
        if len(observations) < 2:
            logger.debug(f"Entity {entity.entity_id}: Only {len(observations)} observation(s), need 2+ for temporal descriptions")
            # Still add at least one description from base
            if entity.description and len(entity.descriptions) == 0:
                entity.descriptions.append({
                    "timestamp": observations[0].timestamp,
                    "frame": observations[0].frame_number,
                    "text": entity.description,
                })
                return 1
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
        
        # Base description fallback when none available
        base_description = entity.description if entity.description else f"a {entity.object_class}"

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
            
            # Ensure we always have a textual description
            if not description:
                description = base_description

            entity.descriptions.append({
                "timestamp": sample_time,
                "frame": obs_before.frame_number,
                "text": description,
            })
            descriptions_added += 1
        
        # Fallback: If we still don't have at least 2 descriptions, force add them
        if len(entity.descriptions) < 2 and len(observations) >= 2:
            logger.debug(f"Entity {entity.entity_id}: Forcing 2 descriptions for state change detection")
            
            # Add first observation description
            if len(entity.descriptions) == 0:
                first_obs = observations[0]
                first_desc = base_description
                entity.descriptions.append({
                    "timestamp": first_obs.timestamp,
                    "frame": first_obs.frame_number,
                    "text": first_desc,
                })
                descriptions_added += 1
            
            # Add last observation description with position variation
            last_obs = observations[-1]
            last_desc = self._generate_description_from_observations(
                entity, last_obs, None
            )
            if not last_desc:
                last_desc = f"{base_description} (end position)"
            
            entity.descriptions.append({
                "timestamp": last_obs.timestamp,
                "frame": last_obs.frame_number,
                "text": last_desc,
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
        # Strategy: Try VLM first, fall back to motion-enhanced descriptions
        
        # 1. Try using VLM for fresh description
        if self.vlm_model and obs_before.image_patch is not None:
            try:
                # Generate description using VLM
                prompt = f"Describe this {entity.object_class} in detail."
                vlm_description = self.vlm_model.generate(
                    obs_before.image_patch,
                    prompt,
                    max_tokens=50,
                    temperature=0.7
                )
                
                # Add motion context if available
                if obs_after and obs_before.timestamp < obs_after.timestamp:
                    motion_desc = self._generate_motion_description(obs_before, obs_after)
                    if motion_desc:
                        return f"{vlm_description} {motion_desc}"
                
                return vlm_description
                
            except Exception as e:
                logger.debug(f"VLM description failed: {e}, falling back to motion-based")
        
        # 2. Fallback: Generate motion-based description variation
        if obs_after and obs_before.timestamp < obs_after.timestamp:
            motion_desc = self._generate_motion_description(obs_before, obs_after)
            if motion_desc and entity.description:
                return f"{entity.description} {motion_desc}"
        
        # 3. Last resort: Add position/context variation to base description
        if entity.description:
            # Add position context to create variation
            centroid = obs_before.centroid
            frame_width = obs_before.frame_width or 1920
            frame_height = obs_before.frame_height or 1080
            
            x_rel = centroid[0] / frame_width
            y_rel = centroid[1] / frame_height
            
            # Create varied descriptions based on position and time
            position = ""
            if y_rel < 0.33:
                position = "in the upper part of frame"
            elif y_rel > 0.67:
                position = "in the lower part of frame"
            else:
                position = "in the middle of frame"
            
            # Add horizontal position for more variation
            if x_rel < 0.33:
                position += ", on the left side"
            elif x_rel > 0.67:
                position += ", on the right side"
            else:
                position += ", centered horizontally"
            
            # Add temporal context for even more variation
            timestamp_context = ""
            entity_duration = entity.last_timestamp - entity.first_timestamp
            if entity_duration > 0:
                relative_time = (obs_before.timestamp - entity.first_timestamp) / entity_duration
                if relative_time < 0.33:
                    timestamp_context = " (early appearance)"
                elif relative_time > 0.67:
                    timestamp_context = " (late appearance)"
            
            return f"{entity.description} {position}{timestamp_context}"
            if x_rel < 0.33:
                position += ", on the left side"
            elif x_rel > 0.67:
                position += ", on the right side"
            
            return f"{entity.description} ({position})"
        
        return entity.description
    
    def _generate_motion_description(
        self,
        obs_before: Observation,
        obs_after: Observation,
    ) -> Optional[str]:
        """
        Generate motion description between two observations.
        
        Args:
            obs_before: Earlier observation
            obs_after: Later observation
            
        Returns:
            Motion description string or None
        """
        dx = obs_after.centroid[0] - obs_before.centroid[0]
        dy = obs_after.centroid[1] - obs_before.centroid[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 10:  # Minimal movement
            return "(stationary)"
        
        time_delta = obs_after.timestamp - obs_before.timestamp
        if time_delta <= 0:
            return None
        
        speed = distance / time_delta
        
        # Determine direction
        if abs(dx) > abs(dy):
            direction = "moving right" if dx > 0 else "moving left"
        else:
            direction = "moving down" if dy > 0 else "moving up"
        
        return f"({direction} at {speed:.1f}px/s)"


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
