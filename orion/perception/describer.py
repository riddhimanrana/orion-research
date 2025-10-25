"""
Entity Describer
================

Generates natural language descriptions for entities using FastVLM.
Applies spatial analysis and class correction for quality improvement.

Responsibilities:
- Select best frame for each entity
- Generate VLM description from best observation
- Enrich with spatial context
- Validate and correct classifications
- Optimize by describing each entity only once

Author: Orion Research Team
Date: October 2025
"""

import logging
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from orion.perception.types import PerceptionEntity, Observation
from orion.perception.config import DescriptionConfig
from orion.perception.spatial_analyzer import calculate_spatial_zone
from orion.perception.corrector import ClassCorrector

logger = logging.getLogger(__name__)


class EntityDescriber:
    """
    Describes entities using FastVLM.
    
    Generates rich natural language descriptions by selecting
    the best observation for each entity.
    """
    
    def __init__(
        self,
        vlm_model,
        config: DescriptionConfig,
        enable_class_correction: bool = True,
        enable_spatial_analysis: bool = True,
    ):
        """
        Initialize describer.
        
        Args:
            vlm_model: FastVLM model instance from ModelManager
            config: Description configuration
            enable_class_correction: Enable YOLO class correction
            enable_spatial_analysis: Enable spatial zone analysis
        """
        self.vlm = vlm_model
        self.config = config
        self.enable_class_correction = enable_class_correction
        self.enable_spatial_analysis = enable_spatial_analysis
        
        # Initialize quality improvement modules
        self.class_corrector = ClassCorrector() if enable_class_correction else None
        
        logger.debug(
            f"EntityDescriber initialized: max_tokens={config.max_tokens}, "
            f"temperature={config.temperature}, describe_once={config.describe_once}, "
            f"class_correction={enable_class_correction}, spatial_analysis={enable_spatial_analysis}"
        )
    
    def describe_entities(
        self,
        entities: List[PerceptionEntity],
    ) -> List[PerceptionEntity]:
        """
        Add descriptions to entities.
        
        Args:
            entities: List of entities from tracker
            
        Returns:
            Same entities with descriptions added
        """
        logger.info("="*80)
        logger.info("PHASE 1D: ENTITY DESCRIPTION")
        logger.info("="*80)
        
        if not entities:
            logger.warning("No entities to describe")
            return entities
        
        logger.info(f"Describing {len(entities)} entities...")
        logger.info(f"  Mode: {'Describe once (from best frame)' if self.config.describe_once else 'Describe all observations'}")
        
        described_count = 0
        
        for i, entity in enumerate(entities):
            # Select best observation
            best_obs = entity.get_best_observation()
            
            if best_obs is None:
                logger.warning(f"  Entity {entity.entity_id}: No valid observation for description")
                continue
            
            # Generate description
            description = self.describe_observation(best_obs)
            
            # Attach to entity
            entity.description = description
            entity.description_frame = best_obs.frame_number
            
            described_count += 1
            
            if (i + 1) % 10 == 0 or i == len(entities) - 1:
                logger.info(f"  Described {i + 1}/{len(entities)} entities")
        
        logger.info(f"\n✓ Generated {described_count} descriptions")
        
        # Apply quality improvements
        if self.enable_spatial_analysis or self.enable_class_correction:
            logger.info("\nApplying quality improvements...")
            entities = self._apply_quality_improvements(entities)
        
        # Report correction statistics
        if self.class_corrector:
            stats = self.class_corrector.get_statistics()
            if stats['corrections_attempted'] > 0:
                logger.info(f"\n📊 Class Correction Statistics:")
                logger.info(f"  Attempted: {stats['corrections_attempted']}")
                logger.info(f"  Applied: {stats['corrections_applied']}")
                logger.info(f"  Success rate: {stats['correction_rate']:.1%}")
        
        logger.info("  Sample descriptions:")
        for i, entity in enumerate(entities[:5]):
            if entity.description:
                desc_preview = entity.description[:80] + "..." if len(entity.description) > 80 else entity.description
                
                # Show correction if it happened
                class_info = entity.object_class.value
                if hasattr(entity, 'corrected_class') and entity.corrected_class:
                    class_info = f"{entity.object_class.value} → {entity.corrected_class}"
                
                logger.info(f"    {i+1}. {entity.entity_id} ({class_info}): {desc_preview}")
        
        logger.info("="*80 + "\n")
        
        return entities
    
    def _apply_quality_improvements(
        self,
        entities: List[PerceptionEntity],
    ) -> List[PerceptionEntity]:
        """
        Apply spatial analysis and class correction to entities.
        
        Args:
            entities: Entities with descriptions
            
        Returns:
            Enriched entities
        """
        spatial_count = 0
        correction_count = 0
        
        for entity in entities:
            # Get best observation for analysis
            best_obs = entity.get_best_observation()
            if not best_obs:
                continue
            
            # Apply spatial analysis
            if self.enable_spatial_analysis and best_obs.bounding_box:
                best_obs.spatial_zone = calculate_spatial_zone(
                    best_obs.bounding_box,
                    frame_width=1920.0,  # TODO: Get from video metadata
                    frame_height=1080.0,
                )
                spatial_count += 1
            
            # Apply class correction
            if self.enable_class_correction and self.class_corrector and entity.description:
                # Calculate average confidence
                avg_confidence = sum(obs.confidence for obs in entity.observations) / len(entity.observations)
                
                # Attempt correction
                entity = self.class_corrector.correct_entity_class(
                    entity,
                    entity.description,
                    avg_confidence,
                )
                
                if hasattr(entity, 'corrected_class') and entity.corrected_class:
                    correction_count += 1
        
        if spatial_count > 0:
            logger.info(f"  ✓ Added spatial context to {spatial_count} entities")
        
        if correction_count > 0:
            logger.info(f"  ✓ Corrected {correction_count} misclassifications")
        
        return entities
    
    def describe_observation(self, observation: Observation) -> str:
        """
        Generate UNBIASED description for a single observation.
        
        **CRITICAL:** Does NOT provide YOLO class hint to avoid confirmation bias.
        The VLM describes what it sees independently, then we verify against YOLO.
        
        Args:
            observation: Observation to describe
            
        Returns:
            Natural language description (unbiased by YOLO label)
        """
        # Get crop image
        crop = observation.image_patch
        if crop is None:
            logger.warning(f"No image patch for observation at frame {observation.frame_number}")
            return f"An object detected by YOLO as {observation.object_class.value}"

        if not self._is_crop_valid(crop):
            logger.warning(
                "Image patch failed sanity checks (frame=%s, size=%sx%s, std=%.2f)", 
                observation.frame_number,
                crop.shape[1],
                crop.shape[0],
                float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std()) if crop.size else 0.0,
            )
            return (
                f"An object with unclear visual signal, originally detected as "
                f"{observation.object_class.value}"
            )
        
        # Convert BGR to RGB
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)
        
        # UNBIASED PROMPT - No class hint!
        # This allows FastVLM to describe what it actually sees
        prompt = "Describe what you see in this image in detail. Focus on the main object, its appearance, color, shape, and any distinguishing features."
        
        try:
            description = self.vlm.generate(
                image=pil_image,
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            
            # Clean up description
            description = description.strip()
            
            # Validate description
            if not description or len(description) < 10:
                logger.warning(f"Short/empty description generated, using fallback")
                description = f"An object with characteristics suggesting {observation.object_class.value}"
            
            return description
        
        except Exception as e:
            logger.error(f"Failed to generate description: {e}")
            return f"An object detected by YOLO as {observation.object_class.value}"
    
    def select_best_observation(self, entity: PerceptionEntity) -> Optional[Observation]:
        """
        Select the best observation for description.
        
        Uses weighted scoring:
        - Size weight: Larger objects are clearer
        - Centrality weight: Objects closer to center are less occluded
        - Confidence weight: Higher confidence = better detection
        
        Args:
            entity: Entity to select from
            
        Returns:
            Best observation, or None if entity has no observations
        """
        if not entity.observations:
            return None
        
        best_obs = None
        best_score = -1
        
        for obs in entity.observations:
            # Compute score
            score = self._score_observation(obs)
            
            if score > best_score:
                best_score = score
                best_obs = obs
        
        return best_obs
    
    def _score_observation(self, obs: Observation) -> float:
        """
        Score an observation for description quality.
        
        Args:
            obs: Observation to score
            
        Returns:
            Score (higher is better)
        """
        # Size score (normalized by frame area)
        bbox = obs.bounding_box
        frame_area = obs.frame_number * 1.0  # Placeholder - should use actual frame size
        if frame_area > 0:
            size_score = bbox.area / max(frame_area, 1)
        else:
            size_score = bbox.area / 1000000  # Fallback
        
        # Centrality score (distance from frame center)
        # Assume frame center at (frame_width/2, frame_height/2)
        # For now, use a simple heuristic
        cx, cy = obs.centroid
        centrality_score = 1.0  # Placeholder - would need frame dimensions
        
        # Confidence score
        confidence_score = obs.confidence
        
        # Weighted combination
        total_score = (
            self.config.size_weight * size_score +
            self.config.centrality_weight * centrality_score +
            self.config.confidence_weight * confidence_score
        )
        
        return total_score

    def _is_crop_valid(self, crop: np.ndarray) -> bool:
        """Basic pixel-level sanity checks before trusting FastVLM output."""
        if crop is None or crop.size == 0:
            return False

        height, width = crop.shape[:2]
        if height < self.config.min_crop_size or width < self.config.min_crop_size:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < self.config.min_crop_std:
            return False

        return True
