"""
Scene Assembler
================

Assembles temporal windows into coherent scene segments.

Responsibilities:
- Group temporal windows into scenes
- Infer location profiles
- Detect scene boundaries
- Compute scene-level statistics

Author: Orion Research Team
Date: October 2025
"""

import logging
from typing import Dict, Set
from collections import defaultdict

from orion.semantic.types import TemporalWindow, SceneSegment, LocationProfile
from orion.semantic.config import TemporalWindowConfig

logger = logging.getLogger(__name__)


class SceneAssembler:
    """
    Assembles temporal windows into scene segments.
    
    Groups windows by location and temporal proximity to identify
    coherent scenes with stable contexts.
    """
    
    def __init__(self, config: TemporalWindowConfig):
        """
        Initialize scene assembler.
        
        Args:
            config: Configuration for scene assembly
        """
        self.config = config
        self.scenes: list[SceneSegment] = []
        
        logger.debug("SceneAssembler initialized")
    
    def assemble_scenes(
        self,
        windows: list[TemporalWindow],
    ) -> list[SceneSegment]:
        """
        Assemble temporal windows into scene segments.
        
        Args:
            windows: list of temporal windows
            
        Returns:
            list of scene segments
        """
        logger.info("="*80)
        logger.info("PHASE 2D: SCENE ASSEMBLY")
        logger.info("="*80)
        
        if not windows:
            logger.warning("No windows to assemble into scenes")
            # Create a default scene as fallback
            logger.info("  Creating default scene as fallback...")
            default_scene = SceneSegment(
                start_time=0.0,
                end_time=10.0,  # Default 10 second scene
                scene_id="scene_default",
            )
            logger.info("  ✓ Created 1 default scene")
            logger.info("="*80 + "\n")
            return [default_scene]
        
        logger.info(f"Assembling {len(windows)} temporal windows into scenes...")
        
        # Sort windows by time
        sorted_windows = sorted(windows, key=lambda w: w.start_time)
        
        scenes = []
        current_scene = None
        
        for window in sorted_windows:
            if current_scene is None:
                # Start new scene
                current_scene = SceneSegment(
                    start_time=window.start_time,
                    end_time=window.end_time,
                )
                current_scene.add_window(window)
            else:
                # Check if window belongs to current scene
                time_gap = window.start_time - current_scene.end_time
                
                if time_gap <= self.config.max_gap_between_changes * 2:
                    # Continue current scene
                    current_scene.add_window(window)
                else:
                    # Start new scene
                    scenes.append(current_scene)
                    current_scene = SceneSegment(
                        start_time=window.start_time,
                        end_time=window.end_time,
                    )
                    current_scene.add_window(window)
        
        # Add final scene
        if current_scene:
            scenes.append(current_scene)
        
        # Infer locations for each scene
        for scene in scenes:
            scene.location_profile = self._infer_location(scene)
        
        logger.info(f"\n✓ Assembled {len(scenes)} scene segments")
        
        if scenes:
            logger.info("  Scene statistics:")
            total_windows = sum(len(s.temporal_windows) for s in scenes)
            avg_duration = sum(s.duration for s in scenes) / len(scenes)
            logger.info(f"    Total temporal windows: {total_windows}")
            logger.info(f"    Average scene duration: {avg_duration:.2f}s")
            logger.info(f"    Average windows per scene: {total_windows / len(scenes):.1f}")
        
        logger.info("="*80 + "\n")
        
        self.scenes = scenes
        return scenes
    
    def _infer_location(self, scene: SceneSegment) -> LocationProfile:
        """
        Infer location profile for a scene.
        
        Args:
            scene: Scene segment to analyze
            
        Returns:
            Inferred location profile
        """
        # Count entity class frequencies
        class_counts: dict[str, int] = defaultdict(int)
        
        for window in scene.temporal_windows:
            for entity_id in window.active_entities:
                # Extract class from entity_id (format: "class_N")
                if "_" in entity_id:
                    obj_class = entity_id.split("_")[0]
                    class_counts[obj_class] += 1
        
        # Determine scene type based on dominant classes
        indoor_classes = {"couch", "chair", "bed", "tv", "laptop", "book"}
        outdoor_classes = {"car", "truck", "bicycle", "motorcycle", "bus"}
        kitchen_classes = {"bowl", "cup", "fork", "knife", "bottle", "microwave"}
        
        indoor_count = sum(class_counts[c] for c in indoor_classes if c in class_counts)
        outdoor_count = sum(class_counts[c] for c in outdoor_classes if c in class_counts)
        kitchen_count = sum(class_counts[c] for c in kitchen_classes if c in class_counts)
        
        # Simple heuristic
        if kitchen_count > indoor_count and kitchen_count > outdoor_count:
            scene_type = "kitchen"
        elif outdoor_count > indoor_count:
            scene_type = "outdoor"
        else:
            scene_type = "indoor"
        
        # Extract dominant entities
        dominant_entities = sorted(
            class_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Create LocationProfile with proper fields
        return LocationProfile(
            location_id=f"location_{scene.start_time}",
            signature=f"{scene_type}|{len(dominant_entities)}entities",
            label=scene_type.capitalize(),
            object_classes=[e[0] for e in dominant_entities],
            zone_ids=[],
            scene_ids=[],
        )
    
    def get_scene_at_time(self, timestamp: float) -> list[SceneSegment]:
        """
        Get scenes active at a given timestamp.
        
        Args:
            timestamp: Target timestamp
            
        Returns:
            list of scenes overlapping timestamp
        """
        return [
            s for s in self.scenes
            if s.start_time <= timestamp <= s.end_time
        ]
