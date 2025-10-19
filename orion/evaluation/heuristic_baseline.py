"""
Heuristic Baseline Knowledge Graph Constructor
===============================================

This module implements a rule-based baseline that receives the same
Structured Perception Log as the main system but uses hand-crafted
if/then rules instead of AI-based causal inference.

This serves as a rigorous baseline for evaluation, demonstrating the
value added by our CIS + LLM approach.

Author: Orion Research Team  
Date: October 2025
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("HeuristicBaseline")


@dataclass
class HeuristicConfig:
    """Configuration for heuristic rule thresholds"""
    
    # Proximity rules
    proximity_distance_threshold: float = 50.0  # pixels
    proximity_duration_threshold: int = 10      # frames
    
    # Containment rules
    containment_overlap_threshold: float = 0.95  # 95% overlap
    
    # Causal rules
    causal_max_time_gap: float = 2.0  # seconds
    
    # State change detection
    state_change_keyword_threshold: float = 0.3  # word overlap threshold


class HeuristicBaseline:
    """
    Rule-based knowledge graph constructor using hand-crafted heuristics
    
    This baseline demonstrates what can be achieved without ML-based
    causal inference or LLM reasoning.
    """
    
    def __init__(self, config: Optional[HeuristicConfig] = None):
        """
        Args:
            config: Configuration for rule thresholds
        """
        self.config = config or HeuristicConfig()
        self.entities: Dict[str, Dict] = {}
        self.relationships: List[Dict] = []
        self.events: List[Dict] = []
        
        logger.info(f"Initialized HeuristicBaseline with config: {self.config}")
    
    def process_perception_log(
        self,
        perception_objects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a perception log and construct knowledge graph using rules
        
        Args:
            perception_objects: List of RichPerceptionObject dictionaries
            
        Returns:
            Dictionary containing entities, relationships, and events
        """
        logger.info(f"Processing {len(perception_objects)} perception objects...")
        
        # Step 1: Extract unique entities (simple clustering by temp_id)
        self._extract_entities(perception_objects)
        
        # Step 2: Apply proximity rules
        self._apply_proximity_rules(perception_objects)
        
        # Step 3: Apply containment rules
        self._apply_containment_rules(perception_objects)
        
        # Step 4: Detect state changes and apply causal rules
        self._apply_causal_rules(perception_objects)
        
        logger.info(
            f"Heuristic baseline generated: "
            f"{len(self.entities)} entities, "
            f"{len(self.relationships)} relationships, "
            f"{len(self.events)} events"
        )
        
        return {
            "entities": list(self.entities.values()),
            "relationships": self.relationships,
            "events": self.events,
        }
    
    def _extract_entities(self, perception_objects: List[Dict[str, Any]]):
        """
        Extract unique entities from perception log
        
        In the heuristic baseline, we simply use temp_id as entity_id
        (no sophisticated visual embedding clustering)
        """
        entity_observations = {}
        
        for obj in perception_objects:
            temp_id = obj.get("temp_id") or obj.get("entity_id", "unknown")
            
            if temp_id not in entity_observations:
                entity_observations[temp_id] = []
            
            entity_observations[temp_id].append(obj)
        
        # Create entity nodes
        for entity_id, observations in entity_observations.items():
            # Take most recent description
            descriptions = [
                obs.get("rich_description", obs.get("object_class", "unknown"))
                for obs in observations
                if obs.get("rich_description") or obs.get("object_class")
            ]
            
            primary_description = descriptions[-1] if descriptions else "unknown"
            
            self.entities[entity_id] = {
                "entity_id": entity_id,
                "class": observations[0].get("object_class", "unknown"),
                "description": primary_description,
                "first_seen": observations[0].get("timestamp", 0.0),
                "last_seen": observations[-1].get("timestamp", 0.0),
                "appearance_count": len(observations),
            }
    
    def _apply_proximity_rules(self, perception_objects: List[Dict[str, Any]]):
        """
        Rule: If two objects are within proximity_distance_threshold for
        proximity_duration_threshold consecutive frames, create IS_NEAR relationship
        """
        # Group by frame
        frames = {}
        for obj in perception_objects:
            frame_num = obj.get("frame_number", 0)
            if frame_num not in frames:
                frames[frame_num] = []
            frames[frame_num].append(obj)
        
        # Track consecutive proximity
        proximity_tracker = {}  # (entity1, entity2) -> consecutive_count
        
        for frame_num in sorted(frames.keys()):
            objects_in_frame = frames[frame_num]
            
            # Check all pairs
            for i, obj1 in enumerate(objects_in_frame):
                for obj2 in objects_in_frame[i + 1:]:
                    eid1 = obj1.get("temp_id") or obj1.get("entity_id")
                    eid2 = obj2.get("temp_id") or obj2.get("entity_id")
                    
                    if eid1 == eid2:
                        continue
                    
                    # Calculate distance between centroids
                    bbox1 = obj1.get("bounding_box", [0, 0, 0, 0])
                    bbox2 = obj2.get("bounding_box", [0, 0, 0, 0])
                    
                    centroid1 = self._get_centroid(bbox1)
                    centroid2 = self._get_centroid(bbox2)
                    
                    distance = self._euclidean_distance(centroid1, centroid2)
                    
                    # Create canonical pair key (sorted)
                    pair = tuple(sorted([eid1, eid2]))
                    
                    if distance < self.config.proximity_distance_threshold:
                        # Increment proximity counter
                        proximity_tracker[pair] = proximity_tracker.get(pair, 0) + 1
                        
                        # Create relationship if threshold met
                        if proximity_tracker[pair] >= self.config.proximity_duration_threshold:
                            # Only create once
                            if not self._relationship_exists(eid1, eid2, "IS_NEAR"):
                                self.relationships.append({
                                    "source": eid1,
                                    "target": eid2,
                                    "type": "IS_NEAR",
                                    "frame_start": frame_num - self.config.proximity_duration_threshold + 1,
                                    "frame_end": frame_num,
                                })
                    else:
                        # Reset counter if not close
                        proximity_tracker[pair] = 0
    
    def _apply_containment_rules(self, perception_objects: List[Dict[str, Any]]):
        """
        Rule: If object A's bbox is containment_overlap_threshold% inside
        object B's bbox, create IS_INSIDE relationship
        """
        # Group by frame
        frames = {}
        for obj in perception_objects:
            frame_num = obj.get("frame_number", 0)
            if frame_num not in frames:
                frames[frame_num] = []
            frames[frame_num].append(obj)
        
        for frame_num, objects_in_frame in frames.items():
            for i, obj1 in enumerate(objects_in_frame):
                for obj2 in objects_in_frame[i + 1:]:
                    eid1 = obj1.get("temp_id") or obj1.get("entity_id")
                    eid2 = obj2.get("temp_id") or obj2.get("entity_id")
                    
                    if eid1 == eid2:
                        continue
                    
                    bbox1 = obj1.get("bounding_box", [0, 0, 0, 0])
                    bbox2 = obj2.get("bounding_box", [0, 0, 0, 0])
                    
                    # Check if bbox1 is inside bbox2
                    overlap1_in_2 = self._overlap_ratio(bbox1, bbox2)
                    overlap2_in_1 = self._overlap_ratio(bbox2, bbox1)
                    
                    if overlap1_in_2 >= self.config.containment_overlap_threshold:
                        # obj1 is inside obj2
                        if not self._relationship_exists(eid1, eid2, "IS_INSIDE"):
                            self.relationships.append({
                                "source": eid1,
                                "target": eid2,
                                "type": "IS_INSIDE",
                                "frame": frame_num,
                            })
                    
                    elif overlap2_in_1 >= self.config.containment_overlap_threshold:
                        # obj2 is inside obj1
                        if not self._relationship_exists(eid2, eid1, "IS_INSIDE"):
                            self.relationships.append({
                                "source": eid2,
                                "target": eid1,
                                "type": "IS_INSIDE",
                                "frame": frame_num,
                            })
    
    def _apply_causal_rules(self, perception_objects: List[Dict[str, Any]]):
        """
        Rule: If object A is near object B, and B's description changes,
        then A CAUSED the state change
        
        This is the naive baseline for causal reasoning
        """
        # Detect state changes
        state_changes = self._detect_state_changes(perception_objects)
        
        for change in state_changes:
            patient_id = change["entity_id"]
            change_time = change["timestamp"]
            
            # Find nearby entities at the time of change
            nearby_entities = self._find_nearby_at_time(
                perception_objects,
                patient_id,
                change_time
            )
            
            # Create CAUSED relationship for all nearby entities
            for agent_id in nearby_entities:
                self.events.append({
                    "type": "StateChange",
                    "agent": agent_id,
                    "patient": patient_id,
                    "relationship": "CAUSED",
                    "timestamp": change_time,
                    "old_state": change["old_description"],
                    "new_state": change["new_description"],
                    "method": "heuristic_proximity",
                })
    
    def _detect_state_changes(
        self,
        perception_objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect state changes by comparing consecutive descriptions
        """
        # Group by entity
        entity_timeline = {}
        for obj in perception_objects:
            eid = obj.get("temp_id") or obj.get("entity_id")
            if eid not in entity_timeline:
                entity_timeline[eid] = []
            entity_timeline[eid].append(obj)
        
        state_changes = []
        
        for eid, timeline in entity_timeline.items():
            # Sort by timestamp
            timeline.sort(key=lambda x: x.get("timestamp", 0))
            
            prev_desc = None
            for obj in timeline:
                curr_desc = obj.get("rich_description")
                
                if curr_desc and prev_desc and curr_desc != prev_desc:
                    # Detected state change
                    state_changes.append({
                        "entity_id": eid,
                        "timestamp": obj.get("timestamp", 0.0),
                        "old_description": prev_desc,
                        "new_description": curr_desc,
                    })
                
                if curr_desc:
                    prev_desc = curr_desc
        
        return state_changes
    
    def _find_nearby_at_time(
        self,
        perception_objects: List[Dict[str, Any]],
        target_entity: str,
        timestamp: float
    ) -> Set[str]:
        """
        Find entities near the target entity at a specific timestamp
        """
        nearby = set()
        
        # Find target's position at timestamp
        target_pos = None
        for obj in perception_objects:
            eid = obj.get("temp_id") or obj.get("entity_id")
            if eid == target_entity:
                obj_time = obj.get("timestamp", 0.0)
                if abs(obj_time - timestamp) < self.config.causal_max_time_gap:
                    target_pos = self._get_centroid(obj.get("bounding_box", [0, 0, 0, 0]))
                    break
        
        if not target_pos:
            return nearby
        
        # Find other entities nearby at similar time
        for obj in perception_objects:
            eid = obj.get("temp_id") or obj.get("entity_id")
            if eid == target_entity:
                continue
            
            obj_time = obj.get("timestamp", 0.0)
            if abs(obj_time - timestamp) < self.config.causal_max_time_gap:
                obj_pos = self._get_centroid(obj.get("bounding_box", [0, 0, 0, 0]))
                distance = self._euclidean_distance(target_pos, obj_pos)
                
                if distance < self.config.proximity_distance_threshold:
                    nearby.add(eid)
        
        return nearby
    
    def _get_centroid(self, bbox: List[int]) -> Tuple[float, float]:
        """Calculate centroid of bounding box"""
        if len(bbox) != 4:
            return (0.0, 0.0)
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    
    def _euclidean_distance(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance"""
        return np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    def _overlap_ratio(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate what fraction of bbox1 is inside bbox2"""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
        
        x1_max = max(bbox1[0], bbox2[0])
        y1_max = max(bbox1[1], bbox2[1])
        x2_min = min(bbox1[2], bbox2[2])
        y2_min = min(bbox1[3], bbox2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        
        if bbox1_area == 0:
            return 0.0
        
        return intersection / bbox1_area
    
    def _relationship_exists(
        self,
        source: str,
        target: str,
        rel_type: str
    ) -> bool:
        """Check if a relationship already exists"""
        for rel in self.relationships:
            if (
                rel["source"] == source and
                rel["target"] == target and
                rel["type"] == rel_type
            ):
                return True
        return False
    
    def export_to_json(self, output_path: str):
        """Export knowledge graph to JSON file"""
        data = {
            "entities": list(self.entities.values()),
            "relationships": self.relationships,
            "events": self.events,
            "metadata": {
                "method": "heuristic_baseline",
                "config": {
                    "proximity_threshold": self.config.proximity_distance_threshold,
                    "containment_threshold": self.config.containment_overlap_threshold,
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported heuristic baseline to {output_path}")
