"""
Action Genome Ground Truth Adapter
===================================

Converts Action Genome annotations to standardized GroundTruthGraph format
for evaluation.

This adapter handles:
- Person and object bounding boxes
- Spatial relationships between objects
- Actions/events with temporal bounds
- Causal relationships (inferred from temporal ordering)

Author: Orion Research Team
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from orion.evaluation.benchmarks.action_genome_loader import ActionGenomeDataset, AGObject, AGRelationship, AGAction
from orion.evaluation.benchmark_evaluator import GroundTruthGraph

logger = logging.getLogger("orion.evaluation.ag_adapter")


class ActionGenomeAdapter:
    """
    Convert Action Genome data to standardized GroundTruthGraph
    """
    
    def __init__(self, fps: float = 30.0):
        """
        Args:
            fps: Frames per second for temporal conversion
        """
        self.fps = fps
    
    def convert_to_ground_truth(
        self,
        ag_dataset: ActionGenomeDataset,
    ) -> GroundTruthGraph:
        """
        Convert Action Genome dataset to GroundTruthGraph
        
        Args:
            ag_dataset: Loaded Action Genome dataset
        
        Returns:
            GroundTruthGraph ready for evaluation
        """
        # 1. Convert objects to entities
        entities = self._convert_entities(ag_dataset.objects)
        
        # 2. Convert relationships
        relationships = self._convert_relationships(ag_dataset.relationships)
        
        # 3. Convert actions to events
        events = self._convert_events(ag_dataset.actions)
        
        # 4. Infer causal links from temporal ordering of actions
        causal_links = self._infer_causal_links(ag_dataset.actions, events)
        
        # 5. Get temporal info
        total_frames = 0
        if ag_dataset.objects:
            total_frames = max(obj.frame_id for obj in ag_dataset.objects)
        
        gt = GroundTruthGraph(
            video_id=ag_dataset.clip_id,
            entities=entities,
            relationships=relationships,
            events=events,
            causal_links=causal_links,
            fps=self.fps,
            total_frames=total_frames,
            dataset_name="action_genome",
            metadata={
                "num_objects": len(ag_dataset.objects),
                "num_relationships": len(ag_dataset.relationships),
                "num_actions": len(ag_dataset.actions),
            },
        )
        
        logger.info(
            f"Converted AG clip {ag_dataset.clip_id}: "
            f"{len(entities)} entities, {len(relationships)} relationships, "
            f"{len(events)} events, {len(causal_links)} causal links"
        )
        
        return gt
    
    def _convert_entities(
        self,
        objects: List[AGObject],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert AG objects to entities
        
        Aggregates frame-level objects into track-level entities
        """
        # Group objects by ID
        tracks = defaultdict(list)
        for obj in objects:
            tracks[obj.object_id].append(obj)
        
        entities = {}
        for obj_id, obj_list in tracks.items():
            # Sort by frame
            obj_list.sort(key=lambda x: x.frame_id)
            
            # Extract temporal info
            frames = [obj.frame_id for obj in obj_list]
            first_frame = min(frames)
            last_frame = max(frames)
            
            # Extract bboxes by frame
            bboxes = {}
            for obj in obj_list:
                bboxes[obj.frame_id] = obj.bbox
            
            # Get class (should be consistent)
            obj_class = obj_list[0].class_label
            
            entity = {
                "entity_id": obj_id,
                "class": obj_class,
                "label": obj_class,
                "frames": frames,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "bboxes": bboxes,
                "attributes": obj_list[0].attributes,
            }
            
            entities[obj_id] = entity
        
        return entities
    
    def _convert_relationships(
        self,
        ag_relationships: List[AGRelationship],
    ) -> List[Dict[str, Any]]:
        """
        Convert AG relationships to standardized format
        """
        relationships = []
        for rel in ag_relationships:
            relationships.append({
                "subject": rel.subject_id,
                "object": rel.object_id,
                "predicate": self._normalize_predicate(rel.predicate),
                "frame_id": rel.frame_id,
                "confidence": rel.confidence,
            })
        
        return relationships
    
    @staticmethod
    def _normalize_predicate(predicate: str) -> str:
        """Normalize AG predicates to standard names"""
        # Action Genome has predicates like:
        # "on", "in", "next_to", "holding", "looking_at", etc.
        
        # Map common variations
        mapping = {
            "on": "on",
            "in": "in",
            "next to": "next_to",
            "next_to": "next_to",
            "holding": "holding",
            "looking at": "looking_at",
            "looking_at": "looking_at",
            "sitting on": "sitting_on",
            "standing on": "standing_on",
            "wearing": "wearing",
            "touching": "touching",
        }
        
        normalized = predicate.lower().replace(' ', '_')
        return mapping.get(normalized, normalized)
    
    def _convert_events(
        self,
        ag_actions: List[AGAction],
    ) -> List[Dict[str, Any]]:
        """
        Convert AG actions to events
        """
        events = []
        for action in ag_actions:
            event = {
                "event_id": action.action_id,
                "type": self._normalize_action_type(action.action_class),
                "start_frame": action.start_frame,
                "end_frame": action.end_frame,
                "entities": [action.person_id] + action.objects_involved,
                "agent": action.person_id,
                "patients": action.objects_involved,
            }
            events.append(event)
        
        return events
    
    @staticmethod
    def _normalize_action_type(action_class: str) -> str:
        """Normalize AG action classes"""
        # Action Genome actions: "open", "close", "pick_up", "put_down", etc.
        return action_class.lower().replace(' ', '_')
    
    def _infer_causal_links(
        self,
        ag_actions: List[AGAction],
        events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Infer causal links from temporal ordering of actions
        
        Heuristic: If two actions involve the same object and are temporally close,
        there may be a causal relationship.
        
        Examples:
        - "pick_up" followed by "put_down" on same object
        - "open" followed by "enter" through same door
        """
        causal_links = []
        
        # Sort actions by start time
        sorted_actions = sorted(ag_actions, key=lambda a: a.start_frame)
        
        # Look for causal patterns
        for i in range(len(sorted_actions) - 1):
            action1 = sorted_actions[i]
            
            for j in range(i + 1, len(sorted_actions)):
                action2 = sorted_actions[j]
                
                # Check temporal proximity (within 3 seconds)
                time_diff = (action2.start_frame - action1.end_frame) / self.fps
                if time_diff > 3.0:
                    break  # Too far apart
                
                # Check if actions share objects
                shared_objects = set(action1.objects_involved) & set(action2.objects_involved)
                
                # Check for causal patterns
                is_causal = False
                
                # Pattern 1: pick_up -> put_down
                if action1.action_class == "pick_up" and action2.action_class == "put_down":
                    if shared_objects:
                        is_causal = True
                
                # Pattern 2: open -> enter
                if action1.action_class == "open" and action2.action_class in ["enter", "walk_through"]:
                    if shared_objects:
                        is_causal = True
                
                # Pattern 3: close -> leave
                if action1.action_class == "close" and action2.action_class == "leave":
                    if shared_objects:
                        is_causal = True
                
                # Pattern 4: Sequential actions on same object
                if shared_objects and len(shared_objects) > 0:
                    is_causal = True
                
                if is_causal:
                    # Find corresponding event IDs
                    cause_event_id = action1.action_id
                    effect_event_id = action2.action_id
                    
                    causal_links.append({
                        "cause": cause_event_id,
                        "effect": effect_event_id,
                        "time_diff": time_diff,
                        "confidence": 0.8,  # Heuristic confidence
                    })
        
        logger.debug(f"Inferred {len(causal_links)} causal links from {len(ag_actions)} actions")
        return causal_links


def convert_action_genome_clip(
    ag_dataset: ActionGenomeDataset,
    fps: float = 30.0,
) -> GroundTruthGraph:
    """
    Convenience function to convert Action Genome clip to GroundTruthGraph
    
    Args:
        ag_dataset: Loaded Action Genome dataset
        fps: Frames per second
    
    Returns:
        GroundTruthGraph
    """
    adapter = ActionGenomeAdapter(fps=fps)
    return adapter.convert_to_ground_truth(ag_dataset)


if __name__ == "__main__":
    # Example usage
    from orion.evaluation.benchmarks.action_genome_loader import ActionGenomeBenchmark
    
    # Load benchmark
    benchmark = ActionGenomeBenchmark("/path/to/action_genome")
    
    # Convert a clip
    if benchmark.clips:
        clip_id = list(benchmark.clips.keys())[0]
        ag_dataset = benchmark.clips[clip_id]
        
        gt = convert_action_genome_clip(ag_dataset)
        
        print(f"Ground Truth for {gt.video_id}:")
        print(f"  Entities: {len(gt.entities)}")
        print(f"  Relationships: {len(gt.relationships)}")
        print(f"  Events: {len(gt.events)}")
        print(f"  Causal Links: {len(gt.causal_links)}")
