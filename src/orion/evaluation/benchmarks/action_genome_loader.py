"""
Action Genome Benchmark Loader
===============================

Loader for the Action Genome dataset - a widely-used benchmark for
video scene graph generation with 10K clips and dense annotations.

Dataset: https://github.com/JingweiJ/ActionGenome
Paper: "Action Genome: Actions as Composition of Spatio-temporal Scene Graphs" (CVPR 2020)

Expected Directory Structure:
```
action_genome/
├── videos/
│   ├── clip_001.mp4
│   └── ...
├── annotations/
│   ├── person_bbox.pkl
│   ├── object_bbox_and_relationship.pkl
│   └── ...
└── metadata/
    ├── object_classes.txt
    └── relationship_classes.txt
```

Author: Orion Research Team
Date: October 2025
"""

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ActionGenomeBenchmark")


@dataclass
class AGObject:
    """Object in Action Genome ground truth"""
    object_id: str
    class_label: str
    bbox: List[int]  # [x1, y1, x2, y2]
    frame_id: int
    attributes: Dict[str, Any]


@dataclass
class AGRelationship:
    """Relationship in Action Genome ground truth"""
    subject_id: str
    object_id: str
    predicate: str
    frame_id: int
    confidence: float = 1.0


@dataclass
class AGAction:
    """Action/Event in Action Genome"""
    action_id: str
    action_class: str
    person_id: str
    start_frame: int
    end_frame: int
    objects_involved: List[str]


class ActionGenomeDataset:
    """
    Represents a single video clip from Action Genome with annotations
    """
    
    def __init__(
        self,
        clip_id: str,
        video_path: Path,
        annotation_data: Dict[str, Any]
    ):
        """
        Args:
            clip_id: Unique identifier for this clip
            video_path: Path to video file
            annotation_data: Parsed annotation data
        """
        self.clip_id = clip_id
        self.video_path = video_path
        self.raw_annotations = annotation_data
        
        # Parse annotations
        self.objects = self._parse_objects(annotation_data.get("objects", {}))
        self.relationships = self._parse_relationships(
            annotation_data.get("relationships", [])
        )
        self.actions = self._parse_actions(annotation_data.get("actions", []))
        
        logger.debug(
            f"Loaded AG clip '{clip_id}': "
            f"{len(self.objects)} objects, "
            f"{len(self.relationships)} relationships, "
            f"{len(self.actions)} actions"
        )
    
    def _parse_objects(self, objects_data: Dict) -> List[AGObject]:
        """Parse object annotations"""
        objects = []
        for frame_id, frame_objects in objects_data.items():
            for obj_data in frame_objects:
                obj = AGObject(
                    object_id=obj_data.get("id", "unknown"),
                    class_label=obj_data.get("class", "object"),
                    bbox=obj_data.get("bbox", [0, 0, 0, 0]),
                    frame_id=int(frame_id),
                    attributes=obj_data.get("attributes", {})
                )
                objects.append(obj)
        return objects
    
    def _parse_relationships(
        self,
        relationships_data: List[Dict]
    ) -> List[AGRelationship]:
        """Parse relationship annotations"""
        relationships = []
        for rel_data in relationships_data:
            rel = AGRelationship(
                subject_id=rel_data.get("subject", ""),
                object_id=rel_data.get("object", ""),
                predicate=rel_data.get("predicate", "related_to"),
                frame_id=rel_data.get("frame_id", 0),
                confidence=rel_data.get("confidence", 1.0)
            )
            relationships.append(rel)
        return relationships
    
    def _parse_actions(self, actions_data: List[Dict]) -> List[AGAction]:
        """Parse action/event annotations"""
        actions = []
        for action_data in actions_data:
            action = AGAction(
                action_id=action_data.get("id", "unknown"),
                action_class=action_data.get("class", "action"),
                person_id=action_data.get("person_id", ""),
                start_frame=action_data.get("start_frame", 0),
                end_frame=action_data.get("end_frame", 0),
                objects_involved=action_data.get("objects", [])
            )
            actions.append(action)
        return actions
    
    def to_orion_format(self) -> Dict[str, Any]:
        """
        Convert Action Genome ground truth to Orion's graph format
        
        Returns:
            Dictionary with entities, relationships, events in Orion format
        """
        # Extract unique entities from objects
        entity_map = {}
        for obj in self.objects:
            if obj.object_id not in entity_map:
                entity_map[obj.object_id] = {
                    "entity_id": obj.object_id,
                    "class": obj.class_label,
                    "description": obj.class_label,
                    "first_seen": obj.frame_id / 30.0,  # Assume 30 FPS
                    "last_seen": obj.frame_id / 30.0,
                    "attributes": obj.attributes,
                }
            else:
                # Update last_seen
                entity_map[obj.object_id]["last_seen"] = obj.frame_id / 30.0
        
        orion_entities = list(entity_map.values())
        
        # Convert relationships
        orion_relationships = []
        for rel in self.relationships:
            orion_relationships.append({
                "source": rel.subject_id,
                "target": rel.object_id,
                "type": rel.predicate,
                "frame": rel.frame_id,
                "timestamp": rel.frame_id / 30.0,
                "confidence": rel.confidence,
            })
        
        # Convert actions to events
        orion_events = []
        for action in self.actions:
            event = {
                "type": action.action_class,
                "agent": action.person_id,
                "relationship": action.action_class,
                "timestamp": (action.start_frame + action.end_frame) / 2 / 30.0,
                "start_time": action.start_frame / 30.0,
                "end_time": action.end_frame / 30.0,
                "objects_involved": action.objects_involved,
            }
            
            # Add patient if available
            if action.objects_involved:
                event["patient"] = action.objects_involved[0]
            
            orion_events.append(event)
        
        return {
            "entities": orion_entities,
            "relationships": orion_relationships,
            "events": orion_events,
            "metadata": {
                "clip_id": self.clip_id,
                "source": "action_genome_ground_truth",
                "num_frames": max([obj.frame_id for obj in self.objects]) if self.objects else 0,
            }
        }


class ActionGenomeBenchmark:
    """
    Loader and evaluator for the Action Genome benchmark dataset
    """
    
    def __init__(self, dataset_root: str):
        """
        Args:
            dataset_root: Path to Action Genome dataset root directory
        """
        self.root = Path(dataset_root)
        self.videos_dir = self.root / "videos"
        self.annotations_dir = self.root / "annotations"
        self.metadata_dir = self.root / "metadata"
        
        # Validate structure
        if not self.root.exists():
            raise FileNotFoundError(f"Action Genome dataset not found at {dataset_root}")
        
        # Load class mappings
        self.object_classes = self._load_class_mapping("object_classes.txt")
        self.relationship_classes = self._load_class_mapping("relationship_classes.txt")
        
        # Discover clips
        self.clips = self._discover_clips()
        
        logger.info(
            f"Loaded Action Genome benchmark from {dataset_root} "
            f"with {len(self.clips)} clips"
        )
    
    def _load_class_mapping(self, filename: str) -> Dict[int, str]:
        """Load class index to name mapping"""
        mapping_path = self.metadata_dir / filename
        if not mapping_path.exists():
            logger.warning(f"Class mapping not found: {mapping_path}")
            return {}
        
        mapping = {}
        with open(mapping_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    idx, name = line.split('\t', 1)
                    mapping[int(idx)] = name
        
        return mapping
    
    def _discover_clips(self) -> Dict[str, ActionGenomeDataset]:
        """Discover and load all clips in the dataset"""
        clips = {}
        
        # Action Genome uses pickle files for annotations
        person_bbox_file = self.annotations_dir / "person_bbox.pkl"
        object_bbox_file = self.annotations_dir / "object_bbox_and_relationship.pkl"
        
        if not person_bbox_file.exists() or not object_bbox_file.exists():
            logger.warning(
                "Action Genome annotation files not found. "
                "Expected person_bbox.pkl and object_bbox_and_relationship.pkl"
            )
            return clips
        
        # Load pickle files (AG-specific format)
        with open(person_bbox_file, 'rb') as f:
            person_data = pickle.load(f)
        
        with open(object_bbox_file, 'rb') as f:
            object_data = pickle.load(f)
        
        # Process each video
        for clip_id in person_data.keys():
            # Find corresponding video file
            video_path = None
            for ext in ['.mp4', '.avi', '.mov']:
                candidate = self.videos_dir / f"{clip_id}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
            
            if video_path is None:
                logger.debug(f"Video file not found for clip '{clip_id}', skipping")
                continue
            
            # Combine annotations
            annotation_data = {
                "objects": object_data.get(clip_id, {}),
                "relationships": object_data.get(clip_id, {}).get("relationships", []),
                "actions": person_data.get(clip_id, {}).get("actions", []),
            }
            
            # Create dataset object
            clips[clip_id] = ActionGenomeDataset(clip_id, video_path, annotation_data)
        
        return clips
    
    def get_clip(self, clip_id: str) -> Optional[ActionGenomeDataset]:
        """Get a specific clip by ID"""
        return self.clips.get(clip_id)
    
    def list_clips(self) -> List[str]:
        """Get list of all available clip IDs"""
        return list(self.clips.keys())
    
    def export_ground_truth(self, clip_id: str, output_path: str):
        """
        Export ground truth for a clip in Orion format
        
        Args:
            clip_id: Clip identifier
            output_path: Path to save JSON
        """
        clip = self.get_clip(clip_id)
        if clip is None:
            raise ValueError(f"Clip '{clip_id}' not found")
        
        orion_format = clip.to_orion_format()
        
        with open(output_path, 'w') as f:
            json.dump(orion_format, f, indent=2)
        
        logger.info(f"Exported ground truth for '{clip_id}' to {output_path}")
    
    def run_evaluation(
        self,
        clip_id: str,
        predicted_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate predicted graph against Action Genome ground truth
        
        Args:
            clip_id: Clip identifier
            predicted_graph: Predicted knowledge graph in Orion format
            
        Returns:
            Evaluation metrics dictionary
        """
        # Import here to avoid circular dependency
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from src.orion.evaluation.metrics import compare_graphs
        
        clip = self.get_clip(clip_id)
        if clip is None:
            raise ValueError(f"Clip '{clip_id}' not found")
        
        ground_truth = clip.to_orion_format()
        metrics = compare_graphs(predicted_graph, ground_truth)
        
        return metrics.to_dict()
    
    def batch_evaluate(
        self,
        predictions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate predictions for multiple clips
        
        Args:
            predictions: Dict mapping clip_id -> predicted_graph
            
        Returns:
            Aggregated metrics across all clips
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from src.orion.evaluation.metrics import compare_graphs
        import numpy as np
        
        all_metrics = []
        
        for clip_id, predicted_graph in predictions.items():
            clip = self.get_clip(clip_id)
            if clip is None:
                logger.warning(f"Clip '{clip_id}' not found, skipping")
                continue
            
            ground_truth = clip.to_orion_format()
            metrics = compare_graphs(predicted_graph, ground_truth)
            all_metrics.append(metrics)
        
        if not all_metrics:
            return {}
        
        # Aggregate metrics
        aggregated = {
            "num_clips": len(all_metrics),
            "edge_precision": np.mean([m.edge_precision for m in all_metrics]),
            "edge_recall": np.mean([m.edge_recall for m in all_metrics]),
            "edge_f1": np.mean([m.edge_f1 for m in all_metrics]),
            "event_precision": np.mean([m.event_precision for m in all_metrics]),
            "event_recall": np.mean([m.event_recall for m in all_metrics]),
            "event_f1": np.mean([m.event_f1 for m in all_metrics]),
            "causal_precision": np.mean([m.causal_precision for m in all_metrics]),
            "causal_recall": np.mean([m.causal_recall for m in all_metrics]),
            "causal_f1": np.mean([m.causal_f1 for m in all_metrics]),
            
            # Standard deviations for statistical analysis
            "edge_f1_std": np.std([m.edge_f1 for m in all_metrics]),
            "event_f1_std": np.std([m.event_f1 for m in all_metrics]),
            "causal_f1_std": np.std([m.causal_f1 for m in all_metrics]),
        }
        
        logger.info(
            f"Action Genome evaluation complete: "
            f"{aggregated['num_clips']} clips, "
            f"avg edge F1={aggregated['edge_f1']:.3f}, "
            f"avg causal F1={aggregated['causal_f1']:.3f}"
        )
        
        return aggregated
