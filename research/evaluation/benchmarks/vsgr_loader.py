"""
VSGR (Video Scene Graph Recognition) Benchmark Loader
======================================================

This module provides utilities for loading and evaluating on the VSGR
benchmark dataset, which includes video clips with annotated scene graphs
containing objects, relationships, and temporal interactions.

VSGR Dataset Structure (expected):
```
vsgr_dataset/
├── videos/
│   ├── clip_001.mp4
│   ├── clip_002.mp4
│   └── ...
├── annotations/
│   ├── clip_001.json
│   ├── clip_002.json
│   └── ...
└── metadata.json
```

Each annotation JSON contains:
- entities: List of objects with IDs and labels
- relationships: Spatial/semantic relationships between entities
- temporal_events: Events that occur over time
- frames: Per-frame bounding boxes and attributes

Author: Orion Research Team
Date: October 2025
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("VSGRBenchmark")


@dataclass
class VSGREntity:
    """Entity in VSGR ground truth"""
    entity_id: str
    class_label: str
    attributes: Dict[str, Any]
    first_frame: int
    last_frame: int


@dataclass
class VSGRRelationship:
    """Relationship in VSGR ground truth"""
    subject: str  # entity_id
    object: str   # entity_id
    predicate: str
    frame_start: int
    frame_end: int


@dataclass
class VSGREvent:
    """Temporal event in VSGR ground truth"""
    event_id: str
    event_type: str
    participants: List[str]  # entity_ids
    timestamp: float
    description: str


class VSGRDataset:
    """
    Represents a single video clip from VSGR with its ground truth annotations
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
            annotation_data: Parsed annotation JSON
        """
        self.clip_id = clip_id
        self.video_path = video_path
        self.raw_annotations = annotation_data
        
        # Parse annotations
        self.entities = self._parse_entities(annotation_data.get("entities", []))
        self.relationships = self._parse_relationships(
            annotation_data.get("relationships", [])
        )
        self.events = self._parse_events(annotation_data.get("temporal_events", []))
        self.frames = annotation_data.get("frames", {})
        
        logger.debug(
            f"Loaded VSGR clip '{clip_id}': "
            f"{len(self.entities)} entities, "
            f"{len(self.relationships)} relationships, "
            f"{len(self.events)} events"
        )
    
    def _parse_entities(self, entities_data: List[Dict]) -> List[VSGREntity]:
        """Parse entity annotations"""
        entities = []
        for e in entities_data:
            entity = VSGREntity(
                entity_id=e.get("id", "unknown"),
                class_label=e.get("class", "object"),
                attributes=e.get("attributes", {}),
                first_frame=e.get("first_frame", 0),
                last_frame=e.get("last_frame", 9999),
            )
            entities.append(entity)
        return entities
    
    def _parse_relationships(
        self,
        relationships_data: List[Dict]
    ) -> List[VSGRRelationship]:
        """Parse relationship annotations"""
        relationships = []
        for r in relationships_data:
            rel = VSGRRelationship(
                subject=r.get("subject", ""),
                object=r.get("object", ""),
                predicate=r.get("predicate", "related_to"),
                frame_start=r.get("frame_start", 0),
                frame_end=r.get("frame_end", 9999),
            )
            relationships.append(rel)
        return relationships
    
    def _parse_events(self, events_data: List[Dict]) -> List[VSGREvent]:
        """Parse temporal event annotations"""
        events = []
        for e in events_data:
            event = VSGREvent(
                event_id=e.get("id", "unknown"),
                event_type=e.get("type", "interaction"),
                participants=e.get("participants", []),
                timestamp=e.get("timestamp", 0.0),
                description=e.get("description", ""),
            )
            events.append(event)
        return events
    
    def to_orion_format(self) -> Dict[str, Any]:
        """
        Convert VSGR ground truth to Orion's graph format for comparison
        
        Returns:
            Dictionary with entities, relationships, events in Orion format
        """
        orion_entities = []
        for entity in self.entities:
            orion_entities.append({
                "entity_id": entity.entity_id,
                "class": entity.class_label,
                "description": entity.class_label,
                "first_seen": entity.first_frame / 30.0,  # Assume 30 FPS
                "last_seen": entity.last_frame / 30.0,
                "attributes": entity.attributes,
            })
        
        orion_relationships = []
        for rel in self.relationships:
            orion_relationships.append({
                "source": rel.subject,
                "target": rel.object,
                "type": rel.predicate,
                "frame_start": rel.frame_start,
                "frame_end": rel.frame_end,
            })
        
        orion_events = []
        for event in self.events:
            # Map to causal event format if possible
            if len(event.participants) >= 2:
                orion_events.append({
                    "type": event.event_type,
                    "agent": event.participants[0],
                    "patient": event.participants[1] if len(event.participants) > 1 else None,
                    "relationship": event.event_type,
                    "timestamp": event.timestamp,
                    "description": event.description,
                })
        
        return {
            "entities": orion_entities,
            "relationships": orion_relationships,
            "events": orion_events,
            "metadata": {
                "clip_id": self.clip_id,
                "source": "vsgr_ground_truth",
            }
        }


class VSGRBenchmark:
    """
    Loader and evaluator for the VSGR benchmark dataset
    """
    
    def __init__(self, dataset_root: str):
        """
        Args:
            dataset_root: Path to VSGR dataset root directory
        """
        self.root = Path(dataset_root)
        self.videos_dir = self.root / "videos"
        self.annotations_dir = self.root / "annotations"
        self.metadata_path = self.root / "metadata.json"
        
        # Validate structure
        if not self.root.exists():
            raise FileNotFoundError(f"VSGR dataset not found at {dataset_root}")
        
        if not self.videos_dir.exists():
            logger.warning(f"Videos directory not found: {self.videos_dir}")
        
        if not self.annotations_dir.exists():
            raise FileNotFoundError(
                f"Annotations directory not found: {self.annotations_dir}"
            )
        
        # Load metadata
        self.metadata = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Discover clips
        self.clips = self._discover_clips()
        
        logger.info(
            f"Loaded VSGR benchmark from {dataset_root} "
            f"with {len(self.clips)} clips"
        )
    
    def _discover_clips(self) -> Dict[str, VSGRDataset]:
        """Discover and load all clips in the dataset"""
        clips = {}
        
        # Find all annotation files
        annotation_files = list(self.annotations_dir.glob("*.json"))
        
        for ann_file in annotation_files:
            clip_id = ann_file.stem
            
            # Find corresponding video
            video_path = None
            for ext in ['.mp4', '.avi', '.mov']:
                candidate = self.videos_dir / f"{clip_id}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
            
            if video_path is None:
                logger.warning(
                    f"Video file not found for clip '{clip_id}', skipping"
                )
                continue
            
            # Load annotation
            with open(ann_file, 'r') as f:
                annotation_data = json.load(f)
            
            # Create dataset object
            clips[clip_id] = VSGRDataset(clip_id, video_path, annotation_data)
        
        return clips
    
    def get_clip(self, clip_id: str) -> Optional[VSGRDataset]:
        """
        Get a specific clip by ID
        
        Args:
            clip_id: Clip identifier
            
        Returns:
            VSGRDataset or None if not found
        """
        return self.clips.get(clip_id)
    
    def list_clips(self) -> List[str]:
        """Get list of all available clip IDs"""
        return list(self.clips.keys())
    
    def export_ground_truth(
        self,
        clip_id: str,
        output_path: str
    ):
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
        Evaluate predicted graph against VSGR ground truth
        
        Args:
            clip_id: Clip identifier
            predicted_graph: Predicted knowledge graph in Orion format
            
        Returns:
            Evaluation metrics dictionary
        """
        from ..metrics import compare_graphs
        
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
        from ..metrics import compare_graphs
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
        }
        
        logger.info(
            f"Batch evaluation complete: "
            f"{aggregated['num_clips']} clips, "
            f"avg edge F1={aggregated['edge_f1']:.3f}, "
            f"avg causal F1={aggregated['causal_f1']:.3f}"
        )
        
        return aggregated
