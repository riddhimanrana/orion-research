"""
PVSG (Panoptic Video Scene Graph) Benchmark Loader
===================================================

Loader for the PVSG dataset - a comprehensive benchmark for panoptic
video scene graph generation with dense frame-level annotations.

Dataset: PVSG (NeurIPS 2023)
Paper: "Panoptic Scene Graph Generation from Monocular Videos"

Expected Directory Structure:
```
pvsg/
├── videos/
│   ├── video_001.mp4
│   └── ...
├── annotations/
│   ├── train/
│   │   ├── video_001.json
│   │   └── ...
│   ├── val/
│   └── test/
└── metadata/
    ├── object_classes.json
    └── predicate_classes.json
```

Author: Orion Research Team
Date: October 2025
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("PVSGBenchmark")


@dataclass
class PVSGObject:
    """Object annotation in PVSG"""
    object_id: str
    class_label: str
    mask: Optional[List] = None  # Segmentation mask
    bbox: Optional[List[int]] = None
    frame_id: int = 0


@dataclass
class PVSGRelationship:
    """Relationship in PVSG"""
    subject_id: str
    object_id: str
    predicate: str
    start_frame: int
    end_frame: int
    temporal_type: str = "during"  # during, before, after


class PVSGDataset:
    """Single video from PVSG with annotations"""
    
    def __init__(self, video_id: str, video_path: Path, annotation_data: Dict):
        self.video_id = video_id
        self.video_path = video_path
        self.raw_annotations = annotation_data
        
        self.objects = self._parse_objects(annotation_data.get("objects", []))
        self.relationships = self._parse_relationships(
            annotation_data.get("relationships", [])
        )
    
    def _parse_objects(self, objects_data: List[Dict]) -> List[PVSGObject]:
        """Parse object annotations"""
        objects = []
        for obj_data in objects_data:
            obj = PVSGObject(
                object_id=obj_data.get("id", "unknown"),
                class_label=obj_data.get("category", "object"),
                bbox=obj_data.get("bbox"),
                frame_id=obj_data.get("frame_id", 0)
            )
            objects.append(obj)
        return objects
    
    def _parse_relationships(self, rels_data: List[Dict]) -> List[PVSGRelationship]:
        """Parse relationship annotations"""
        relationships = []
        for rel_data in rels_data:
            rel = PVSGRelationship(
                subject_id=rel_data.get("subject_id", ""),
                object_id=rel_data.get("object_id", ""),
                predicate=rel_data.get("predicate", "related"),
                start_frame=rel_data.get("begin_fid", 0),
                end_frame=rel_data.get("end_fid", 0),
                temporal_type=rel_data.get("temporal_type", "during")
            )
            relationships.append(rel)
        return relationships
    
    def to_orion_format(self) -> Dict[str, Any]:
        """Convert PVSG annotations to Orion format"""
        # Extract unique entities
        entity_map = {}
        for obj in self.objects:
            if obj.object_id not in entity_map:
                entity_map[obj.object_id] = {
                    "entity_id": obj.object_id,
                    "class": obj.class_label,
                    "description": obj.class_label,
                    "first_seen": obj.frame_id / 30.0,
                    "last_seen": obj.frame_id / 30.0,
                }
            else:
                entity_map[obj.object_id]["last_seen"] = obj.frame_id / 30.0
        
        # Convert relationships
        orion_relationships = []
        for rel in self.relationships:
            orion_relationships.append({
                "source": rel.subject_id,
                "target": rel.object_id,
                "type": rel.predicate,
                "start_frame": rel.start_frame,
                "end_frame": rel.end_frame,
                "temporal_type": rel.temporal_type,
            })
        
        return {
            "entities": list(entity_map.values()),
            "relationships": orion_relationships,
            "events": [],  # PVSG focuses on relationships, not explicit events
            "metadata": {
                "video_id": self.video_id,
                "source": "pvsg_ground_truth",
            }
        }


class PVSGBenchmark:
    """Loader for PVSG benchmark"""
    
    def __init__(self, dataset_root: str, split: str = "test"):
        self.root = Path(dataset_root)
        self.split = split
        self.videos_dir = self.root / "videos"
        self.annotations_dir = self.root / "annotations" / split
        
        if not self.root.exists():
            raise FileNotFoundError(f"PVSG dataset not found at {dataset_root}")
        
        self.clips = self._discover_clips()
        logger.info(f"Loaded PVSG {split} split with {len(self.clips)} videos")
    
    def _discover_clips(self) -> Dict[str, PVSGDataset]:
        """Discover all videos in the split"""
        clips = {}
        
        annotation_files = list(self.annotations_dir.glob("*.json"))
        
        for ann_file in annotation_files:
            video_id = ann_file.stem
            
            # Find video file
            video_path = None
            for ext in ['.mp4', '.avi']:
                candidate = self.videos_dir / f"{video_id}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
            
            if not video_path:
                logger.debug(f"Video not found for {video_id}")
                continue
            
            # Load annotation
            with open(ann_file, 'r') as f:
                annotation_data = json.load(f)
            
            clips[video_id] = PVSGDataset(video_id, video_path, annotation_data)
        
        return clips
    
    def get_clip(self, video_id: str) -> Optional[PVSGDataset]:
        return self.clips.get(video_id)
    
    def list_clips(self) -> List[str]:
        return list(self.clips.keys())
