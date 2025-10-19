"""
ASPIRe (Action Segmentation for Procedural Recognition) Benchmark Loader
=========================================================================

Loader for ASPIRe dataset focusing on egocentric procedural tasks
like cooking, assembly, and multi-step activities.

Author: Orion Research Team
Date: October 2025
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ASPIReBenchmark")


@dataclass
class ASPIReStep:
    """Procedural step annotation"""
    step_id: str
    step_name: str
    start_frame: int
    end_frame: int
    objects_involved: List[str]
    preconditions: List[str]
    effects: List[str]


class ASPIReDataset:
    """Single procedural task video"""
    
    def __init__(self, video_id: str, video_path: Path, annotation_data: Dict):
        self.video_id = video_id
        self.video_path = video_path
        self.raw_annotations = annotation_data
        
        self.steps = self._parse_steps(annotation_data.get("steps", []))
        self.task_type = annotation_data.get("task_type", "unknown")
    
    def _parse_steps(self, steps_data: List[Dict]) -> List[ASPIReStep]:
        """Parse procedural steps"""
        steps = []
        for step_data in steps_data:
            step = ASPIReStep(
                step_id=step_data.get("id", ""),
                step_name=step_data.get("name", ""),
                start_frame=step_data.get("start_frame", 0),
                end_frame=step_data.get("end_frame", 0),
                objects_involved=step_data.get("objects", []),
                preconditions=step_data.get("preconditions", []),
                effects=step_data.get("effects", [])
            )
            steps.append(step)
        return steps
    
    def to_orion_format(self) -> Dict[str, Any]:
        """Convert ASPIRe to Orion format"""
        # Extract entities from objects involved in steps
        entities = {}
        events = []
        
        for step in self.steps:
            # Create event for each step
            event = {
                "type": "procedural_step",
                "relationship": step.step_name,
                "timestamp": (step.start_frame + step.end_frame) / 2 / 30.0,
                "start_time": step.start_frame / 30.0,
                "end_time": step.end_frame / 30.0,
                "preconditions": step.preconditions,
                "effects": step.effects,
            }
            
            # Add participating objects
            if step.objects_involved:
                event["agent"] = step.objects_involved[0] if step.objects_involved else None
                event["patient"] = step.objects_involved[1] if len(step.objects_involved) > 1 else None
            
            events.append(event)
            
            # Track entities
            for obj_id in step.objects_involved:
                if obj_id not in entities:
                    entities[obj_id] = {
                        "entity_id": obj_id,
                        "class": "object",
                        "description": obj_id,
                    }
        
        return {
            "entities": list(entities.values()),
            "relationships": [],  # ASPIRe focuses on steps, not static relationships
            "events": events,
            "metadata": {
                "video_id": self.video_id,
                "task_type": self.task_type,
                "source": "aspire_ground_truth",
            }
        }


class ASPIReBenchmark:
    """Loader for ASPIRe benchmark"""
    
    def __init__(self, dataset_root: str):
        self.root = Path(dataset_root)
        self.videos_dir = self.root / "videos"
        self.annotations_dir = self.root / "annotations"
        
        if not self.root.exists():
            raise FileNotFoundError(f"ASPIRe dataset not found at {dataset_root}")
        
        self.clips = self._discover_clips()
        logger.info(f"Loaded ASPIRe with {len(self.clips)} task videos")
    
    def _discover_clips(self) -> Dict[str, ASPIReDataset]:
        """Discover all task videos"""
        clips = {}
        
        annotation_files = list(self.annotations_dir.glob("*.json"))
        
        for ann_file in annotation_files:
            video_id = ann_file.stem
            
            video_path = None
            for ext in ['.mp4', '.avi']:
                candidate = self.videos_dir / f"{video_id}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
            
            if not video_path:
                continue
            
            with open(ann_file, 'r') as f:
                annotation_data = json.load(f)
            
            clips[video_id] = ASPIReDataset(video_id, video_path, annotation_data)
        
        return clips
    
    def get_clip(self, video_id: str) -> Optional[ASPIReDataset]:
        return self.clips.get(video_id)
    
    def list_clips(self) -> List[str]:
        return list(self.clips.keys())
