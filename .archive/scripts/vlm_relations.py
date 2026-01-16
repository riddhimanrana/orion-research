#!/usr/bin/env python3
"""
VLM-based semantic relation classification for Orion SGG.
Uses MLX-VLM (FastVLM) to classify relations from object bounding boxes.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2

try:
    from mlx_vlm import load, process_images
    HAS_VLM = True
except ImportError:
    print("Warning: mlx_vlm not available, VLM relations disabled")
    HAS_VLM = False

# Relation classification prompts
RELATION_PROMPT = """Given two objects in an image with bounding boxes, what is the spatial/semantic relation between them?

Subject (box 1): {subject}
Object (box 2): {object}

Is the subject:
- holding the object? 
- looking at the object?
- standing on the object?
- sitting on the object?
- next to the object?
- above the object?
- below the object?

Answer with just the relation name, or "none" if no clear relation.
"""

class VLMRelationClassifier:
    """Classify relations between objects using VLM."""
    
    def __init__(self, model_name: str = "mlx-community/FastVLM-3b"):
        """Initialize VLM for relation classification."""
        if not HAS_VLM:
            self.model = None
            self.processor = None
            return
        
        try:
            self.model, self.processor = load(model_name)
            print(f"✓ Loaded VLM: {model_name}")
        except Exception as e:
            print(f"⚠️  Failed to load VLM: {e}")
            self.model = None
            self.processor = None
    
    def classify_relation(self, image: np.ndarray, subject_bbox: Tuple, 
                         object_bbox: Tuple, subject_class: str, 
                         object_class: str) -> Optional[Tuple[str, float]]:
        """
        Classify relation between two objects.
        Returns (relation_type, confidence)
        """
        if self.model is None:
            return None
        
        try:
            # Extract object crops
            x1, y1, x2, y2 = map(int, subject_bbox)
            subject_crop = image[y1:y2, x1:x2]
            
            x1, y1, x2, y2 = map(int, object_bbox)
            object_crop = image[y1:y2, x1:x2]
            
            if subject_crop.size == 0 or object_crop.size == 0:
                return None
            
            # Create visualization with both bboxes
            h, w = image.shape[:2]
            vis_image = image.copy()
            x1, y1, x2, y2 = map(int, subject_bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, subject_class, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            x1, y1, x2, y2 = map(int, object_bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis_image, object_class, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Process with VLM
            prompt = RELATION_PROMPT.format(subject=subject_class, object=object_class)
            
            # This would require proper MLX-VLM integration
            # For now, return None as placeholder
            return None
            
        except Exception as e:
            return None


def add_vlm_relations(video_id: str, vlm: Optional[VLMRelationClassifier] = None) -> int:
    """
    Add VLM-classified relations to a video's scene graph.
    Returns number of relations added.
    """
    if vlm is None or vlm.model is None:
        return 0
    
    results_dir = Path("results")
    sg_file = results_dir / video_id / "scene_graph.jsonl"
    video_file = f"datasets/PVSG/VidOR/mnt/lustre/jkyang/CVPR23/openpvsg/data/vidor/videos/{video_id}.mp4"
    
    if not sg_file.exists() or not Path(video_file).exists():
        return 0
    
    # Load video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return 0
    
    # Load memory for object info
    mem_file = results_dir / video_id / "memory.json"
    with open(mem_file) as f:
        mem = json.load(f)
    
    # Process each frame and add VLM relations
    added = 0
    frame_idx = 0
    
    with open(sg_file) as f:
        scene_graphs = [json.loads(line) for line in f]
    
    cap.release()
    
    # For now, just return 0 as placeholder
    # Full implementation would:
    # 1. Load video frame by frame
    # 2. For each frame's objects, call VLM to classify pairwise relations
    # 3. Add high-confidence relations to scene_graph.jsonl
    
    return added


def main():
    """Test VLM relation classification on first few videos."""
    if not HAS_VLM:
        print("MLX-VLM not available. Install with:")
        print("pip install mlx-vlm")
        return
    
    vlm = VLMRelationClassifier()
    
    if vlm.model is None:
        print("Failed to load VLM model")
        return
    
    videos = [
        "0001_4164158586", "0003_3396832512", "0003_6141007489",
        "0004_11566980553", "0005_2505076295"
    ]
    
    print(f"\nAdding VLM-classified relations to {len(videos)} videos...\n")
    
    for vid in videos:
        print(f"  {vid}... ", end="", flush=True)
        added = add_vlm_relations(vid, vlm)
        print(f"Added {added} relations")
    
    print("\n✓ VLM relations added")


if __name__ == '__main__':
    main()
