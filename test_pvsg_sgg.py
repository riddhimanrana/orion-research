#!/usr/bin/env python3
"""
Test PVSG Scene Graph Generation with Improved Specificity
"""

import json
import logging
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import asdict

import numpy as np

# Import the improved classifier
try:
    from orion.perception.pvsg_action_classifier import PVSGActionClassifier, Detection, HandDetection
except ImportError:
    # Fallback/Mock for local testing if orion not installed as package
    print("WARNING: Could not import PVSGActionClassifier from orion.perception")
    from typing import dataclass
    
    @dataclass
    class Detection:
        track_id: int
        class_name: str
        bbox: List[float]
        frame_id: int
        confidence: float = 1.0

    @dataclass
    class HandDetection:
        bbox: List[float]
        handedness: str = "unknown"

    class PVSGActionClassifier:
        def predict_relations(self, *args, **kwargs):
            return []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PVSG Relation Vocabulary (index 0 to 56)
PVSG_PREDICATES = [
    "beside", "biting", "blowing", "brushing", "caressing", "carrying", "catching", "chasing",
    "cleaning", "closing", "cooking", "cutting", "drinking from", "eating", "entering", "feeding",
    "grabbing", "guiding", "hanging from", "hitting", "holding", "hugging", "in", "in front of",
    "jumping from", "jumping over", "kicking", "kissing", "licking", "lighting", "looking at",
    "lying on", "next to", "on", "opening", "over", "picking", "playing", "playing with",
    "pointing to", "pulling", "pushing", "riding", "running on", "shaking hand with", "sitting on",
    "standing on", "stepping on", "stirring", "swinging", "talking to", "throwing", "touching",
    "toward", "walking on", "watering", "wearing"
]

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def estimate_hands_from_person(person_bbox: List[float]) -> List[HandDetection]:
    """
    Estimate hand positions from person bbox.
    """
    x1, y1, x2, y2 = person_bbox
    width = x2 - x1
    height = y2 - y1
    
    # Estimate left and right hand positions
    # Hands are typically at ~50-70% height depending on pose
    hand_y = y1 + height * 0.6
    hand_size = width * 0.2
    
    left_hand = HandDetection(
        bbox=[
            x1,
            hand_y - hand_size/2,
            x1 + hand_size,
            hand_y + hand_size/2
        ],
        handedness="left"
    )
    
    right_hand = HandDetection(
        bbox=[
            x2 - hand_size,
            hand_y - hand_size/2,
            x2,
            hand_y + hand_size/2
        ],
        handedness="right"
    )
    
    return [left_hand, right_hand]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True, help="Path to tracks.jsonl")
    parser.add_argument("--out", default="pvsg_relations.jsonl", help="Output path")
    parser.add_argument("--video_id", default=None, help="Explicit video ID. If not provided, inferred from path.")
    args = parser.parse_args()
    
    tracks_path = Path(args.tracks)
    if not tracks_path.exists():
        print(f"Error: Tracks file not found: {tracks_path}")
        return

    # Infer Video ID
    video_id = args.video_id
    if not video_id:
        # Try parent directory name
        # e.g. results/pvsg_batch_1_100/0001_4164158586/tracks.jsonl -> 0001_4164158586
        parent_name = tracks_path.parent.name
        # Simple heuristic: check if it looks like a PVSG ID (digits_digits)
        if "_" in parent_name and any(c.isdigit() for c in parent_name):
            video_id = parent_name
        else:
            video_id = "test_video"
            print(f"Warning: Could not infer video_id from path {tracks_path}, using '{video_id}'")
    
    print(f"Processing tracks for video_id: {video_id}")
    
    tracks = load_jsonl(tracks_path)
    
    # Group by frame
    by_frame = {}
    for t in tracks:
        by_frame.setdefault(t["frame_id"], []).append(t)
    
    # Initialize Classifier
    classifier = PVSGActionClassifier(
        holding_hand_dist=150.0,
        holding_min_frames=1,
        throwing_velocity_thresh=20.0,
        picking_hand_dist=150.0
    )
    
    results = []
    
    print(f"Processing {len(by_frame)} frames...")
    
    for fid, frame_tracks in sorted(by_frame.items()):
        # Convert tracks to Detections
        detections = []
        hand_detections = {}
        
        for t in frame_tracks:
            # Handle potential missing confidence
            conf = t.get("confidence", 1.0)
            
            # Create Detection
            det = Detection(
                track_id=t.get("track_id", t.get("id")), # Fallback to 'id' if 'track_id' missing
                class_name=t.get("category", t.get("class_name", t.get("label", "object"))),
                bbox=t["bbox"],
                frame_id=fid,
                confidence=conf
            )
            detections.append(det)
            
            # Generate estimated hands for people
            if det.class_name in ['person', 'adult', 'child', 'man', 'woman']:
                hands = estimate_hands_from_person(det.bbox)
                hand_detections[det.track_id] = hands
        
        # Predict
        relations = classifier.predict_relations(detections, hand_detections, fid)
        
        # Format results as Scene Graph (nodes + edges)
        nodes = []
        edges = []
        
        # Add all detections as nodes
        for det in detections:
            nodes.append({
                "memory_id": str(det.track_id),
                "class": det.class_name,
                "bbox": det.bbox,
                "confidence": det.confidence
            })
            
        # Add relations as edges
        for subj_id, obj_id, pred, conf in relations:
            pvsg_pred = pred.replace("_", " ")
            
            # Map predicates
            if pvsg_pred not in PVSG_PREDICATES:
                 if pvsg_pred == "sitting at": pvsg_pred = "sitting on"
                 elif pred == "near": pvsg_pred = "next to"
                 else: continue
            
            edges.append({
                "subject": str(subj_id),
                "object": str(obj_id),
                "relation": pvsg_pred,
                "confidence": conf
            })
            
        results.append({
            "video_id": video_id,
            "frame_id": fid,
            "nodes": nodes,
            "edges": edges
        })
            
    print(f"Generated graphs for {len(results)} frames.")
        
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
