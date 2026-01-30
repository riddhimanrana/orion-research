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

# Import DINOv3 and V-JEPA v2 backends (optional)
try:
    from orion.backends.dino_backend import DINOEmbedder
    DINO_AVAILABLE = True
except ImportError:
    logger.warning("DINOv3 backend not available - semantic similarity will use fallback")
    DINO_AVAILABLE = False
    DINOEmbedder = None

try:
    from orion.backends.vjepa2_backend import VJepa2Embedder
    VJEPA_AVAILABLE = True
except ImportError:
    logger.warning("V-JEPA v2 backend not available - temporal action recognition will use heuristics")
    VJEPA_AVAILABLE = False
    VJepa2Embedder = None

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
    
    # Initialize Deep Learning Backends (Optional)
    dino_embedder = None
    vjepa_embedder = None
    
    if DINO_AVAILABLE:
        try:
            logger.info("Initializing DINOv3 embedder for semantic similarity...")
            dino_embedder = DINOEmbedder(
                model_name="facebook/dinov2-base",
                device="cpu"  # Use GPU if available: "cuda"
            )
            logger.info("✓ DINOv3 embedder loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load DINOv3: {e}")
            dino_embedder = None
    
    if VJEPA_AVAILABLE:
        try:
            logger.info("Initializing V-JEPA v2 embedder for temporal actions...")
            vjepa_embedder = VJepa2Embedder(
                model_name="facebook/vjepa2-vitl-fpc64-256",
                device="cpu",  # Use GPU if available: "cuda"
                dtype=torch.float32  # Use float16 for GPU
            )
            logger.info("✓ V-JEPA v2 embedder loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load V-JEPA v2: {e}")
            vjepa_embedder = None
    
    # Initialize Classifiers
    classifier = PVSGActionClassifier(
        holding_hand_dist=150.0,
        holding_min_frames=3,      # Increase min frames to 3 to reduce noise
        throwing_velocity_thresh=60.0,  # Increase from 20 to 60 (more realistic)
        throwing_accel_thresh=30.0,
        picking_hand_dist=120.0,
        vjepa_embedder=vjepa_embedder
    )
    
    # Initialize CIS Scorer (Orion)
    try:
        from orion.analysis.cis_scorer import CausalInfluenceScorer
        # Disable depth gating since we are in 2D mode for now
        cis_scorer = CausalInfluenceScorer(
            enable_depth_gating=False,
            cis_threshold=0.1,  # Minor threshold to filter extreme junk
            dino_embedder=dino_embedder,
            weight_motion=0.15,  # Slightly lower motion weight
            weight_semantic=0.15  # Slightly higher semantic weight
        )
        has_cis = True
        print("CIS Scorer initialized successfully.")
    except ImportError:
        print("WARNING: Could not import CausalInfluenceScorer. CIS scores will be 0.0.")
        cis_scorer = None
        has_cis = False
    
    results = []
    
    print(f"Processing {len(by_frame)} frames...")
    
    # Track velocities for CIS (using PVSG classifier's tracker)
    # Process frames strictly in order
    sorted_frames = sorted(by_frame.items())
    
    for fid, frame_tracks in sorted_frames:
        # Convert tracks to Detections
        detections = []
        hand_detections = {}
        
        # Map for CIS entity lookup
        det_map = {}
        
        for t in frame_tracks:
            # Handle potential missing confidence
            conf = t.get("confidence", 1.0)
            track_id = t.get("track_id", t.get("id"))
            
            # Create Detection
            det = Detection(
                track_id=track_id,
                class_name=t.get("category", t.get("class_name", t.get("label", "object"))),
                bbox=t["bbox"],
                frame_id=fid,
                confidence=conf
            )
            
            # Pre-compute velocity using PVSGClassifier's history logic
            # (We need to update history first in the loop below, but let's do it consistent with flow)
            detections.append(det)
            det_map[track_id] = det
            
            # Generate estimated hands for people
            if det.class_name in ['person', 'adult', 'child', 'man', 'woman', 'boy', 'girl', 'stuffed animal']:
                hands = estimate_hands_from_person(det.bbox)
                hand_detections[det.track_id] = hands
        
        # 1. Predict PVSG Relations
        # This returns candidates. Use a larger top-k internally in classifier or just take all.
        # Ideally we'd modify classifier to return more, but for now we take the top-5 it gives.
        relations = classifier.predict_relations(detections, hand_detections, fid)
        
        # 2. Enrich with attributes needed for CIS (Velocity)
        if has_cis:
            for det in detections:
                vel = classifier.compute_velocity(det.track_id, fid)
                if vel:
                    det.velocity = [vel[0], vel[1]]
                else:
                    det.velocity = [0.0, 0.0]
                det.object_class = det.class_name
        
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
            
        # 3. RE-RANKING LOGIC: Combine Action Score + CIS Score
        scored_edges = []
        
        for subj_id, obj_id, pred, action_score in relations:
            pvsg_pred = pred.replace("_", " ")
            
            # Map predicates
            if pvsg_pred not in PVSG_PREDICATES:
                 if pvsg_pred == "sitting at": pvsg_pred = "sitting on"
                 elif pred == "near": pvsg_pred = "next to"
                 elif pred == "touching": pvsg_pred = "touching"
                 elif pred == "eating": pvsg_pred = "eating"
                 elif pred == "swinging": pvsg_pred = "swinging"
                 else: continue
            
            # Calculate CIS Score
            cis_score = 0.0
            cis_info = {}
            if has_cis:
                subj = det_map.get(subj_id)
                obj = det_map.get(obj_id)
                if subj and obj:
                    score, components = cis_scorer.calculate_cis(
                        agent_entity=subj,
                        patient_entity=obj,
                        time_delta=0.033
                    )
                    cis_score = score
                    cis_info = {
                        "temporal": components.temporal,
                        "spatial": components.spatial,
                        "motion": components.motion,
                        "semantic": components.semantic
                    }
            
            # Final Score Fusion
            # Action Score (0.8-0.9 usually) + CIS Score (0.0-1.0)
            # We enforce CIS as a validator for physical interaction predicates
            if pred in ['picking', 'holding', 'touching', 'eating', 'swinging']:
                # Physcial relations need decent CIS
                # Boost if CIS is high, penalize if low
                final_score = action_score * (0.5 + 0.5 * cis_score)
            else:
                # Spatial predicates (on, next to) rely less on motion/interaction CIS (except spatial component)
                final_score = action_score
                
            scored_edges.append({
                "subject": str(subj_id),
                "object": str(obj_id),
                "relation": pvsg_pred,
                "confidence": round(final_score, 3), # Use fused score
                "raw_action_score": action_score,
                "cis_score": round(cis_score, 3),
                "cis_components": cis_info
            })
            
        # Filter and optimize edges generally
        
        # 4. SPECIFICITY FILTERING (Deduplication)
        # If a subject has specific spatial relation (grass, carpet), remove generic (ground, floor)
        # Group by subject
        subj_spatial = {} # subj_id -> list of (obj_name, index)
        for i, e in enumerate(scored_edges):
            if e['relation'] in ['walking on', 'standing on', 'on']:
                # Find obj class
                obj_id = e['object']
                obj_det = det_map.get(int(obj_id))
                if obj_det:
                    subj = e['subject']
                    if subj not in subj_spatial: subj_spatial[subj] = []
                    subj_spatial[subj].append((obj_det.class_name, i))
        
        indices_to_remove = set()
        for subj, items in subj_spatial.items():
            classes = [x[0] for x in items]
            has_specific = any(c in ['grass', 'carpet', 'mat', 'road', 'sidewalk', 'field'] for c in classes)
            if has_specific:
                # Remove generic 'ground', 'floor'
                for cls_name, idx in items:
                    if cls_name in ['ground', 'floor', 'earth']:
                        indices_to_remove.add(idx)
                        
        final_edges = []
        for i, e in enumerate(scored_edges):
            if i in indices_to_remove:
                continue
            if e["confidence"] > 0.1:
                final_edges.append(e)
        
        # Add to results
        edges.extend(final_edges)
            
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
