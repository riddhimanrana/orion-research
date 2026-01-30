#!/usr/bin/env python3
"""
Test PVSG Action-Aware Scene Graph Generation (Simplified)

Runs action-aware relation classification WITHOUT hand detection:
1. Uses person-object proximity as proxy for hand-object interaction
2. Temporal motion analysis for "throwing"
3. Object appearance for "picking"
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from orion.perception.pvsg_action_classifier import PVSGActionClassifier, Detection, HandDetection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tracks(tracks_path: str) -> Dict:
    """Load tracks from JSONL file."""
    tracks_by_frame = defaultdict(list)
    
    with open(tracks_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            track = json.loads(line)
            frame_id = track['frame_id']
            tracks_by_frame[frame_id].append(track)
    
    return tracks_by_frame


def estimate_hands_from_person(person_bbox: List[float]) -> List[HandDetection]:
    """
    Estimate hand positions from person bbox.
    
    Assumes hands are at the sides/bottom of person bbox.
    This is a rough approximation when hand detection is unavailable.
    """
    x1, y1, x2, y2 = person_bbox
    width = x2 - x1
    height = y2 - y1
    
    # Estimate left and right hand positions
    # Hands are typically at ~70-80% height, sides of bbox
    hand_y = y1 + height * 0.7
    hand_size = width * 0.15
    
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


def generate_action_scene_graph(
    episode_dir: str,
    output_path: str
):
    """
    Generate PVSG action-aware scene graph using bbox proximity.
    
    Args:
        episode_dir: Path to episode directory (e.g., results/pvsg_0003)
        output_path: Output path for scene graph JSONL
    """
    episode_path = Path(episode_dir)
    
    # Load tracks
    logger.info("Loading tracks...")
    tracks_path = episode_path / "tracks.jsonl"
    tracks_by_frame = load_tracks(str(tracks_path))
    
    # Initialize action classifier
    logger.info("Initializing action classifier...")
    classifier = PVSGActionClassifier(
        holding_hand_dist=200.0,  # Increased for bbox-based estimation
        holding_min_frames=3,
        throwing_velocity_thresh=30.0,  # Lowered for better detection
        throwing_accel_thresh=20.0,
        picking_hand_dist=250.0,  # Increased for bbox-based estimation
        picking_appearance_window=15
    )
    
    # Process each frame
    logger.info("Generating action relations...")
    frame_count = 0
    relation_count = 0
    
    with open(output_path, 'w') as f:
        for frame_id in sorted(tracks_by_frame.keys()):
            tracks = tracks_by_frame[frame_id]
            
            # Convert to Detection objects
            detections = []
            for track in tracks:
                det = Detection(
                    track_id=track['track_id'],
                    class_name=track['category'],
                    bbox=track['bbox'],
                    frame_id=frame_id,
                    confidence=track.get('confidence', 1.0)
                )
                detections.append(det)
            
            # Estimate hands from person bboxes
            person_hands = {}
            for track in tracks:
                if track['category'] in ['person', 'adult', 'child']:
                    estimated_hands = estimate_hands_from_person(track['bbox'])
                    person_hands[track['track_id']] = estimated_hands
            
            # Predict relations
            relations = classifier.predict_relations(
                detections,
                person_hands,
                frame_id
            )
            
            # Convert to output format
            nodes = []
            for det in detections:
                nodes.append({
                    'track_id': det.track_id,
                    'class': det.class_name,
                    'bbox': det.bbox
                })
            
            edges = []
            for subj_id, obj_id, predicate in relations:
                edges.append({
                    'subject': subj_id,
                    'object': obj_id,
                    'relation': predicate
                })
            
            # Write frame
            frame_data = {
                'frame': frame_id,
                'nodes': nodes,
                'edges': edges
            }
            f.write(json.dumps(frame_data) + '\n')
            
            frame_count += 1
            relation_count += len(edges)
    
    logger.info(f"✓ Generated scene graph:")
    logger.info(f"  Frames: {frame_count}")
    logger.info(f"  Relations: {relation_count}")
    logger.info(f"  Output: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PVSG action-aware scene graph')
    parser.add_argument('--episode', required=True, help='Episode directory (e.g., results/pvsg_0003)')
    parser.add_argument('--output', required=True, help='Output scene graph JSONL path')
    
    args = parser.parse_args()
    
    generate_action_scene_graph(
        episode_dir=args.episode,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
