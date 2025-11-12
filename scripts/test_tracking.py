#!/usr/bin/env python3
"""Phase 4: Object Tracking with Temporal Persistence"""

import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orion.perception.object_tracker import ObjectTracker
from orion.perception.unified_frame import Object3D


def main():
    print("\n" + "="*80)
    print("PHASE 4: OBJECT TRACKING TEST")
    print("="*80)
    print("\nDemonstrating temporal object tracking")
    print("Converting 72 frame-level detections → ~6 unique tracked objects\n")
    
    # Use default tracking
    tracker = ObjectTracker(
        max_distance_px=100.0,
        max_distance_3d=1.0,
        max_age_frames=10  # Keep tracks alive longer
    )
    
    # Simulate 20 frames with 6 persistent objects
    # Objects appear together (simulating ~4 detections per frame)
    objects_timeline = {
        'keyboard': list(range(0, 20)),
        'mouse': list(range(0, 20)),
        'tv': list(range(0, 20)),
        'laptop': list(range(0, 20)),
        'chair': list(range(0, 20)),
        'person': list(range(0, 20)),
    }
    
    total_detections = 0
    obj_positions = {
        'keyboard': ([50, 100], [50, 100], np.array([0.3, 0.2, 1.5])),
        'mouse': ([150, 200], [50, 100], np.array([0.5, 0.2, 1.5])),
        'tv': ([250, 300], [50, 100], np.array([1.0, 0.5, 0.5])),
        'laptop': ([50, 100], [150, 200], np.array([0.3, 0.5, 2.0])),
        'chair': ([150, 200], [150, 200], np.array([-0.5, 0.0, 1.0])),
        'person': ([250, 300], [150, 200], np.array([2.0, 1.0, 2.0])),
    }
    
    for frame_idx in range(20):
        detections = []
        centroids = []
        
        for obj_class, frame_list in objects_timeline.items():
            if frame_idx in frame_list:
                x_range, y_range, pos_3d = obj_positions[obj_class]
                x_center = (x_range[0] + x_range[1]) // 2
                y_center = (y_range[0] + y_range[1]) // 2
                
                det = Object3D(
                    id=-1,
                    class_name=obj_class,
                    confidence=0.9,
                    bbox_2d=(x_range[0], y_range[0], x_range[1], y_range[1]),
                    position_3d=pos_3d,
                    clip_embedding=None,
                )
                detections.append(det)
                centroids.append((x_center, y_center))
        
        tracker.update(detections, centroids, frame_idx)
        total_detections += len(detections)
        print(f"  Frame {frame_idx:2d}: {len(detections):2d} detections → {len(tracker.tracks):2d} active tracks")
    
    print(f"\nResults:")
    print(f"  Total detections: {total_detections}")
    print(f"  Unique tracked objects: {len(tracker.tracks)}")
    print(f"  Reduction factor: {total_detections / max(len(tracker.tracks), 1):.1f}x")
    
    print(f"\nTracked objects:")
    for tid, track in sorted(tracker.tracks.items()):
        print(f"  Track {tid}: {track.class_name:12s} age={track.age}")
    
    success = len(tracker.tracks) == 6
    print(f"\n{'PASSED' if success else 'FAILED'}: Expected 6, got {len(tracker.tracks)}")
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
