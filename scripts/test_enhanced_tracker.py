#!/usr/bin/env python3
"""
Minimal test script for EnhancedTracker integration.

Verifies:
- Tracker initialization
- Detection conversion
- Per-frame updates with embeddings
- Track confirmation and statistics
"""

import numpy as np
import sys
from pathlib import Path

# Add orion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.enhanced_tracker import EnhancedTracker


def generate_mock_detections(frame_idx: int, num_objects: int = 3) -> tuple:
    """Generate mock detections for testing."""
    detections = []
    embeddings = []
    
    for i in range(num_objects):
        # Simple motion: objects move slightly each frame
        x_base = 100 + i * 200
        y_base = 150
        x_offset = frame_idx * 5  # drift right
        
        x1, y1 = x_base + x_offset, y_base
        x2, y2 = x1 + 100, y1 + 120
        
        det = {
            'bbox_3d': np.array([x1 + 50, y1 + 60, 1500.0, 100, 120, 100], dtype=np.float32),
            'bbox_2d': np.array([x1, y1, x2, y2], dtype=np.float32),
            'class_name': f'object_{i}',
            'confidence': 0.85,
            'depth_mm': 1500.0,
        }
        detections.append(det)
        
        # Mock embedding (512-dim normalized)
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        embeddings.append(emb)
    
    return detections, embeddings


def main():
    print("="*80)
    print("Testing EnhancedTracker Integration")
    print("="*80)
    
    # Initialize tracker
    tracker = EnhancedTracker(
        max_age=10,
        min_hits=2,
        iou_threshold=0.3,
        appearance_threshold=0.5,
    )
    print("✓ Tracker initialized\n")
    
    # Simulate 20 frames
    num_frames = 20
    for frame_idx in range(num_frames):
        detections, embeddings = generate_mock_detections(frame_idx, num_objects=3)
        
        # Update tracker
        tracks = tracker.update(
            detections=detections,
            embeddings=embeddings,
            camera_pose=None,  # No CMC for this test
            frame_idx=frame_idx,
        )
        
        # Print stats every 5 frames
        if frame_idx % 5 == 0:
            stats = tracker.get_statistics()
            print(f"Frame {frame_idx:3d}: "
                  f"total={stats['total_tracks']}, "
                  f"confirmed={stats['confirmed_tracks']}, "
                  f"active={stats['active_tracks']}")
    
    # Final stats
    print("\n" + "="*80)
    stats = tracker.get_statistics()
    print(f"Final Statistics:")
    print(f"  Total tracks:     {stats['total_tracks']}")
    print(f"  Confirmed tracks: {stats['confirmed_tracks']}")
    print(f"  Active tracks:    {stats['active_tracks']}")
    print(f"  Next ID:          {stats['next_id']}")
    print("="*80)
    
    # Verify reasonable results
    assert stats['confirmed_tracks'] > 0, "No confirmed tracks!"
    assert stats['total_tracks'] <= 10, "Too many tracks (fragmentation?)"
    
    print("\n✅ All checks passed!")


if __name__ == "__main__":
    main()
