#!/usr/bin/env python3
"""
Test script to validate depth consistency filtering statistics.

Phase 3 Week 6 Day 2 - Depth Consistency Checking
"""

import numpy as np
import cv2
from orion.slam.slam_engine import SLAMEngine, SLAMConfig


def create_synthetic_scene(num_frames=30):
    """Create synthetic test data with known ground truth."""
    frames = []
    depth_maps = []
    
    # Create a simple textured pattern (BGR format for OpenCV)
    pattern_gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    pattern = cv2.cvtColor(pattern_gray, cv2.COLOR_GRAY2BGR)
    
    for i in range(num_frames):
        # Add slight motion blur
        frame = cv2.GaussianBlur(pattern, (3, 3), 0.5)
        
        # Create depth map with some noise
        depth = np.ones((480, 640), dtype=np.float32) * 1000.0
        depth += np.random.randn(480, 640) * 50.0  # 50mm noise
        
        frames.append(frame)
        depth_maps.append(depth)
    
    return frames, depth_maps


def test_depth_consistency_filtering():
    """Test depth consistency filtering on synthetic data."""
    
    print("="*70)
    print("DEPTH CONSISTENCY FILTERING - VALIDATION TEST")
    print("="*70)
    print()
    
    # Create synthetic test data
    print("✓ Generating synthetic test data (30 frames)...")
    frames, depth_maps = create_synthetic_scene(num_frames=30)
    
    # Configure SLAM with depth consistency enabled
    config = SLAMConfig(
        # Depth integration
        use_depth_uncertainty=True,
        use_temporal_depth_filter=True,
        use_scale_kalman=True,
        
        # Pose fusion
        enable_pose_fusion=True,
        rotation_weight_visual=0.8,
        
        # Depth consistency (Phase 3 Week 6 Day 2)
        enable_depth_consistency=True,
        epipolar_threshold=1.0,
        depth_ratio_threshold=0.3,
    )
    
    # Initialize SLAM
    slam = SLAMEngine(config)
    print("✓ SLAM initialized with depth consistency enabled")
    print()
    
    # Process frames
    print("Processing frames...")
    for i, (frame, depth_map) in enumerate(zip(frames, depth_maps)):
        # Create uncertainty map (uniform for simplicity)
        uncertainty_map = np.ones_like(depth_map) * 0.5
        
        # Process frame with correct signature: frame, timestamp, frame_idx, depth_map
        timestamp = i / 30.0  # 30 FPS
        pose = slam.process_frame(frame, timestamp, i, depth_map=depth_map)
        
        if i > 0 and i % 10 == 0:
            print(f"  Frame {i}: {'✓ Tracked' if pose is not None else '✗ Lost'}")
    
    print()
    print("="*70)
    print("DEPTH CONSISTENCY STATISTICS")
    print("="*70)
    
    # Get depth consistency statistics
    stats = slam.get_depth_consistency_stats()
    
    if not stats:
        print("⚠ WARNING: Depth consistency not enabled or no data")
        return
    
    print()
    print(f"Total checks:       {stats['total_checks']:,}")
    print(f"Total inliers:      {stats['total_inliers']:,}")
    print(f"Total outliers:     {stats['total_outliers']:,}")
    print()
    print(f"Inlier ratio:       {stats['inlier_ratio']:.1%}")
    print(f"Outlier ratio:      {stats['outlier_ratio']:.1%}")
    print()
    
    # Validation
    expected_inlier_ratio = 0.50  # Expect at least 50% inliers
    
    if stats['inlier_ratio'] >= expected_inlier_ratio:
        print(f"✓ PASS: Inlier ratio ({stats['inlier_ratio']:.1%}) >= {expected_inlier_ratio:.1%}")
    else:
        print(f"✗ FAIL: Inlier ratio ({stats['inlier_ratio']:.1%}) < {expected_inlier_ratio:.1%}")
    
    print()
    print("="*70)
    print("SLAM TRACKING QUALITY")
    print("="*70)
    print()
    
    quality = slam.get_tracking_quality()
    print(f"Tracking quality:   {quality:.1%}")
    print(f"Total poses:        {len(slam.poses)}")
    print()
    
    if quality >= 0.8:
        print(f"✓ PASS: Tracking quality ({quality:.1%}) >= 80%")
    else:
        print(f"⚠ WARNING: Tracking quality ({quality:.1%}) < 80%")
    
    print()
    print("="*70)
    print("TEST COMPLETE")
    print("="*70)
    print()
    print("Key Observations:")
    print(f"  • Depth consistency filtering processed {stats['total_checks']:,} point pairs")
    print(f"  • {stats['inlier_ratio']:.1%} passed consistency checks (epipolar + ratio + range)")
    print(f"  • {stats['outlier_ratio']:.1%} rejected as outliers")
    print(f"  • Tracking maintained {quality:.1%} quality over {len(slam.poses)} frames")
    print()


if __name__ == "__main__":
    test_depth_consistency_filtering()
