#!/usr/bin/env python3
"""
Test script for multi-frame depth fusion.

Phase 3 Week 6 Day 3 - Validates depth fusion across sliding window.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from orion.slam.slam_engine import SLAMEngine, SLAMConfig
from orion.slam.multi_frame_depth_fusion import (
    warp_depth_to_frame,
    fuse_depth_maps_weighted,
    reject_depth_outliers_temporal,
    MultiFrameDepthFusion,
)


def create_synthetic_scene_with_motion(num_frames=50):
    """Create synthetic test data with camera motion."""
    frames = []
    depth_maps = []
    ground_truth_poses = []
    
    # Create a textured pattern
    pattern_gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    pattern = cv2.cvtColor(pattern_gray, cv2.COLOR_GRAY2BGR)
    
    # Simulate forward motion
    for i in range(num_frames):
        # Add slight variations to texture
        noise = np.random.randint(-10, 10, (480, 640, 3), dtype=np.int16)
        frame = np.clip(pattern.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frame = cv2.GaussianBlur(frame, (3, 3), 0.5)
        
        # Create depth map with some noise (simulate moving forward)
        base_depth = 1000.0 - i * 10.0  # Moving closer
        depth = np.ones((480, 640), dtype=np.float32) * base_depth
        depth += np.random.randn(480, 640) * 50.0  # 50mm noise
        
        # Ground truth pose (forward motion along Z)
        pose = np.eye(4)
        pose[2, 3] = i * 10.0  # 10mm per frame forward
        
        frames.append(frame)
        depth_maps.append(depth)
        ground_truth_poses.append(pose)
    
    return frames, depth_maps, ground_truth_poses


def test_depth_warping():
    """Test depth warping between frames."""
    print("="*70)
    print("TEST 1: Depth Warping")
    print("="*70)
    print()
    
    # Create simple test case
    H, W = 480, 640
    depth_src = np.ones((H, W), dtype=np.float32) * 1000.0
    
    # Camera intrinsics
    K = np.array([
        [500.0, 0, 320.0],
        [0, 500.0, 240.0],
        [0, 0, 1.0]
    ])
    
    # Identity pose
    pose_src = np.eye(4)
    
    # Target pose (moved 100mm forward)
    pose_tgt = np.eye(4)
    pose_tgt[2, 3] = 100.0  # 100mm translation in Z
    
    # Warp depth
    warped_depth, valid_mask = warp_depth_to_frame(depth_src, pose_src, pose_tgt, K)
    
    valid_ratio = valid_mask.sum() / (H * W)
    
    print(f"✓ Depth warping complete")
    print(f"  Valid ratio: {valid_ratio:.1%}")
    print(f"  Mean warped depth: {warped_depth[valid_mask].mean():.1f}mm")
    print(f"  Expected: ~900mm (closer due to forward motion)")
    
    if valid_ratio > 0.5:
        print(f"✓ PASS: Valid ratio ({valid_ratio:.1%}) > 50%")
    else:
        print(f"✗ FAIL: Valid ratio ({valid_ratio:.1%}) <= 50%")
    
    print()


def test_confidence_weighted_fusion():
    """Test confidence-weighted depth fusion."""
    print("="*70)
    print("TEST 2: Confidence-Weighted Fusion")
    print("="*70)
    print()
    
    H, W = 480, 640
    
    # Create 3 depth maps with different noise levels
    depth1 = np.ones((H, W), dtype=np.float32) * 1000.0 + np.random.randn(H, W) * 20.0
    depth2 = np.ones((H, W), dtype=np.float32) * 1000.0 + np.random.randn(H, W) * 50.0
    depth3 = np.ones((H, W), dtype=np.float32) * 1000.0 + np.random.randn(H, W) * 100.0
    
    # Confidence maps (higher confidence for less noisy)
    conf1 = np.ones((H, W), dtype=np.float32) * 0.9  # High confidence
    conf2 = np.ones((H, W), dtype=np.float32) * 0.5  # Medium confidence
    conf3 = np.ones((H, W), dtype=np.float32) * 0.2  # Low confidence
    
    # Valid masks
    valid1 = np.ones((H, W), dtype=bool)
    valid2 = np.ones((H, W), dtype=bool)
    valid3 = np.ones((H, W), dtype=bool)
    
    # Fuse
    fused_depth, fused_conf = fuse_depth_maps_weighted(
        [depth1, depth2, depth3],
        [conf1, conf2, conf3],
        [valid1, valid2, valid3]
    )
    
    # Compute noise levels
    noise_d1 = np.std(depth1 - 1000.0)
    noise_d2 = np.std(depth2 - 1000.0)
    noise_d3 = np.std(depth3 - 1000.0)
    noise_fused = np.std(fused_depth - 1000.0)
    
    print(f"✓ Confidence-weighted fusion complete")
    print(f"  Depth 1 noise: {noise_d1:.1f}mm (conf=0.9)")
    print(f"  Depth 2 noise: {noise_d2:.1f}mm (conf=0.5)")
    print(f"  Depth 3 noise: {noise_d3:.1f}mm (conf=0.2)")
    print(f"  Fused noise:   {noise_fused:.1f}mm")
    print(f"  Mean confidence: {fused_conf.mean():.2f}")
    
    # Fused should have less noise than average
    avg_noise = (noise_d1 + noise_d2 + noise_d3) / 3
    if noise_fused < avg_noise:
        print(f"✓ PASS: Fused noise ({noise_fused:.1f}) < average ({avg_noise:.1f})")
    else:
        print(f"⚠ WARNING: Fused noise ({noise_fused:.1f}) >= average ({avg_noise:.1f})")
    
    print()


def test_temporal_outlier_rejection():
    """Test temporal outlier rejection."""
    print("="*70)
    print("TEST 3: Temporal Outlier Rejection")
    print("="*70)
    print()
    
    H, W = 480, 640
    
    # Create 5 depth maps with one outlier
    depth_maps = []
    valid_masks = []
    
    for i in range(5):
        if i == 2:
            # Frame 2 is an outlier (500mm offset)
            depth = np.ones((H, W), dtype=np.float32) * 1500.0
        else:
            # Normal frames
            depth = np.ones((H, W), dtype=np.float32) * 1000.0
        
        depth += np.random.randn(H, W) * 20.0
        depth_maps.append(depth)
        valid_masks.append(np.ones((H, W), dtype=bool))
    
    # Reject outliers
    refined_masks = reject_depth_outliers_temporal(
        depth_maps, valid_masks, threshold_mm=100.0
    )
    
    # Check frame 2 has low inlier ratio
    inlier_ratios = [mask.sum() / (H * W) for mask in refined_masks]
    
    print(f"✓ Temporal outlier rejection complete")
    for i, ratio in enumerate(inlier_ratios):
        status = "OUTLIER" if ratio < 0.5 else "INLIER"
        print(f"  Frame {i}: {ratio:.1%} inliers ({status})")
    
    if inlier_ratios[2] < 0.5 and all(inlier_ratios[i] > 0.9 for i in [0,1,3,4]):
        print(f"✓ PASS: Outlier frame detected (frame 2: {inlier_ratios[2]:.1%})")
    else:
        print(f"✗ FAIL: Outlier detection incorrect")
    
    print()


def test_multi_frame_fusion_pipeline():
    """Test complete multi-frame fusion pipeline with SLAM."""
    print("="*70)
    print("TEST 4: Multi-Frame Fusion Pipeline")
    print("="*70)
    print()
    
    # Generate synthetic scene
    print("✓ Generating synthetic scene (50 frames with forward motion)...")
    frames, depth_maps, gt_poses = create_synthetic_scene_with_motion(num_frames=50)
    
    # Configure SLAM with all features
    config = SLAMConfig(
        # Depth integration
        use_depth_uncertainty=True,
        use_temporal_depth_filter=True,
        use_scale_kalman=True,
        
        # Pose fusion
        enable_pose_fusion=True,
        
        # Depth consistency
        enable_depth_consistency=True,
        
        # Multi-frame fusion (Day 3)
        enable_multi_frame_fusion=True,
        fusion_window_size=5,
        fusion_outlier_threshold=100.0,
        fusion_min_confidence=0.3,
    )
    
    # Initialize SLAM
    slam = SLAMEngine(config)
    print("✓ SLAM initialized with multi-frame fusion")
    print()
    
    # Process frames
    print("Processing frames...")
    for i, (frame, depth_map) in enumerate(zip(frames, depth_maps)):
        timestamp = i / 30.0  # 30 FPS
        pose = slam.process_frame(frame, timestamp, i, depth_map=depth_map)
        
        if i > 0 and i % 10 == 0:
            print(f"  Frame {i}: {'✓ Tracked' if pose is not None else '✗ Lost'}")
    
    print()
    print("="*70)
    print("MULTI-FRAME FUSION STATISTICS")
    print("="*70)
    print()
    
    # Get fusion statistics
    fusion_stats = slam.get_multi_frame_fusion_stats()
    
    if fusion_stats:
        print(f"Total fusions:          {fusion_stats['total_fusions']}")
        print(f"Total frames used:      {fusion_stats['total_frames_used']}")
        print(f"Avg frames per fusion:  {fusion_stats['avg_frames_per_fusion']:.2f}")
        print(f"Current window size:    {fusion_stats['window_size']}")
        print()
        
        if fusion_stats['avg_frames_per_fusion'] >= 2.0:
            print(f"✓ PASS: Using multiple frames ({fusion_stats['avg_frames_per_fusion']:.2f} avg)")
        else:
            print(f"⚠ WARNING: Low frame usage ({fusion_stats['avg_frames_per_fusion']:.2f} avg)")
    else:
        print("⚠ WARNING: No fusion statistics available")
    
    print()
    print("="*70)
    print("DEPTH CONSISTENCY STATISTICS")
    print("="*70)
    print()
    
    # Get depth consistency statistics
    consistency_stats = slam.get_depth_consistency_stats()
    
    if consistency_stats:
        print(f"Total checks:       {consistency_stats['total_checks']:,}")
        print(f"Total inliers:      {consistency_stats['total_inliers']:,}")
        print(f"Inlier ratio:       {consistency_stats['inlier_ratio']:.1%}")
        print()
    
    print("="*70)
    print("SLAM TRACKING QUALITY")
    print("="*70)
    print()
    
    quality = slam.get_tracking_quality()
    print(f"Tracking quality:   {quality:.1%}")
    print(f"Total poses:        {len(slam.poses)}")
    
    if quality >= 0.8:
        print(f"✓ PASS: Tracking quality ({quality:.1%}) >= 80%")
    else:
        print(f"⚠ WARNING: Tracking quality ({quality:.1%}) < 80%")
    
    print()
    print("="*70)
    print("TEST COMPLETE")
    print("="*70)
    print()
    
    if fusion_stats:
        print("Key Results:")
        print(f"  • Multi-frame fusion processed {fusion_stats['total_fusions']} fusions")
        print(f"  • Average {fusion_stats['avg_frames_per_fusion']:.1f} frames per fusion")
        print(f"  • Sliding window size: {fusion_stats['window_size']} frames")
        if consistency_stats:
            print(f"  • Depth consistency: {consistency_stats['inlier_ratio']:.1%} inliers")
        print(f"  • Tracking quality: {quality:.1%}")
        print()


if __name__ == "__main__":
    # Run all tests
    test_depth_warping()
    test_confidence_weighted_fusion()
    test_temporal_outlier_rejection()
    test_multi_frame_fusion_pipeline()
