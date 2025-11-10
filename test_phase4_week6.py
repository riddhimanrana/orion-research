#!/usr/bin/env python3
"""
Phase 4 SLAM Test with Week 6 Improvements

Tests the complete SLAM pipeline with all Phase 3 Week 6 enhancements:
- Depth uncertainty estimation
- Temporal depth filtering
- Hybrid visual-depth pose fusion
- Depth consistency checking
- Multi-frame depth fusion
- Loop closure detection

Compare with old Phase 4 results (Nov 5, 2025):
- Old tracking success: 14.4%
- Expected new tracking: 60-80%
- Old zone count: 5
- Expected zone count: 3-4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import argparse
from typing import List, Dict, Tuple
import time

from orion.slam.slam_engine import SLAMEngine, SLAMConfig


def create_synthetic_video_sequence(num_frames=100, add_motion=True):
    """
    Create synthetic video with known camera motion for testing.
    
    Simulates walking through 3 rooms with loop closure (return to room 1).
    """
    frames = []
    depth_maps = []
    ground_truth_poses = []
    ground_truth_zones = []
    
    # Room parameters
    room_1_frames = range(0, 30)  # Bedroom 1
    room_2_frames = range(30, 60)  # Kitchen
    room_3_frames = range(60, 80)  # Living room
    room_1_return_frames = range(80, 100)  # Back to bedroom 1 (loop closure)
    
    # Create textured patterns for each room
    np.random.seed(42)
    room_1_pattern = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    room_2_pattern = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    room_3_pattern = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    for i in range(num_frames):
        # Determine current room
        if i in room_1_frames or i in room_1_return_frames:
            pattern = room_1_pattern
            zone_id = 0
            base_depth = 2000.0
        elif i in room_2_frames:
            pattern = room_2_pattern
            zone_id = 1
            base_depth = 2500.0
        else:  # room_3_frames
            pattern = room_3_pattern
            zone_id = 2
            base_depth = 1800.0
        
        # Add motion blur and noise
        noise = np.random.randint(-20, 20, (480, 640), dtype=np.int16)
        frame_gray = np.clip(pattern.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        frame = cv2.GaussianBlur(frame, (3, 3), 0.5)
        
        # Create depth map
        depth = np.ones((480, 640), dtype=np.float32) * base_depth
        depth += np.random.randn(480, 640) * 50.0  # 50mm noise
        
        # Add camera motion
        if add_motion:
            if i in room_1_frames:
                # Forward motion in room 1
                motion_x, motion_y, motion_z = 0, 0, i * 20.0
            elif i in room_2_frames:
                # Moving through kitchen
                motion_x, motion_y, motion_z = (i - 30) * 30.0, 0, 600.0
            elif i in room_3_frames:
                # Living room
                motion_x, motion_y, motion_z = 900.0, 0, (i - 60) * 25.0
            else:  # room_1_return
                # Return to room 1 (loop closure)
                motion_x, motion_y, motion_z = (80 - i) * 15.0, 0, (i - 80) * 20.0
        else:
            motion_x, motion_y, motion_z = 0, 0, 0
        
        # Ground truth pose
        pose = np.eye(4)
        pose[0, 3] = motion_x
        pose[1, 3] = motion_y
        pose[2, 3] = motion_z
        
        frames.append(frame)
        depth_maps.append(depth)
        ground_truth_poses.append(pose)
        ground_truth_zones.append(zone_id)
    
    return frames, depth_maps, ground_truth_poses, ground_truth_zones


def test_slam_with_week6_improvements(
    num_frames=100,
    enable_all_features=True,
    verbose=True
):
    """
    Test SLAM with all Phase 3 Week 6 improvements enabled.
    """
    print("="*70)
    print("PHASE 4 SLAM TEST - WITH WEEK 6 IMPROVEMENTS")
    print("="*70)
    print()
    
    # Generate synthetic test sequence
    print(f"✓ Generating synthetic sequence ({num_frames} frames)...")
    frames, depth_maps, gt_poses, gt_zones = create_synthetic_video_sequence(
        num_frames=num_frames,
        add_motion=True
    )
    print(f"  Expected zones: 3 (room 1, room 2, room 3)")
    print(f"  Expected loop closure: Frame 80 should match frame 0-30")
    print()
    
    # Configure SLAM with all Week 6 features
    if enable_all_features:
        print("✓ Configuring SLAM with ALL Week 6 features...")
        config = SLAMConfig(
            # Base SLAM
            num_features=1500,
            min_matches=15,
            ransac_threshold=1.0,
            
            # Loop closure
            enable_loop_closure=True,
            bow_similarity_threshold=0.70,
            
            # Phase 3 Week 5
            use_depth_uncertainty=True,
            use_temporal_depth_filter=True,
            use_scale_kalman=True,
            
            # Phase 3 Week 6 Day 1
            enable_pose_fusion=True,
            rotation_weight_visual=0.8,
            
            # Phase 3 Week 6 Day 2
            enable_depth_consistency=True,
            epipolar_threshold=1.0,
            depth_ratio_threshold=0.3,
            
            # Phase 3 Week 6 Day 3
            enable_multi_frame_fusion=True,
            fusion_window_size=5,
        )
    else:
        print("✓ Configuring SLAM with BASELINE settings (no Week 6)...")
        config = SLAMConfig(
            num_features=1500,
            min_matches=15,
            enable_loop_closure=False,
            use_depth_uncertainty=False,
            use_temporal_depth_filter=False,
            use_scale_kalman=False,
            enable_pose_fusion=False,
            enable_depth_consistency=False,
            enable_multi_frame_fusion=False,
        )
    
    slam = SLAMEngine(config)
    print()
    
    # Process frames
    print("Processing frames...")
    start_time = time.time()
    
    tracking_successes = 0
    tracking_failures = 0
    loop_closures_detected = 0
    
    for i, (frame, depth_map) in enumerate(zip(frames, depth_maps)):
        timestamp = i / 30.0  # 30 FPS
        
        # Process frame
        pose = slam.process_frame(frame, timestamp, i, depth_map=depth_map)
        
        if pose is not None:
            tracking_successes += 1
        else:
            tracking_failures += 1
        
        # Check for loop closures
        if hasattr(slam, 'loop_detector') and slam.loop_detector:
            loop_count = len(slam.loop_detector.loop_closures)
            if loop_count > loop_closures_detected:
                loop_closures_detected = loop_count
                if verbose:
                    print(f"  Frame {i}: Loop closure detected!")
        
        # Progress update
        if verbose and i > 0 and i % 20 == 0:
            success_rate = tracking_successes / (i + 1) * 100
            print(f"  Frame {i}: Tracking success rate: {success_rate:.1f}%")
    
    elapsed = time.time() - start_time
    fps = num_frames / elapsed
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    
    # Tracking performance
    tracking_success_rate = tracking_successes / num_frames * 100
    print(f"**Tracking Performance**:")
    print(f"  Success rate: {tracking_success_rate:.1f}% ({tracking_successes}/{num_frames})")
    print(f"  Failures: {tracking_failures}")
    print(f"  Target: >80%")
    if tracking_success_rate >= 80:
        print(f"  Status: ✅ PASS")
    elif tracking_success_rate >= 60:
        print(f"  Status: 🟡 ACCEPTABLE")
    else:
        print(f"  Status: ❌ FAIL")
    print()
    
    # Loop closure
    print(f"**Loop Closure**:")
    print(f"  Detected: {loop_closures_detected}")
    print(f"  Expected: 1-3 (return to room 1)")
    if loop_closures_detected > 0:
        print(f"  Status: ✅ PASS")
    else:
        print(f"  Status: ❌ FAIL")
    print()
    
    # Get statistics
    if hasattr(slam, 'get_depth_consistency_stats'):
        consistency_stats = slam.get_depth_consistency_stats()
        if consistency_stats:
            print(f"**Depth Consistency** (Week 6 Day 2):")
            print(f"  Total checks: {consistency_stats.get('total_checks', 0):,}")
            print(f"  Inlier ratio: {consistency_stats.get('inlier_ratio', 0):.1%}")
            print(f"  Target: >70%")
            if consistency_stats.get('inlier_ratio', 0) >= 0.70:
                print(f"  Status: ✅ PASS")
            print()
    
    if hasattr(slam, 'get_multi_frame_fusion_stats'):
        fusion_stats = slam.get_multi_frame_fusion_stats()
        if fusion_stats:
            print(f"**Multi-Frame Fusion** (Week 6 Day 3):")
            print(f"  Total fusions: {fusion_stats.get('total_fusions', 0)}")
            print(f"  Avg frames/fusion: {fusion_stats.get('avg_frames_per_fusion', 0):.2f}")
            print(f"  Status: ✅ Active")
            print()
    
    # Performance
    print(f"**Performance**:")
    print(f"  Processing time: {elapsed:.1f}s")
    print(f"  FPS: {fps:.2f}")
    print(f"  Target: >0.5 FPS")
    if fps >= 0.5:
        print(f"  Status: ✅ PASS")
    else:
        print(f"  Status: ⚠️ SLOW")
    print()
    
    # Trajectory
    trajectory = slam.get_trajectory()
    if len(trajectory) > 0:
        trajectory_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        print(f"**Trajectory**:")
        print(f"  Total poses: {len(slam.poses)}")
        print(f"  Path length: {trajectory_length:.2f} mm ({trajectory_length/1000:.2f} m)")
        print(f"  Status: ✅ Computed")
    
    print()
    print("="*70)
    print("COMPARISON WITH OLD PHASE 4 RESULTS (Nov 5, 2025)")
    print("="*70)
    print()
    
    print(f"| Metric | Old (Nov 5) | New (Week 6) | Improvement |")
    print(f"|--------|-------------|--------------|-------------|")
    print(f"| Tracking Success | 14.4% | {tracking_success_rate:.1f}% | {tracking_success_rate - 14.4:+.1f}% |")
    print(f"| Loop Closures | Partial | {loop_closures_detected} detected | {'✅' if loop_closures_detected > 0 else '❌'} |")
    print(f"| Depth Consistency | Unknown | {consistency_stats.get('inlier_ratio', 0):.1%} | {'✅' if consistency_stats.get('inlier_ratio', 0) > 0 else 'N/A'} |")
    print(f"| Processing FPS | 0.73 | {fps:.2f} | {fps - 0.73:+.2f} |")
    
    print()
    print("="*70)
    
    return {
        'tracking_success_rate': tracking_success_rate,
        'loop_closures': loop_closures_detected,
        'fps': fps,
        'trajectory_length': trajectory_length if len(trajectory) > 0 else 0,
        'consistency_stats': consistency_stats if hasattr(slam, 'get_depth_consistency_stats') else {},
    }


def main():
    parser = argparse.ArgumentParser(description='Test Phase 4 SLAM with Week 6 improvements')
    parser.add_argument('--frames', type=int, default=100,
                       help='Number of frames to test (default: 100)')
    parser.add_argument('--baseline', action='store_true',
                       help='Test with baseline SLAM (no Week 6 features)')
    parser.add_argument('--quiet', action='store_true',
                       help='Less verbose output')
    
    args = parser.parse_args()
    
    if args.baseline:
        print("\n🔵 Testing BASELINE SLAM (No Week 6 improvements)")
        print("="*70)
        print()
        results_baseline = test_slam_with_week6_improvements(
            num_frames=args.frames,
            enable_all_features=False,
            verbose=not args.quiet
        )
        
        print("\n\n")
        print("🟢 Testing WITH WEEK 6 IMPROVEMENTS")
        print("="*70)
        print()
        results_week6 = test_slam_with_week6_improvements(
            num_frames=args.frames,
            enable_all_features=True,
            verbose=not args.quiet
        )
        
        # Compare
        print("\n\n")
        print("="*70)
        print("BASELINE vs WEEK 6 COMPARISON")
        print("="*70)
        print()
        print(f"| Metric | Baseline | Week 6 | Improvement |")
        print(f"|--------|----------|--------|-------------|")
        print(f"| Tracking | {results_baseline['tracking_success_rate']:.1f}% | {results_week6['tracking_success_rate']:.1f}% | {results_week6['tracking_success_rate'] - results_baseline['tracking_success_rate']:+.1f}% |")
        print(f"| Loop Closures | {results_baseline['loop_closures']} | {results_week6['loop_closures']} | {results_week6['loop_closures'] - results_baseline['loop_closures']:+d} |")
        print(f"| FPS | {results_baseline['fps']:.2f} | {results_week6['fps']:.2f} | {results_week6['fps'] - results_baseline['fps']:+.2f} |")
    else:
        results = test_slam_with_week6_improvements(
            num_frames=args.frames,
            enable_all_features=True,
            verbose=not args.quiet
        )
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()
