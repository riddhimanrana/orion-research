"""
Test 3D CIS Integration
========================

Quick test to verify 3D CIS is working with SLAM.

Usage:
    python scripts/test_3d_cis.py
"""

import cv2
import numpy as np
from orion.slam.slam_engine import SLAMEngine
from orion.slam.projection_3d import project_bbox_to_3d, compute_3d_velocity
from orion.semantic.cis_scorer_3d import CausalInfluenceScorer3D
from orion.backends.midas_depth import MiDaSDepth

def main():
    print("="*80)
    print("Testing 3D CIS Integration")
    print("="*80)
    
    # Initialize components
    print("\n[1/5] Initializing SLAM and depth estimation...")
    slam = SLAMEngine()
    depth_backend = MiDaSDepth()
    
    print("\n[2/5] Initializing 3D CIS scorer...")
    cis_scorer = CausalInfluenceScorer3D(
        weight_temporal=0.30,
        weight_spatial=0.44,
        weight_motion=0.21,
        weight_semantic=0.06,
        temporal_decay_tau=4.0,
        max_spatial_distance_mm=600.0,
        cis_threshold=0.50,
    )
    print(f"   Weights: T=0.30, S=0.44, M=0.21, Se=0.06")
    print(f"   Threshold: 0.50")
    print(f"   Max distance: 600mm (60cm)")
    
    # Load test video
    print("\n[3/5] Loading test video...")
    cap = cv2.VideoCapture('data/examples/video.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Process first few frames to get SLAM working
    print("\n[4/5] Processing frames to establish SLAM...")
    frame_idx = 0
    slam_poses = []
    depth_maps = []
    
    while frame_idx < 30:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 3 == 0:  # Process every 3rd frame
            depth = depth_backend.estimate(frame)
            result = slam.process_frame(frame, depth)
            
            if result.success:
                slam_poses.append(slam.current_pose.copy())
                depth_maps.append(depth)
                
                if len(slam_poses) % 5 == 0:
                    pos = slam.current_pose[:3, 3]
                    print(f"   Frame {frame_idx}: SLAM pose = [{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}]mm")
        
        frame_idx += 1
    
    cap.release()
    
    print(f"\n   ✓ Processed {len(slam_poses)} frames with SLAM")
    
    # Test 3D projection
    print("\n[5/5] Testing 3D CIS calculation...")
    
    if len(slam_poses) >= 2:
        # Create mock entities
        h, w = depth_maps[0].shape
        
        # Entity 1: Center-left of frame
        bbox1 = (int(w*0.3), int(h*0.4), int(w*0.4), int(h*0.6))
        pos1 = project_bbox_to_3d(bbox1, depth_maps[0], slam.K, slam_poses[0])
        
        # Entity 2: Center-right of frame
        bbox2 = (int(w*0.6), int(h*0.4), int(w*0.7), int(h*0.6))
        pos2_t0 = project_bbox_to_3d(bbox2, depth_maps[0], slam.K, slam_poses[0])
        pos2_t1 = project_bbox_to_3d(bbox2, depth_maps[1], slam.K, slam_poses[1]) if len(slam_poses) > 1 else pos2_t0
        
        if pos1 is not None and pos2_t0 is not None and pos2_t1 is not None:
            # Compute velocity for entity 2
            time_delta = 1.0 / fps  # Time between frames
            vel2 = compute_3d_velocity(pos2_t0, pos2_t1, time_delta)
            
            # Create entity dicts
            entity1 = {
                'centroid_3d_mm': pos1,
                'velocity_3d': np.array([0, 0, 0]),  # Static
                'embedding': np.random.rand(512),  # Mock embedding
                'class_label': 'cup',
                'is_hand': False,
            }
            
            entity2 = {
                'centroid_3d_mm': pos2_t1,
                'velocity_3d': vel2,
                'embedding': np.random.rand(512),
                'class_label': 'hand',
                'is_hand': True,
            }
            
            # Compute CIS
            cis_score, components = cis_scorer.calculate_cis(
                entity1,
                entity2,
                time_delta=0.5  # 0.5 seconds apart
            )
            
            print(f"\n   Entity 1 (cup) at {pos1}")
            print(f"   Entity 2 (hand) at {pos2_t1}")
            print(f"   Distance: {np.linalg.norm(pos1 - pos2_t1):.0f}mm")
            print(f"\n   CIS Components:")
            print(f"      Temporal:  {components.temporal:.3f}")
            print(f"      Spatial:   {components.spatial:.3f}")
            print(f"      Motion:    {components.motion:.3f}")
            print(f"      Semantic:  {components.semantic:.3f}")
            print(f"      Hand Bonus:{components.hand_bonus:.3f}")
            print(f"\n   ✓ Final CIS: {cis_score:.3f}")
            
            if cis_score > cis_scorer.cis_threshold:
                print(f"   ✓ Causal link detected (threshold: {cis_scorer.cis_threshold:.2f})")
            else:
                print(f"   ✗ Below threshold ({cis_scorer.cis_threshold:.2f})")
        else:
            print("   ✗ Could not project entities to 3D (invalid depth)")
    else:
        print("   ✗ Not enough SLAM poses")
    
    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run full pipeline with use_3d_cis=True")
    print("2. Compare CIS scores with 2D baseline")
    print("3. Collect ground truth for HPO")

if __name__ == "__main__":
    main()
