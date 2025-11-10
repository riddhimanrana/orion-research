"""
Phase 4 Week 2 - Zone Detection Refinement Test

Tests world-coordinate zone clustering and semantic zone refinement.

Week 2 Goals:
- Day 4: World-coordinate zone clustering (use SLAM transforms)
- Day 5: Semantic zone refinement (spatial + semantic clustering)

Success Criteria:
- Detect correct number of zones (3 rooms = 3 zones)
- Zone positions stable in world coordinates
- Semantic patterns improve clustering
- Zone re-identification when revisiting

Author: Orion Research Team
Date: November 9, 2025
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import argparse

# Add orion to path - import only what we need
sys.path.insert(0, str(Path(__file__).parent))

# Import directly to avoid loading full orion package (mediapipe/tensorflow)
from orion.slam.slam_engine import SLAMEngine, SLAMConfig
from orion.slam.world_coordinate_tracker import WorldCoordinateTracker
from orion.semantic.zone_manager import ZoneManager

print("✓ Imports complete (bypassed mediapipe/tensorflow loading)")


def generate_3room_sequence_with_objects(num_frames=150):
    """
    Generate synthetic sequence with 3 rooms and realistic objects.
    
    Room layout:
    - Room 1 (Bedroom): frames 0-40, returns at 120-149 (LOOP CLOSURE)
    - Room 2 (Kitchen): frames 41-80
    - Room 3 (Living room): frames 81-119
    
    Each room has semantically consistent objects:
    - Bedroom: bed, nightstand, dresser, lamp
    - Kitchen: stove, sink, fridge, table
    - Living room: couch, TV, coffee table, bookshelf
    
    Returns:
        frames: List of numpy arrays (640x480 RGB)
        depths: List of depth maps (640x480 uint16, mm)
        uncertainties: List of uncertainty maps (640x480 float32)
        objects: List of object detections per frame
    """
    print(f"✓ Generating 3-room sequence with objects ({num_frames} frames)...")
    print("  Room layout:")
    print("    • Room 1 (Bedroom): frames 0-40, 120-149 (LOOP CLOSURE)")
    print("    • Room 2 (Kitchen): frames 41-80")
    print("    • Room 3 (Living room): frames 81-119")
    
    frames = []
    depths = []
    uncertainties = []
    objects_per_frame = []
    
    # Room definitions (world coordinates in mm)
    room_centers = {
        'bedroom': np.array([0.0, 0.0, 0.0]),      # Origin
        'kitchen': np.array([5000.0, 0.0, 0.0]),   # 5m to the right
        'living': np.array([10000.0, 0.0, 0.0])    # 10m to the right
    }
    
    # Object templates per room (relative positions within room)
    room_objects = {
        'bedroom': [
            {'class': 'bed', 'pos': np.array([0.0, 0.0, 2500.0]), 'size': (200, 150)},
            {'class': 'nightstand', 'pos': np.array([-800.0, 0.0, 2500.0]), 'size': (60, 80)},
            {'class': 'dresser', 'pos': np.array([1200.0, 0.0, 2000.0]), 'size': (100, 120)},
            {'class': 'lamp', 'pos': np.array([-800.0, 500.0, 2500.0]), 'size': (40, 50)},
        ],
        'kitchen': [
            {'class': 'stove', 'pos': np.array([0.0, 0.0, 3000.0]), 'size': (80, 100)},
            {'class': 'sink', 'pos': np.array([-1000.0, 0.0, 3000.0]), 'size': (70, 90)},
            {'class': 'refrigerator', 'pos': np.array([1500.0, 0.0, 2500.0]), 'size': (90, 110)},
            {'class': 'dining table', 'pos': np.array([0.0, 0.0, 1500.0]), 'size': (120, 100)},
        ],
        'living': [
            {'class': 'couch', 'pos': np.array([0.0, 0.0, 3500.0]), 'size': (180, 140)},
            {'class': 'tv', 'pos': np.array([0.0, 0.0, 5000.0]), 'size': (100, 80)},
            {'class': 'coffee table', 'pos': np.array([0.0, 0.0, 2500.0]), 'size': (90, 70)},
            {'class': 'bookshelf', 'pos': np.array([1200.0, 0.0, 4000.0]), 'size': (80, 120)},
        ]
    }
    
    for i in range(num_frames):
        # Determine current room
        if i <= 40:
            room_name = 'bedroom'
            progress = i / 40
        elif i <= 80:
            room_name = 'kitchen'
            progress = (i - 41) / 39
        elif i <= 119:
            room_name = 'living'
            progress = (i - 81) / 38
        else:
            # Return to bedroom (loop closure)
            room_name = 'bedroom'
            progress = (i - 120) / 29
        
        # Room center in world coordinates
        room_center = room_centers[room_name]
        
        # Camera movement within room (slight oscillation)
        camera_offset = np.array([
            200 * np.sin(progress * 2 * np.pi),  # Left-right
            0.0,  # No vertical movement
            0.0   # No forward-back
        ])
        
        # Create synthetic frame (640x480)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 50  # Dark background
        
        # Add room-specific color tint
        if room_name == 'bedroom':
            frame[:, :] = [70, 60, 50]  # Blue-ish tint
        elif room_name == 'kitchen':
            frame[:, :] = [50, 70, 60]  # Green-ish tint
        else:
            frame[:, :] = [60, 50, 70]  # Red-ish tint
        
        # Add some texture
        noise = np.random.randint(-20, 20, (480, 640, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Create depth map
        depth = np.ones((480, 640), dtype=np.uint16) * 3000  # 3m walls
        
        # Object detections for this frame
        frame_objects = []
        
        # Render objects
        for obj in room_objects[room_name]:
            # Object position in world coordinates
            obj_world_pos = room_center + obj['pos'] + camera_offset
            
            # Simple perspective projection (assuming camera at origin looking forward)
            # This is approximate but good enough for testing
            focal_length = 500.0
            cx, cy = 320, 240
            
            # Project to image
            x_proj = int(cx + focal_length * obj_world_pos[0] / max(obj_world_pos[2], 1.0))
            y_proj = int(cy + focal_length * obj_world_pos[1] / max(obj_world_pos[2], 1.0))
            
            # Check if in frame
            if 0 <= x_proj < 640 and 0 <= y_proj < 480:
                # Draw bounding box
                w, h = obj['size']
                x1 = max(0, x_proj - w // 2)
                y1 = max(0, y_proj - h // 2)
                x2 = min(640, x_proj + w // 2)
                y2 = min(480, y_proj + h // 2)
                
                # Draw object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                
                # Update depth
                depth[y1:y2, x1:x2] = int(obj_world_pos[2])
                
                # Store detection
                frame_objects.append({
                    'class': obj['class'],
                    'bbox': (x1, y1, x2, y2),
                    'centroid_3d': obj_world_pos.copy(),  # World coordinates
                    'confidence': 0.95
                })
        
        # Create uncertainty (higher for distant objects)
        uncertainty = (depth.astype(np.float32) / 10000.0) * 50.0  # 5% at 10m
        uncertainty = np.clip(uncertainty, 10.0, 200.0)
        
        frames.append(frame)
        depths.append(depth)
        uncertainties.append(uncertainty)
        objects_per_frame.append(frame_objects)
    
    print(f"  ✓ Generated {len(frames)} frames with {sum(len(objs) for objs in objects_per_frame)} total detections")
    return frames, depths, uncertainties, objects_per_frame


def test_world_coordinate_zone_clustering(
    frames,
    depths,
    uncertainties,
    objects_per_frame
):
    """
    Test zone clustering in world coordinates.
    
    Args:
        frames: List of frames
        depths: List of depth maps
        uncertainties: List of uncertainty maps
        objects_per_frame: List of object detections
    
    Returns:
        results: Dict with statistics
    """
    print("\n" + "=" * 70)
    print("PHASE 4 WEEK 2 DAY 4: WORLD-COORDINATE ZONE CLUSTERING")
    print("=" * 70)
    
    # Configure SLAM with all Week 6 features
    config = SLAMConfig(
        enable_depth_consistency=True,
        enable_multi_frame_fusion=True,
        enable_pose_fusion=True,
        use_depth_uncertainty=True,
        use_temporal_depth_filter=True,
        use_scale_kalman=True,
        enable_loop_closure=True,
    )
    
    slam = SLAMEngine(config)
    world_tracker = WorldCoordinateTracker(slam)
    zone_manager = ZoneManager(
        mode="dense",
        min_cluster_size=3,
        min_samples=2,
        merge_distance_mm=3000.0,  # 3m for room-scale
        spatial_weight=1.0,
        embedding_weight=0.0,  # Disable for now (no embeddings)
        temporal_weight=0.01,
    )
    
    print("\n✓ Processing frames with SLAM + world coordinate tracking...")
    
    slam_successes = 0
    entity_id_counter = 0
    entity_map = {}  # Track object detections to entity IDs
    
    for i, (frame, depth, unc, objs) in enumerate(zip(frames, depths, uncertainties, objects_per_frame)):
        # SLAM tracking
        success = slam.process_frame(frame, depth, unc, i)
        if success:
            slam_successes += 1
        
        # Add objects to world coordinate tracker
        timestamp = i / 30.0  # Assume 30 FPS
        
        for obj in objs:
            # Create unique entity ID based on object class and approximate position
            obj_key = f"{obj['class']}_{int(obj['centroid_3d'][0]/1000)}_{int(obj['centroid_3d'][2]/1000)}"
            
            if obj_key not in entity_map:
                entity_map[obj_key] = f"entity_{entity_id_counter}"
                entity_id_counter += 1
            
            entity_id = entity_map[obj_key]
            
            # Add to world tracker (it will transform to world coords)
            world_tracker.add_observation(
                entity_id=entity_id,
                timestamp=timestamp,
                pos_camera=obj['centroid_3d'],  # Actually world coords in this test
                frame_idx=i
            )
            
            # Add to zone manager (should use world coords from tracker)
            zone_manager.add_observation(
                entity_id=entity_id,
                timestamp=timestamp,
                centroid_3d_mm=obj['centroid_3d'],  # World coordinates
                embedding=None,
                class_label=obj['class'],
                frame_idx=i
            )
        
        # Update zones periodically
        if (i + 1) % 20 == 0:
            zone_manager.update_zones(timestamp, frame)
            zone_stats = zone_manager.get_zone_statistics()
            print(f"  Frame {i+1:3d}: Zones={zone_stats['total_zones']}, "
                  f"Entities={len(entity_map)}, SLAM={slam_successes}/{i+1}")
    
    # Final zone update
    timestamp = len(frames) / 30.0
    zone_manager.update_zones(timestamp, frames[-1])
    
    # Get final statistics
    zone_stats = zone_manager.get_zone_statistics()
    world_stats = world_tracker.get_statistics()
    
    print("\n" + "=" * 70)
    print("RESULTS - WORLD COORDINATE ZONE CLUSTERING")
    print("=" * 70)
    
    print("\n**SLAM Performance**:")
    print(f"  Tracking rate: {slam_successes / len(frames) * 100:.1f}% ({slam_successes}/{len(frames)})")
    print(f"  Loop closures: {len(slam.loop_detector.loop_closures)}")
    
    print("\n**World Coordinate Tracking**:")
    print(f"  Total entities: {len(entity_map)}")
    print(f"  Total observations: {world_stats['total_observations']}")
    print(f"  Failed transforms: {world_stats['failed_transforms']}")
    
    print("\n**Zone Detection**:")
    print(f"  Total zones: {zone_stats['total_zones']}")
    print(f"  Expected zones: 3 (bedroom, kitchen, living room)")
    print(f"  Status: {'✅ PASS' if zone_stats['total_zones'] == 3 else '⚠️ NEEDS TUNING'}")
    
    # Analyze zone composition
    print("\n**Zone Composition**:")
    for zone_id, zone in zone_manager.zones.items():
        entity_classes = {}
        for entity_id, _ in zone.members:
            # Find object class from entity map
            for obj_key, eid in entity_map.items():
                if eid == entity_id:
                    obj_class = obj_key.split('_')[0]
                    entity_classes[obj_class] = entity_classes.get(obj_class, 0) + 1
                    break
        
        print(f"\n  {zone_id}:")
        print(f"    Centroid: ({zone.centroid_3d_mm[0]/1000:.1f}, "
              f"{zone.centroid_3d_mm[1]/1000:.1f}, "
              f"{zone.centroid_3d_mm[2]/1000:.1f}) m")
        print(f"    Entity count: {zone.entity_count}")
        print(f"    Object types: {dict(entity_classes)}")
    
    print("\n" + "=" * 70)
    
    results = {
        'slam_success_rate': slam_successes / len(frames),
        'zone_count': zone_stats['total_zones'],
        'expected_zones': 3,
        'entity_count': len(entity_map),
        'loop_closures': len(slam.loop_detector.loop_closures),
        'zones': zone_manager.zones,
    }
    
    return results


def test_semantic_zone_refinement(results):
    """
    Test semantic zone refinement.
    
    Uses object class information to validate zone detection.
    
    Args:
        results: Results from world coordinate test
    
    Returns:
        refinement_results: Dict with refinement statistics
    """
    print("\n" + "=" * 70)
    print("PHASE 4 WEEK 2 DAY 5: SEMANTIC ZONE REFINEMENT")
    print("=" * 70)
    
    zones = results['zones']
    
    print("\n✓ Analyzing semantic patterns in detected zones...")
    
    # Define semantic patterns for each room type
    room_patterns = {
        'bedroom': {'bed', 'nightstand', 'dresser', 'lamp'},
        'kitchen': {'stove', 'sink', 'refrigerator', 'dining table'},
        'living': {'couch', 'tv', 'coffee table', 'bookshelf'}
    }
    
    # Analyze each zone
    zone_classifications = {}
    
    for zone_id, zone in zones.items():
        # Extract object classes in this zone
        # Note: This would need to be implemented in the actual zone manager
        print(f"\n  Analyzing {zone_id}...")
        print(f"    Entity count: {zone.entity_count}")
        print(f"    Centroid: ({zone.centroid_3d_mm[0]/1000:.1f}, "
              f"{zone.centroid_3d_mm[1]/1000:.1f}, "
              f"{zone.centroid_3d_mm[2]/1000:.1f}) m")
        
        # TODO: Implement semantic matching
        # For now, just report that semantic analysis is needed
    
    print("\n**Semantic Refinement Status**:")
    print("  ⚠️  Semantic pattern matching needs implementation")
    print("  Current: Pure spatial clustering (DBSCAN)")
    print("  Needed: Spatial + semantic clustering")
    
    print("\n**Next Steps**:")
    print("  1. Add semantic pattern matching to ZoneManager")
    print("  2. Weight spatial similarity (70%) + semantic similarity (30%)")
    print("  3. Merge zones with matching semantic patterns")
    print("  4. Re-test on AG-50 dataset with real objects")
    
    print("\n" + "=" * 70)
    
    return {
        'semantic_analysis_needed': True,
        'zone_count': len(zones),
        'patterns_detected': {}
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Week 2 Zone Detection Test")
    parser.add_argument('--frames', type=int, default=150, help='Number of frames')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("PHASE 4 WEEK 2: ZONE DETECTION REFINEMENT")
    print("=" * 70)
    print("\nObjectives:")
    print("  • Day 4: World-coordinate zone clustering")
    print("  • Day 5: Semantic zone refinement")
    print("\nSuccess Criteria:")
    print("  • Detect 3 zones (not 5)")
    print("  • Zones stable in world coordinates")
    print("  • Zone re-identification on loop closure")
    
    # Generate test data
    frames, depths, uncertainties, objects = generate_3room_sequence_with_objects(args.frames)
    
    # Test world coordinate clustering
    results = test_world_coordinate_zone_clustering(frames, depths, uncertainties, objects)
    
    # Test semantic refinement
    semantic_results = test_semantic_zone_refinement(results)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    # Summary
    print("\n**Summary**:")
    print(f"  SLAM tracking: {results['slam_success_rate']*100:.1f}%")
    print(f"  Zone detection: {results['zone_count']} zones (target: 3)")
    print(f"  Loop closures: {results['loop_closures']}")
    print(f"  Entities tracked: {results['entity_count']}")
    
    if results['zone_count'] == 3:
        print("\n✅ Week 2 Day 4: PASS - Correct zone count!")
    else:
        print(f"\n⚠️  Week 2 Day 4: Needs tuning - Expected 3 zones, got {results['zone_count']}")
    
    if semantic_results['semantic_analysis_needed']:
        print("⚠️  Week 2 Day 5: Semantic refinement needs implementation")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
