"""
Phase 3 Test: Spatial Zones + Enhanced CIS

Tests the new Phase 3 features:
- Scene classification
- Spatial zone detection (HDBSCAN clustering)
- Enhanced CIS with 3D and hand signals
- Visualization with zone overlays

Usage:
    python test_phase3_zones.py --video data/examples/video.mp4 --max-frames 100

Author: Orion Research Team
Date: November 2025
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys
import time
from typing import List, Dict

# Add orion to path
sys.path.insert(0, str(Path(__file__).parent))

from orion.perception.tracking import EntityTracker3D, TrackingConfig
from orion.perception.visualization import TrackingVisualizer, VisualizationConfig
from orion.perception.perception_3d import Perception3DEngine
from orion.managers.model_manager import ModelManager
from orion.semantic.zone_manager import ZoneManager
from orion.semantic.scene_classifier import SceneClassifier
from orion.semantic.cis_scorer_3d import CausalInfluenceScorer3D


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3 Test: Spatial Zones + CIS"
    )
    
    parser.add_argument(
        "--video", type=str, default="data/examples/video.mp4",
        help="Path to input video"
    )
    
    parser.add_argument(
        "--output", type=str, default="test_results/phase3_zones_visual.mp4",
        help="Path to output video"
    )
    
    parser.add_argument(
        "--confidence", type=float, default=0.6,
        help="YOLO confidence threshold"
    )
    
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum frames to process"
    )
    
    parser.add_argument(
        "--zone-mode", type=str, default="dense", choices=["dense", "sparse"],
        help="Zone detection mode (dense=indoor, sparse=outdoor)"
    )
    
    parser.add_argument(
        "--visualize", action="store_true", default=True,
        help="Show visualization"
    )
    
    return parser.parse_args()


def draw_zones_on_spatial_map(
    spatial_map: np.ndarray,
    zones: Dict,
    frame_width: int,
    frame_height: int
) -> np.ndarray:
    """
    Draw zone boundaries on spatial map.
    
    Args:
        spatial_map: Spatial map image
        zones: Dict of zone_id → Zone
        frame_width: Video frame width
        frame_height: Video frame height
    
    Returns:
        Spatial map with zone overlays
    """
    map_h, map_w = spatial_map.shape[:2]
    center_x, center_y = map_w // 2, map_h - 30
    range_mm = 3000.0
    
    for zone_id, zone in zones.items():
        # Convert zone centroid to map coordinates
        x_mm, y_mm, z_mm = zone.centroid_3d_mm
        
        map_x = int(center_x + (x_mm / range_mm) * (map_w / 2))
        map_y = int(center_y - (z_mm / range_mm) * (map_h - 50))
        
        # Clip to bounds
        map_x = np.clip(map_x, 10, map_w - 10)
        map_y = np.clip(map_y, 10, map_h - 40)
        
        # Draw zone circle (larger than entity circles)
        cv2.circle(spatial_map, (map_x, map_y), 20, zone.color, 2)
        cv2.circle(spatial_map, (map_x, map_y), 3, zone.color, -1)
        
        # Draw zone label
        label_text = f"Z{zone_id.split('_')[-1]}: {zone.label}"
        cv2.putText(
            spatial_map, label_text, (map_x + 25, map_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1
        )
    
    return spatial_map


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("PHASE 3 TEST: Spatial Zones + Enhanced CIS")
    print("="*80)
    print(f"  Video: {args.video}")
    print(f"  Zone Mode: {args.zone_mode}")
    print(f"  Confidence: {args.confidence}")
    print(f"  Max Frames: {args.max_frames or 'All'}")
    print("="*80 + "\n")
    
    # Check video
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    # Create output dir
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}\n")
    
    # Initialize models
    print("Loading models...")
    model_manager = ModelManager.get_instance()
    yolo = model_manager.yolo
    clip_model = model_manager.clip
    print(f"  ✓ YOLO + CLIP loaded\n")
    
    yolo_classes = list(yolo.names.values()) if hasattr(yolo, 'names') else []
    
    # Initialize tracking
    print("Initializing Phase 2 tracking...")
    tracking_config = TrackingConfig(
        max_distance_pixels=150.0,
        max_distance_3d_mm=1500.0,
        ttl_frames=30,
        reid_window_frames=90,
    )
    tracker = EntityTracker3D(config=tracking_config, yolo_classes=yolo_classes)
    print(f"  ✓ Tracker initialized\n")
    
    # Initialize 3D perception
    print("Initializing 3D perception...")
    perception_3d = Perception3DEngine(depth_model="midas")
    print(f"  ✓ 3D perception initialized\n")
    
    # Initialize Phase 3 components
    print("Initializing Phase 3 components...")
    
    # Scene classifier
    scene_classifier = SceneClassifier(clip_model=clip_model)
    print(f"  ✓ Scene classifier initialized")
    
    # Zone manager
    zone_manager = ZoneManager(
        mode=args.zone_mode,
        min_cluster_size=8,  # Smaller for short videos
        min_samples=3,
        merge_distance_mm=2500.0,
    )
    print(f"  ✓ Zone manager initialized ({args.zone_mode} mode)")
    
    # Enhanced CIS scorer
    cis_scorer = CausalInfluenceScorer3D()
    print(f"  ✓ Enhanced CIS scorer initialized\n")
    
    # Initialize visualizer
    vis_config = VisualizationConfig(
        show_spatial_map=True,
        show_offscreen_banner=True,
    )
    visualizer = TrackingVisualizer(config=vis_config)
    print(f"  ✓ Visualizer initialized\n")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 4.0, (width, height))
    
    # Processing loop
    print("Processing video...\n")
    
    frame_number = 0
    processed_frames = 0
    start_time = time.time()
    prev_frame = None
    current_scene_type = None
    
    # Process at 1 FPS for efficiency
    process_interval = int(fps)  # Process every Nth frame (e.g., every 30 frames for 30fps video)
    zone_update_interval = process_interval  # Update zones at 1Hz
    scene_classification_interval = process_interval * 30  # Classify scene every 30 seconds
    
    print(f"\nProcessing at ~1 FPS (every {process_interval} frames)")
    print(f"Zone updates: every {zone_update_interval} frames (~1 Hz)")
    print(f"Scene classification: every {scene_classification_interval} frames (~30 seconds)\n")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to achieve ~1 FPS processing
            if frame_number % process_interval != 0:
                frame_number += 1
                continue
            
            if args.max_frames and processed_frames >= args.max_frames:
                print(f"\nReached max frames limit: {args.max_frames}")
                break
            
            timestamp = frame_number / fps
            
            # Scene classification (every 30 seconds)
            if frame_number % scene_classification_interval == 0:
                scene_classification = scene_classifier.classify_detailed(frame, prev_frame)
                current_scene_type = scene_classification.scene_type.value
                print(f"[Frame {frame_number}] Scene: {current_scene_type} "
                      f"(conf: {scene_classification.confidence:.2f}, "
                      f"indoor: {scene_classification.is_indoor})")
            
            # YOLO detection
            results = yolo(frame, conf=args.confidence, verbose=False)
            
            # Extract detections with CLIP embeddings
            detections = []
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = yolo_classes[cls] if cls < len(yolo_classes) else "unknown"
                    
                    if conf < args.confidence:
                        continue
                    
                    # Extract appearance embedding
                    x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
                    x2_int, y2_int = min(width, int(x2)), min(height, int(y2))
                    crop = frame[y1_int:y2_int, x1_int:x2_int]
                    
                    embedding = None
                    if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
                        crop_resized = cv2.resize(crop, (224, 224))
                        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                        embedding = clip_model.encode_image(crop_rgb, normalize=True)
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'class_name': class_name,
                        'confidence': conf,
                        'centroid_2d': np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
                        'appearance_embedding': embedding,
                    }
                    
                    detections.append(detection)
            
            # Get depth map
            depth_map = None
            if detections:
                depth_map, _ = perception_3d.depth_estimator.estimate(frame)
                
                # Add 3D coordinates
                from orion.perception.types import CameraIntrinsics
                from orion.perception.camera_intrinsics import backproject_bbox
                camera_intrinsics = CameraIntrinsics.auto_estimate(width, height)
                
                for det in detections:
                    bbox_3d_info = backproject_bbox(det['bbox'], depth_map, camera_intrinsics)
                    det['centroid_3d_mm'] = np.array(bbox_3d_info['centroid_3d'])
            
            # Track entities
            tracks = tracker.track_frame(detections, frame_number, timestamp)
            
            # Add observations to zone manager
            for track in tracks:
                if track.centroid_3d_mm is not None:
                    zone_manager.add_observation(
                        entity_id=str(track.entity_id),
                        timestamp=timestamp,
                        centroid_3d_mm=track.centroid_3d_mm,
                        embedding=track.appearance_embedding,
                        class_label=track.most_likely_class,
                        frame_idx=frame_number
                    )
            
            # Update zones periodically (every 1 second at 1Hz)
            if frame_number % zone_update_interval == 0:
                zone_manager.update_zones(timestamp, frame)
                
                zone_stats = zone_manager.get_zone_statistics()
                if zone_stats['total_zones'] > 0:
                    print(f"[Frame {frame_number}] Zones: {zone_stats['total_zones']} "
                          f"({zone_stats['zone_types']})")
            
            # Visualize
            vis_frame, spatial_map = visualizer.visualize_frame(
                frame, tracks, depth_map, frame_number
            )
            
            # Draw zones on spatial map
            if spatial_map is not None and len(zone_manager.zones) > 0:
                spatial_map = draw_zones_on_spatial_map(
                    spatial_map, zone_manager.zones, width, height
                )
            
            # Write output
            out.write(vis_frame)
            
            # Show visualization
            if args.visualize:
                display_frame = vis_frame
                if width > 1280:
                    scale = 1280 / width
                    new_width = 1280
                    new_height = int(height * scale)
                    display_frame = cv2.resize(vis_frame, (new_width, new_height))
                
                cv2.imshow('Phase 3: Zones + CIS', display_frame)
                
                if spatial_map is not None:
                    cv2.imshow('Spatial Map with Zones', spatial_map)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nUser requested stop")
                    break
                elif key == ord(' '):
                    print("\nPaused. Press any key to continue...")
                    cv2.waitKey(0)
            
            frame_number += 1
            processed_frames += 1
            prev_frame = frame.copy()
            
            # Progress
            if processed_frames % 10 == 0:
                elapsed = time.time() - start_time
                fps_actual = processed_frames / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_number}/{total_frames} | FPS: {fps_actual:.1f} | "
                      f"Tracks: {len(tracks)} | Zones: {len(zone_manager.zones)}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        fps_actual = processed_frames / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"  Processed frames: {processed_frames}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Average FPS: {fps_actual:.2f}")
        print(f"\nPhase 3 Statistics:")
        
        zone_stats = zone_manager.get_zone_statistics()
        print(f"  Total zones discovered: {zone_stats['total_zones']} (camera-relative viewpoints)")
        print(f"  Zone types: {zone_stats['zone_types']}")
        print(f"  Observations processed: {zone_stats['total_observations']}")
        print(f"\n  NOTE: Multiple zones may represent the same physical room from different")
        print(f"        camera viewpoints. Monocular depth creates camera-relative coordinates.")
        print(f"        For accurate room-level mapping, visual SLAM integration is needed.")
        
        tracker_stats = tracker.get_statistics()
        print(f"\nTracking Statistics:")
        print(f"  Total entities: {tracker_stats.get('total_entities_seen', 0)}")
        print(f"  Re-identifications: {tracker_stats.get('reidentifications', 0)}")
        
        print(f"\n✓ Output saved to: {output_path}")
        print("="*80)
        print("="*80)


if __name__ == "__main__":
    main()
