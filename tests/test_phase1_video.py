#!/usr/bin/env python3
"""
Phase 1 Video Processing Test
Runs depth estimation, hand tracking, and 3D perception on a test video.
Displays outputs frame-by-frame with visualizations.
"""

import argparse
import time
from pathlib import Path
import sys

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception import (
    PerceptionEngine,
    PerceptionConfig,
    CameraIntrinsics,
    EntityState,
)
from orion.perception.config import (
    DepthConfig,
    HandTrackingConfig,
    OcclusionConfig,
    CameraConfig,
)


def visualize_depth(depth_map: np.ndarray, title: str = "Depth Map") -> np.ndarray:
    """Convert depth map to colorized heatmap for visualization."""
    # Normalize to 0-255
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    
    # Apply colormap (TURBO for better perception)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    
    return depth_colored


def draw_hand_landmarks(frame: np.ndarray, hands: list, intrinsics) -> np.ndarray:
    """Draw hand landmarks and poses on the frame."""
    annotated = frame.copy()
    
    for hand in hands:
        # Draw 2D landmarks
        for i, (x, y) in enumerate(hand.landmarks_2d):
            x_px, y_px = int(x), int(y)
            # Palm center is landmark 0, draw it bigger
            if i == 0:
                cv2.circle(annotated, (x_px, y_px), 8, (0, 255, 0), -1)
            else:
                cv2.circle(annotated, (x_px, y_px), 4, (255, 0, 0), -1)
        
        # Draw connections (simplified hand skeleton)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17),  # Palm
        ]
        
        for start, end in connections:
            if start < len(hand.landmarks_2d) and end < len(hand.landmarks_2d):
                x1, y1 = hand.landmarks_2d[start]
                x2, y2 = hand.landmarks_2d[end]
                cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        
        # Draw hand info text
        palm_x, palm_y = hand.landmarks_2d[0]
        text = f"{hand.handedness} {hand.pose.value}"
        cv2.putText(annotated, text, (int(palm_x) + 10, int(palm_y) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw 3D position
        if hand.palm_center_3d is not None:
            x_mm, y_mm, z_mm = hand.palm_center_3d
            pos_text = f"3D: ({x_mm:.0f}, {y_mm:.0f}, {z_mm:.0f})mm"
            cv2.putText(annotated, pos_text, (int(palm_x) + 10, int(palm_y) + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return annotated


def draw_entities(frame: np.ndarray, entities: list) -> np.ndarray:
    """Draw entity bounding boxes and 3D info."""
    annotated = frame.copy()
    
    for entity in entities:
        # Draw bounding box
        x1, y1, x2, y2 = map(int, entity.bbox)
        
        # Color based on visibility state
        color_map = {
            'fully_visible': (0, 255, 0),
            'partially_occluded': (255, 255, 0),
            'hand_occluded': (255, 165, 0),
            'off_screen': (128, 128, 128),
        }
        color = color_map.get(entity.visibility_state, (255, 255, 255))
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{entity.class_name} ({entity.confidence:.2f})"
        cv2.putText(annotated, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw 3D centroid if available
        if entity.centroid_3d_mm is not None:
            x_mm, y_mm, z_mm = entity.centroid_3d_mm
            pos_text = f"3D: {z_mm:.0f}mm"
            cv2.putText(annotated, pos_text, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw visibility state
        cv2.putText(annotated, entity.visibility_state, (x1, y2 + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return annotated


def print_frame_info(result, frame_time: float):
    """Print detailed frame processing information."""
    print(f"\n{'='*80}")
    print(f"Frame {result.frame_number} (t={result.timestamp:.3f}s)")
    print(f"{'='*80}")
    
    # Depth statistics
    if result.depth_map is not None:
        depth_mm = result.depth_map
        print(f"\n📏 Depth Map:")
        print(f"  Min: {np.min(depth_mm):.0f}mm, Max: {np.max(depth_mm):.0f}mm, Mean: {np.mean(depth_mm):.0f}mm")
        print(f"  Median: {np.median(depth_mm):.0f}mm, Std: {np.std(depth_mm):.0f}mm")
    
    # Hand detections
    print(f"\n👋 Hands: {len(result.hands)} detected")
    for i, hand in enumerate(result.hands):
        palm = hand.palm_center_3d if hand.palm_center_3d is not None else (0, 0, 0)
        print(f"  - Hand {i}: {hand.handedness}, {hand.pose.value}, "
              f"palm at ({palm[0]:.0f}, {palm[1]:.0f}, {palm[2]:.0f})mm, "
              f"confidence={hand.confidence:.2f}")
    
    # Entity states
    print(f"\n🎯 Entities: {len(result.entities)} detected")
    for entity in result.entities:
        centroid = entity.centroid_3d_mm if entity.centroid_3d_mm is not None else (0, 0, 0)
        depth_mean = entity.depth_mean_mm if entity.depth_mean_mm is not None else 0
        print(f"  - {entity.class_name}: ({centroid[0]:.0f}, {centroid[1]:.0f}, {centroid[2]:.0f})mm, "
              f"depth={depth_mean:.0f}mm, {entity.visibility_state.value}, conf={entity.confidence:.2f}")
    
    # Performance metrics
    print(f"\n⚡ Performance:")
    print(f"  Processing: {result.processing_time_ms:.1f}ms")
    
    total_time_ms = frame_time * 1000
    fps = 1.0 / frame_time if frame_time > 0 else 0
    print(f"  Total (incl. viz): {total_time_ms:.1f}ms ({fps:.1f} FPS)")


def process_video(
    video_path: str,
    max_frames: int = 30,
    show_viz: bool = True,
    save_output: bool = False,
    output_dir: str = "test_results/phase1_video"
):
    """Process video with Phase 1 perception engine."""
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*80}")
    print(f"Video Info: {video_path}")
    print(f"{'='*80}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f}s")
    print(f"Processing: {min(max_frames, total_frames)} frames")
    
    # Initialize perception engine
    print(f"\n{'='*80}")
    print("Initializing Phase 1 Perception Engine...")
    print(f"{'='*80}")
    
    config = PerceptionConfig(
        depth=DepthConfig(
            model_name="zoe",  # ZoeDepth (optimized for egocentric)
            device=None,  # Auto-detect
            half_precision=False,
        ),
        hand_tracking=HandTrackingConfig(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2,
        ),
        occlusion=OcclusionConfig(
            depth_margin_mm=100.0,
            occlusion_threshold=0.3,
        ),
        camera=CameraConfig(
            width=width,
            height=height,
            auto_estimate=True,
        ),
        enable_hands=True,  # Hand tracking enabled!
    )
    
    try:
        engine = PerceptionEngine(config)
        print("✅ Perception engine initialized successfully")
        print(f"   Depth model: {config.depth.model_name}")
        print(f"   Hand tracking: {'enabled' if config.enable_hands else 'disabled'}")
    except Exception as e:
        print(f"❌ Error initializing perception engine: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get camera intrinsics from config
    intrinsics = CameraIntrinsics.auto_estimate(width, height)
    print(f"\n📷 Camera Intrinsics (auto-estimated):")
    print(f"   fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
    print(f"   cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}")
    
    # Create output directory if saving
    if save_output:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\n💾 Saving outputs to: {output_path}")
    
    # Process frames
    frame_id = 0
    total_time = 0.0
    
    try:
        while frame_id < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_id / fps
            
            # Mock entities (empty for now - would normally come from YOLO)
            # In real pipeline, these would be detected objects
            entities = []
            
            # Process frame
            start_time = time.time()
            result = engine.process_frame(frame, entities, frame_id, timestamp)
            frame_time = time.time() - start_time
            total_time += frame_time
            
            # Print frame info
            print_frame_info(result, frame_time)
            
            # Visualization
            if show_viz or save_output:
                # Create visualization grid
                vis_frame = frame.copy()
                
                # Draw hands
                if result.hands:
                    vis_frame = draw_hand_landmarks(vis_frame, result.hands, intrinsics)
                
                # Draw entities (if any)
                if result.entities:
                    vis_frame = draw_entities(vis_frame, result.entities)
                
                # Create depth visualization
                if result.depth_map is not None:
                    depth_viz = visualize_depth(result.depth_map)
                    
                    # Resize depth to match frame size if needed
                    if depth_viz.shape[:2] != vis_frame.shape[:2]:
                        depth_viz = cv2.resize(depth_viz, (vis_frame.shape[1], vis_frame.shape[0]))
                    
                    # Stack side by side
                    combined = np.hstack([vis_frame, depth_viz])
                    
                    # Add labels
                    cv2.putText(combined, "RGB + Annotations", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(combined, "Depth Map", (width + 10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    combined = vis_frame
                
                # Show visualization
                if show_viz:
                    cv2.imshow("Phase 1 Perception", combined)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⏸️  User requested stop (pressed 'q')")
                        break
                    elif key == ord(' '):
                        print("\n⏸️  Paused (press any key to continue)")
                        cv2.waitKey(0)
                
                # Save frame
                if save_output:
                    output_file = output_path / f"frame_{frame_id:04d}.jpg"
                    cv2.imwrite(str(output_file), combined)
            
            frame_id += 1
        
    finally:
        cap.release()
        if show_viz:
            cv2.destroyAllWindows()
    
    # Print summary
    print(f"\n{'='*80}")
    print("Processing Summary")
    print(f"{'='*80}")
    print(f"Frames processed: {frame_id}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {frame_id / total_time:.2f}")
    print(f"Average time per frame: {(total_time / frame_id * 1000):.1f}ms")
    
    if save_output:
        print(f"\n💾 Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test Phase 1 perception on video")
    parser.add_argument(
        "--video",
        type=str,
        default="data/examples/video_short.mp4",
        help="Path to input video",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization window",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output frames to disk",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results/phase1_video",
        help="Output directory for saved frames",
    )
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        print(f"❌ Error: Video not found: {args.video}")
        print("\nAvailable test videos:")
        print("  - data/examples/video.mp4")
        print("  - data/examples/video_short.mp4")
        return
    
    # Process video
    process_video(
        video_path=args.video,
        max_frames=args.max_frames,
        show_viz=not args.no_viz,
        save_output=args.save,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
