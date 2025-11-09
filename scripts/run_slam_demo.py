#!/usr/bin/env python3
"""
SLAM Demo: Interactive Real-Time Visualization
==============================================

Clean, production-ready demo of Orion's SLAM capabilities with:
- Semantic SLAM (visual + object landmarks)
- Real-time trajectory visualization
- Interactive controls (keyboard + mouse)
- Current frame detections (no history clutter)

Usage:
    python scripts/run_slam_demo.py --video data/examples/video.mp4
    
Interactive Controls:
    m: Toggle SLAM mini-map
    d: Toggle depth heatmap
    h: Show/hide help
    Space: Pause/resume
    q: Quit
    Click: Inspect entity
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import colorsys
from typing import Dict, List, Optional
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.perception_3d import Perception3DEngine
from orion.perception.types import CameraIntrinsics
from orion.perception.interactive_visualizer import InteractiveVisualizer, EntityInfo
from orion.perception.visualization import TrackingVisualizer
from orion.perception.observer import FrameObserver
from orion.slam.slam_engine import OpenCVSLAM, SLAMConfig
from orion.slam.semantic_slam import SemanticSLAM
from orion.managers.model_manager import ModelManager


class SLAMDemo:
    """Clean SLAM demo with real-time visualization"""
    
    def __init__(self, video_path: str, output_path: Optional[str] = None):
        self.video_path = video_path
        self.output_path = output_path
        
        print("=" * 80)
        print("Initializing SLAM Demo")
        print("=" * 80)
        
        # Initialize model manager
        print("  [1/5] Model manager...")
        self.model_manager = ModelManager.get_instance()
        
        # Initialize YOLO detector
        print("  [2/5] YOLO detector...")
        self.yolo_model = self.model_manager.yolo
        
        # Initialize Perception3D (depth + 3D projection)
        print("  [3/5] Perception 3D engine...")
        self.perception = Perception3DEngine(
            enable_depth=True,
            enable_hands=False,  # Not needed for this demo
            enable_occlusion=False,
            depth_model="midas",  # Fast model
            device="mps"
        )
        
        # Initialize Semantic SLAM
        print("  [4/5] Semantic SLAM...")
        slam_config = SLAMConfig(
            num_features=3000,
            match_ratio_test=0.85,
            min_matches=8
        )
        base_slam = OpenCVSLAM(config=slam_config)
        self.slam = SemanticSLAM(
            base_slam=base_slam,
            use_landmarks=True,
            landmark_weight=0.3
        )
        
        # Initialize visualizers
        print("  [5/5] Visualizers...")
        from orion.perception.visualization import VisualizationConfig
        vis_config = VisualizationConfig(
            show_spatial_map=True,
            show_offscreen_banner=False,  # Not needed for SLAM demo
            show_trajectories=True,
            trajectory_length=30
        )
        self.tracking_viz = TrackingVisualizer(config=vis_config)
        self.interactive_viz = InteractiveVisualizer(
            window_name="Orion SLAM Demo"
        )
        
        # State
        self.slam_trajectory = []
        self.frame_idx = 0
        self.next_entity_id = 1
        self.entity_id_map = {}  # Map detection signatures to persistent IDs
        
        print("✓ Initialization complete\n")
    
    def _assign_entity_id(self, detection: Dict) -> int:
        """Simple entity ID assignment based on class and approximate location"""
        # Create signature: class + grid location (divide frame into 9 cells)
        x1, y1, x2, y2 = detection['bbox']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Grid cell (3x3)
        grid_x = int(cx / self.frame_width * 3)
        grid_y = int(cy / self.frame_height * 3)
        
        signature = f"{detection['class']}_{grid_x}_{grid_y}"
        
        if signature not in self.entity_id_map:
            self.entity_id_map[signature] = self.next_entity_id
            self.next_entity_id += 1
        
        return self.entity_id_map[signature]
    
    def run(self):
        """Run the SLAM demo"""
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return 1
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {Path(self.video_path).name}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Frames: {total_frames}")
        print()
        print("Interactive Controls:")
        print("  m: Toggle SLAM trajectory mini-map")
        print("  d: Toggle depth heatmap")
        print("  h: Show/hide help")
        print("  Space: Pause/resume")
        print("  q: Quit")
        print("  Click entity: Inspect details")
        print()
        
        # Output writer
        writer = None
        if self.output_path:
            output_path = Path(self.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, 
                                   (self.frame_width, self.frame_height))
        
        # Camera intrinsics
        camera_intrinsics = CameraIntrinsics.auto_estimate(
            self.frame_width, self.frame_height
        )
        
        # Processing loop
        fps_times = []
        vis_frame = None  # Initialize to avoid unbound variable
        
        try:
            while cap.isOpened():
                # Handle pause
                if not self.interactive_viz.paused:
                    frame_start = time.time()
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    self.frame_idx += 1
                    timestamp = self.frame_idx / fps
                    
                    # Get toggle states
                    toggles = self.interactive_viz.get_toggle_states()
                    
                    # 1. YOLO Detection (current frame only!)
                    yolo_raw = self.yolo_model(frame, conf=0.3, verbose=False)[0]
                    yolo_detections = []
                    for box in yolo_raw.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = yolo_raw.names[cls_id]
                        
                        yolo_detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'class': cls_name,
                            'confidence': conf,
                            'entity_id': None  # Will be assigned below
                        })
                    
                    # Assign entity IDs
                    for det in yolo_detections:
                        det['entity_id'] = self._assign_entity_id(det)
                    
                    # 2. Perception3D (depth + 3D backprojection)
                    perception_result = self.perception.process_frame(
                        frame=frame,
                        yolo_detections=yolo_detections,
                        frame_number=self.frame_idx,
                        timestamp=timestamp,
                        camera_intrinsics=camera_intrinsics
                    )
                    
                    # 3. SLAM tracking with semantic landmarks
                    slam_pose = self.slam.track(
                        frame=frame,
                        timestamp=timestamp,
                        frame_idx=self.frame_idx,
                        yolo_detections=yolo_detections
                    )
                    
                    # Store trajectory
                    if slam_pose is not None:
                        position = slam_pose[:3, 3]
                        self.slam_trajectory.append(position)
                    
                    # 4. Build entity info for interactive viz (CURRENT FRAME ONLY)
                    entities = {}
                    for entity_3d in perception_result.entities:
                        # Convert entity_id to int if it's a string
                        eid = entity_3d.entity_id
                        if isinstance(eid, str):
                            try:
                                eid = int(eid)
                            except (ValueError, TypeError):
                                eid = hash(eid) % 1000000  # Fallback to hash
                        
                        entities[eid] = EntityInfo(
                            entity_id=eid,
                            class_name=entity_3d.class_label,
                            bbox=entity_3d.bbox_2d_px,
                            confidence=entity_3d.class_confidence,
                            depth_mm=entity_3d.depth_mean_mm,
                            zone_id=None,  # Not using zones in this demo
                            world_pos=entity_3d.centroid_3d_mm
                        )
                    
                    # Update interactive visualizer
                    self.interactive_viz.update_entities(entities)
                    
                    # 5. Render visualization
                    vis_frame = self._render_frame(
                        frame=frame,
                        entities=entities,
                        perception_result=perception_result,
                        slam_pose=slam_pose,
                        toggles=toggles
                    )
                    
                    # Write output
                    if writer:
                        writer.write(vis_frame)
                    
                    # Calculate FPS
                    frame_time = time.time() - frame_start
                    fps_times.append(frame_time)
                    if len(fps_times) > 30:
                        fps_times.pop(0)
                    current_fps = 1.0 / (sum(fps_times) / len(fps_times))
                    
                    # Progress
                    if self.frame_idx % 30 == 0:
                        print(f"  Frame {self.frame_idx}/{total_frames} "
                              f"({self.frame_idx/total_frames*100:.1f}%) "
                              f"| FPS: {current_fps:.2f} "
                              f"| SLAM: {len(self.slam_trajectory)} poses")
                
                else:
                    # Paused - just redraw current frame
                    if vis_frame is not None:
                        vis_frame = self.interactive_viz.draw_overlays(vis_frame)
                
                # Show and handle input (only if we have a frame)
                if vis_frame is not None:
                    if not self.interactive_viz.show(vis_frame):
                        print("\nUser quit")
                        break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            self.interactive_viz.close()
            cv2.destroyAllWindows()
        
        # Final statistics
        self._print_statistics()
        
        return 0
    
    def _render_frame(
        self,
        frame: np.ndarray,
        entities: Dict[int, EntityInfo],
        perception_result,
        slam_pose: Optional[np.ndarray],
        toggles: Dict[str, bool]
    ) -> np.ndarray:
        """Render clean visualization of CURRENT frame only"""
        vis_frame = frame.copy()
        
        # Set frame dimensions for tracking visualizer (first frame)
        if self.tracking_viz.frame_width == 0:
            self.tracking_viz.frame_height, self.tracking_viz.frame_width = frame.shape[:2]
        
        # Draw depth heatmap (if enabled)
        if toggles['depth'] and perception_result.depth_map is not None:
            depth_map = perception_result.depth_map
            # Normalize depth map
            depth_normalized = np.zeros_like(depth_map)
            depth_normalized = cv2.normalize(depth_map, depth_normalized, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_TURBO)
            vis_frame = cv2.addWeighted(vis_frame, 0.7, depth_colored, 0.3, 0)
        
        # Draw bounding boxes for CURRENT detections only (with colorful entity-specific colors!)
        for eid, einfo in entities.items():
            x1, y1, x2, y2 = einfo.bbox
            
            # Get entity-specific color (consistent across frames)
            if eid not in self.tracking_viz.id_colors:
                # Generate unique color for this entity ID
                hue = (eid * 137.508) % 360  # Golden angle for distinctiveness
                rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 0.95)
                self.tracking_viz.id_colors[eid] = (
                    int(rgb[2] * 255),  # BGR
                    int(rgb[1] * 255),
                    int(rgb[0] * 255)
                )
            
            color = self.tracking_viz.id_colors[eid]
            
            # Bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
            
            # Label with depth
            label_parts = [f"#{eid}", einfo.class_name]
            if einfo.depth_mm:
                label_parts.append(f"{einfo.depth_mm/1000:.2f}m")
            label = " ".join(label_parts)
            
            # Background for text (same color as bbox)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_frame, (x1, y1 - text_h - 10), (x1 + text_w + 8, y1), color, -1)
            
            # Text (white for readability)
            cv2.putText(vis_frame, label, (x1 + 4, y1 - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw centroid
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(vis_frame, (cx, cy), 5, color, -1)
            cv2.circle(vis_frame, (cx, cy), 6, (255, 255, 255), 2)
        
        # Draw SLAM status
        if slam_pose is not None:
            status = "TRACKING"
            color = (0, 255, 0)
            cv2.putText(vis_frame, f"SLAM: {status}", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Check if landmarks contributed (get stats)
            stats = self.slam.get_statistics()
            landmark_rescues = stats.get('landmark_only', 0)
            if landmark_rescues > 0:
                cv2.putText(vis_frame, f"Semantic Rescues: {landmark_rescues}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            status = "LOST"
            color = (0, 0, 255)
            cv2.putText(vis_frame, f"SLAM: {status}", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw frame info
        info_y = vis_frame.shape[0] - 40
        cv2.putText(vis_frame, f"Frame {self.frame_idx} | Entities: {len(entities)} | FPS: {1.0/0.6:.1f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw SLAM trajectory (if enabled and available)
        spatial_map = None
        if toggles['slam_minimap'] and len(self.slam_trajectory) > 1:
            vis_frame, minimap = self.tracking_viz.draw_slam_trajectory(
                frame=vis_frame,
                slam_trajectory=self.slam_trajectory,
                current_pose=slam_pose,
                minimap=True
            )
            
            # Show mini-map in separate window
            if minimap is not None:
                cv2.imshow("SLAM Trajectory", minimap)
        
        # Create spatial map if enabled (show entity positions top-down)
        if toggles.get('spatial_map', True) and len(entities) > 0:
            spatial_map = self._create_spatial_map(entities, slam_pose)
            if spatial_map is not None:
                cv2.imshow("Spatial Map (Top-Down)", spatial_map)
        
        # Add interactive overlays (status bar, entity panels, hover highlights)
        vis_frame = self.interactive_viz.draw_overlays(vis_frame)
        
        return vis_frame
    
    def _create_spatial_map(
        self,
        entities: Dict[int, EntityInfo],
        slam_pose: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Create top-down spatial map showing entity positions"""
        map_size = 400
        spatial_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        
        # Draw grid
        grid_spacing = 50
        for i in range(0, map_size, grid_spacing):
            cv2.line(spatial_map, (i, 0), (i, map_size), (40, 40, 40), 1)
            cv2.line(spatial_map, (0, i), (map_size, i), (40, 40, 40), 1)
        
        # Center coordinates (camera position)
        center_x, center_y = map_size // 2, map_size - 50
        
        # Draw camera
        cv2.circle(spatial_map, (center_x, center_y), 10, (255, 255, 255), 2)
        cv2.line(spatial_map, (center_x - 8, center_y), (center_x + 8, center_y), (255, 255, 255), 2)
        cv2.line(spatial_map, (center_x, center_y - 8), (center_x, center_y + 8), (255, 255, 255), 2)
        
        # Draw entities
        range_mm = 3000.0  # ±3 meters
        for eid, einfo in entities.items():
            if einfo.world_pos:
                x, y, z = einfo.world_pos
                
                # Convert 3D position to 2D map coordinates (top-down view)
                map_x = int(center_x + (x / range_mm) * (map_size / 2))
                map_y = int(center_y - (z / range_mm) * (map_size / 2))
                
                # Clip to map bounds
                map_x = max(0, min(map_size - 1, map_x))
                map_y = max(0, min(map_size - 1, map_y))
                
                # Get entity color
                color = self.tracking_viz.id_colors.get(eid, (100, 100, 100))
                
                # Draw line from camera to entity
                cv2.line(spatial_map, (center_x, center_y), (map_x, map_y), color, 1)
                
                # Draw entity circle
                cv2.circle(spatial_map, (map_x, map_y), 8, color, -1)
                cv2.circle(spatial_map, (map_x, map_y), 9, (255, 255, 255), 1)
                
                # Draw ID
                cv2.putText(spatial_map, str(eid), (map_x + 12, map_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Title
        cv2.putText(spatial_map, "Spatial Map (Top-Down)", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(spatial_map, f"Range: +/-{range_mm/1000:.1f}m", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return spatial_map
    
    def _print_statistics(self):
        """Print final statistics"""
        print()
        print("=" * 80)
        print("SLAM Demo Complete")
        print("=" * 80)
        print(f"Frames processed: {self.frame_idx}")
        if self.output_path:
            print(f"Output saved: {self.output_path}")
        print()
        
        # SLAM statistics
        stats = self.slam.get_statistics()
        print("Semantic SLAM Statistics:")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Visual success: {stats['visual_success']} "
              f"({stats['visual_success']/max(stats['total_frames'], 1)*100:.1f}%)")
        print(f"  Landmark success: {stats['landmark_success']} "
              f"({stats['landmark_success']/max(stats['total_frames'], 1)*100:.1f}%)")
        print(f"  Fused poses: {stats['fused_poses']}")
        print(f"  Visual only: {stats['visual_only']}")
        print(f"  Landmark rescues: {stats['landmark_only']}")
        print(f"  Trajectory: {len(self.slam_trajectory)} poses")
        
        if len(self.slam_trajectory) > 1:
            # Calculate distance
            trajectory = np.array(self.slam_trajectory)
            distances = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
            total_distance = np.sum(distances)
            print(f"  Distance traveled: {total_distance/1000:.2f} meters")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="SLAM Demo: Interactive real-time visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive Controls:
  m         Toggle SLAM trajectory mini-map
  d         Toggle depth heatmap
  h         Show/hide help overlay
  Space     Pause/resume playback
  q         Quit
  Click     Inspect entity details

Example:
  python scripts/run_slam_demo.py --video data/examples/video.mp4
  python scripts/run_slam_demo.py --video data/examples/video.mp4 --output results/demo.mp4
        """
    )
    parser.add_argument("--video", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to output video (optional)")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    # Run demo
    demo = SLAMDemo(video_path=args.video, output_path=args.output)
    return demo.run()


if __name__ == "__main__":
    sys.exit(main())
