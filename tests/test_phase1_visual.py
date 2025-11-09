#!/usr/bin/env python3
"""
Phase 1 Visual Test: Comprehensive 3D Perception Demo
======================================================

Processes a 10-second video and visualizes:
- Real YOLO object detections
- Depth maps with color coding
- Hand detections with 3D landmarks
- Object bounding boxes with 3D coordinates
- Distance annotations and movement indicators
- Occlusion states
- Terminal output with detailed stats
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from orion.perception import CameraIntrinsics
from orion.perception.perception_3d import Perception3DEngine
from ultralytics import YOLO
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def create_depth_colormap(depth_map: np.ndarray) -> np.ndarray:
    """Convert depth map to colored visualization."""
    # Normalize to 0-255
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    
    # Apply colormap (TURBO for better depth perception)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    
    return depth_colored


def draw_3d_axes(frame: np.ndarray, origin_2d: tuple, scale: float = 50):
    """Draw 3D coordinate axes at a point."""
    ox, oy = int(origin_2d[0]), int(origin_2d[1])
    
    # X-axis (red) - right
    cv2.arrowedLine(frame, (ox, oy), (ox + int(scale), oy), (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(frame, "X", (ox + int(scale) + 5, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Y-axis (green) - down
    cv2.arrowedLine(frame, (ox, oy), (ox, oy + int(scale)), (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(frame, "Y", (ox, oy + int(scale) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Z-axis (blue) - depth (represented diagonally)
    cv2.arrowedLine(frame, (ox, oy), (ox - int(scale*0.7), oy - int(scale*0.7)), (255, 0, 0), 2, tipLength=0.3)
    cv2.putText(frame, "Z", (ox - int(scale*0.7) - 20, oy - int(scale*0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def draw_hand_skeleton(frame: np.ndarray, landmarks_2d: list, frame_shape: tuple, hand_side: str = "right"):
    """Draw MediaPipe hand skeleton with connections."""
    # MediaPipe hand connections
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17),             # Palm connections
    ]
    
    h, w = frame_shape[:2]
    
    # Draw connections (bones)
    for start, end in HAND_CONNECTIONS:
        if start < len(landmarks_2d) and end < len(landmarks_2d):
            x1, y1 = landmarks_2d[start]
            x2, y2 = landmarks_2d[end]
            
            p1 = (int(x1 * w), int(y1 * h))
            p2 = (int(x2 * w), int(y2 * h))
            
            cv2.line(frame, p1, p2, (200, 200, 0), 2)  # Cyan
    
    # Draw joints
    for i, (x_norm, y_norm) in enumerate(landmarks_2d):
        x, y = int(x_norm * w), int(y_norm * h)
        
        # Joint color: fingertip=red, knuckle=green, wrist=blue
        if i in [4, 8, 12, 16, 20]:  # Fingertips
            color = (0, 0, 255)
            radius = 6
        elif i in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]:  # Knuckles
            color = (0, 255, 0)
            radius = 4
        else:  # Wrist (0)
            color = (255, 0, 0)
            radius = 8
        
        cv2.circle(frame, (x, y), radius, color, -1)
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)


def visualize_result(frame: np.ndarray, result, prev_result=None, show_3d_axes: bool = True) -> np.ndarray:
    """Create comprehensive visualization overlay with distance and motion indicators."""
    vis = frame.copy()
    
    # Draw entities with 3D info
    for entity in result.entities:
        x1, y1, x2, y2 = entity.bbox_2d_px
        
        # Color by visibility state
        if entity.visibility_state.value == "fully_visible":
            color = (0, 255, 0)  # Green
        elif entity.visibility_state.value == "partially_occluded":
            color = (0, 165, 255)  # Orange
        elif entity.visibility_state.value == "hand_occluded":
            color = (0, 0, 255)  # Red
        else:
            color = (255, 0, 0)  # Blue
        
        # Draw bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        # Draw class label
        label = f"{entity.class_label} {entity.class_confidence:.2f}"
        cv2.putText(vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw 3D info if available
        if entity.centroid_3d_mm:
            cx, cy = entity.centroid_2d_px
            X, Y, Z = entity.centroid_3d_mm
            
            # Draw center point
            cv2.circle(vis, (int(cx), int(cy)), 4, color, -1)
            
            # Draw 3D coordinates
            coord_text = f"3D: ({X:.0f}, {Y:.0f}, {Z:.0f})mm"
            cv2.putText(vis, coord_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw depth with distance in meters
            if entity.depth_mean_mm:
                depth_m = entity.depth_mean_mm / 1000.0
                depth_text = f"Distance: {depth_m:.2f}m"
                cv2.putText(vis, depth_text, (x1, y2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Calculate and show motion indicator (approaching/receding)
                if prev_result:
                    # Find matching entity in previous frame
                    prev_entity = None
                    for prev_ent in prev_result.entities:
                        if prev_ent.class_label == entity.class_label:
                            # Simple matching by class and proximity
                            prev_cx, prev_cy = prev_ent.centroid_2d_px
                            dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                            if dist < 100:  # pixels
                                prev_entity = prev_ent
                                break
                    
                    if prev_entity and prev_entity.depth_mean_mm:
                        depth_change = entity.depth_mean_mm - prev_entity.depth_mean_mm
                        if abs(depth_change) > 50:  # 5cm threshold
                            if depth_change < 0:
                                motion_text = "↗ APPROACHING"
                                motion_color = (0, 255, 255)  # Yellow
                            else:
                                motion_text = "↘ RECEDING"
                                motion_color = (255, 0, 255)  # Magenta
                            cv2.putText(vis, motion_text, (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, motion_color, 1)
            
            # Draw visibility state
            vis_text = f"{entity.visibility_state.value}"
            cv2.putText(vis, vis_text, (x1, y2 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw 3D axes at centroid
            if show_3d_axes:
                draw_3d_axes(vis, (int(cx), int(cy)), scale=30)
    
    # Draw hands with 3D landmarks and skeleton
    if result.hands:
        console.print(f"[magenta]✋ Detected {len(result.hands)} hands[/magenta]")
        
        for hand_idx, hand in enumerate(result.hands):
            try:
                # Draw hand skeleton with connections
                if hand.landmarks_2d and len(hand.landmarks_2d) >= 21:
                    draw_hand_skeleton(vis, hand.landmarks_2d, frame.shape, hand.handedness)
                    
                    # Draw hand label at wrist (landmark 0)
                    if len(hand.landmarks_2d) > 0:
                        wrist_x, wrist_y = hand.landmarks_2d[0]
                        wx = int(wrist_x * frame.shape[1])
                        wy = int(wrist_y * frame.shape[0])
                        
                        # Hand info box
                        hand_label = f"{hand.handedness} - {hand.pose.value}"
                        cv2.putText(vis, hand_label, (wx - 60, wy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                        
                        # Draw palm center
                        if hand.palm_center_3d:
                            X, Y, Z = hand.palm_center_3d
                            coord_text = f"Palm 3D: ({X:.0f}, {Y:.0f}, {Z:.0f})mm"
                            cv2.putText(vis, coord_text, (wx - 60, wy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                        
                        # Draw wrist indicator circle
                        cv2.circle(vis, (wx, wy), 10, (255, 0, 255), 2)
                        cv2.circle(vis, (wx, wy), 12, (255, 0, 255), 1)
            except Exception as e:
                console.print(f"[red]Error drawing hand {hand_idx}: {e}[/red]")
    
    return vis


def print_frame_stats(frame_num: int, result, elapsed_ms: float):
    """Print detailed frame statistics in rich format."""
    # Create main stats table
    table = Table(title=f"Frame {frame_num} Analysis", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    # Depth statistics
    if result.depth_map is not None:
        depth_mm = result.depth_map
        table.add_row("Depth Min", f"{np.min(depth_mm):.0f}mm")
        table.add_row("Depth Max", f"{np.max(depth_mm):.0f}mm")
        table.add_row("Depth Mean", f"{np.mean(depth_mm):.0f}mm")
        table.add_row("Depth Median", f"{np.median(depth_mm):.0f}mm")
        table.add_row("Depth Std", f"{np.std(depth_mm):.0f}mm")
    
    table.add_row("Entities Detected", str(len(result.entities)))
    table.add_row("Hands Detected", str(len(result.hands)))
    table.add_row("Processing Time", f"{elapsed_ms:.1f}ms")
    table.add_row("FPS", f"{1000/elapsed_ms:.1f}" if elapsed_ms > 0 else "N/A")
    
    console.print(table)
    
    # Print entity details
    if result.entities:
        console.print("\n[bold cyan]🎯 Detected Objects:[/bold cyan]")
        for i, entity in enumerate(result.entities):
            entity_info = [
                f"[yellow]{entity.class_label}[/yellow]",
                f"conf={entity.class_confidence:.2f}",
            ]
            if entity.centroid_3d_mm:
                X, Y, Z = entity.centroid_3d_mm
                entity_info.append(f"3D=({X:.0f}, {Y:.0f}, {Z:.0f})mm")
            if entity.depth_mean_mm:
                entity_info.append(f"depth={entity.depth_mean_mm:.0f}mm")
            entity_info.append(f"[{entity.visibility_state.value}]")
            
            console.print(f"  {i+1}. " + " | ".join(entity_info))
    
    # Print hand details
    if result.hands:
        console.print("\n[bold magenta]👋 Detected Hands:[/bold magenta]")
        for i, hand in enumerate(result.hands):
            X, Y, Z = hand.palm_center_3d
            console.print(
                f"  {i+1}. {hand.handedness} [{hand.pose.value}] "
                f"palm_3d=({X:.0f}, {Y:.0f}, {Z:.0f})mm "
                f"conf={hand.confidence:.2f}"
            )
    else:
        console.print("[dim magenta]👋 No hands detected (expected - test video has no hands)[/dim magenta]")
    
    console.print()


def main():
    """Run Phase 1 visual test."""
    # Configuration
    video_path = "data/examples/video.mp4"
    max_duration_sec = 60.0
    target_fps = 4  # Process at 4 FPS for realistic pipeline speed
    save_output = True
    output_dir = Path("test_results/phase1_visual")
    yolo_conf = 0.3  # Confidence threshold for YOLO detections
    
    console.print(Panel.fit(
        "[bold cyan]Phase 1 Visual Test[/bold cyan]\n"
        "Testing: Depth + Hands + 3D Perception\n"
        f"Video: {video_path}\n"
        f"Duration: {max_duration_sec}s\n"
        f"Processing Speed: {target_fps} FPS\n"
        f"[yellow]⚠️  Note: Test video has NO hands![/yellow]",
        border_style="cyan"
    ))
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        console.print(f"[red]✗ Failed to open video: {video_path}[/red]")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(int(video_fps * max_duration_sec), total_frames)
    
    # Calculate frame skip to achieve target FPS
    frame_skip = max(1, int(video_fps / target_fps))
    
    console.print(f"\n[green]✓ Video opened: {width}x{height} @ {video_fps:.2f}fps[/green]")
    console.print(f"  Processing every {frame_skip} frames → {target_fps:.1f} FPS")
    console.print(f"  Total frames to process: {max_frames // frame_skip}\n")
    
    # Initialize YOLO
    console.print("[cyan]Loading YOLO11x model...[/cyan]")
    yolo = YOLO("yolo11x.pt")
    console.print("[green]✓ YOLO loaded[/green]\n")
    
    # Initialize 3D Perception Engine
    console.print("[cyan]Initializing 3D Perception Engine...[/cyan]")
    engine = Perception3DEngine(
        enable_depth=True,
        enable_hands=True,
        enable_occlusion=True,
        depth_model="zoe",
    )
    console.print("[green]✓ 3D Perception Engine ready[/green]\n")
    
    # Create output directory
    if save_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[cyan]💾 Saving outputs to: {output_dir}[/cyan]\n")
    
    # Process frames
    frame_idx = 0
    processed_count = 0
    prev_result = None
    
    try:
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to achieve target FPS
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            timestamp = frame_idx / video_fps
            
            # Run YOLO detection
            yolo_results = yolo(frame, conf=yolo_conf, verbose=False)[0]
            
            # Convert YOLO results to our format
            yolo_detections = []
            for i, box in enumerate(yolo_results.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = yolo_results.names[cls_id]
                
                yolo_detections.append({
                    'entity_id': f'obj_{frame_idx}_{i}',
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                })
            
            # Process with Phase 1
            result = engine.process_frame(
                frame, yolo_detections, frame_idx, timestamp
            )
            
            # Print statistics
            console.rule(f"[bold]Frame {frame_idx} (Processed #{processed_count + 1})[/bold]")
            print_frame_stats(frame_idx, result, result.processing_time_ms)
            
            # Create visualizations
            vis_frame = visualize_result(frame, result, prev_result, show_3d_axes=True)
            
            # Create depth visualization
            if result.depth_map is not None:
                depth_colored = create_depth_colormap(result.depth_map)
                
                # Resize to match frame size
                if depth_colored.shape[:2] != frame.shape[:2]:
                    depth_colored = cv2.resize(depth_colored, (width, height))
                
                # Create side-by-side visualization
                combined = np.hstack([vis_frame, depth_colored])
                
                # Add labels
                cv2.putText(combined, "RGB + 3D Annotations", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined, "Depth Map (Turbo)", (width + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Save frame
                if save_output:
                    output_file = output_dir / f"frame_{frame_idx:04d}.jpg"
                    cv2.imwrite(str(output_file), combined)
                
                # Show (optional)
                cv2.imshow("Phase 1: 3D Perception", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    console.print("\n[yellow]⏸️  Stopped by user[/yellow]")
                    break
            
            # Store previous result for motion detection
            prev_result = result
            frame_idx += 1
            processed_count += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Final summary
    console.print("\n")
    avg_fps = processed_count / (frame_idx / video_fps) if frame_idx > 0 else 0
    console.print(Panel.fit(
        f"[bold green]✓ Processing Complete[/bold green]\n"
        f"Video frames: {frame_idx}/{max_frames}\n"
        f"Processed frames: {processed_count}\n"
        f"Average FPS: {avg_fps:.2f}\n"
        f"Output saved to: {output_dir}" if save_output else "No output saved",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
