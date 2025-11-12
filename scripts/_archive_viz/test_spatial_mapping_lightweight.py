#!/usr/bin/env python3
"""
Lightweight Spatial Mapping Test - No Heavy Orion Imports

Outputs spatial mapping visualizations:
1. Camera intrinsics
2. Depth heatmaps
3. YOLO detections
4. Depth distributions
5. 3D point clouds (back-projection)
6. 2D reprojections
7. SLAM poses
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import sys
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import time

# Minimal imports - direct paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

OUTPUT_DIR = Path("spatial_mapping_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"📁 Output directory: {OUTPUT_DIR.absolute()}/")


def auto_estimate_intrinsics(width: int, height: int) -> dict:
    """Estimate camera intrinsics based on resolution"""
    # Assume 50 degree FOV
    fov = 50  # degrees
    fx = fy = (width / 2) / np.tan(np.deg2rad(fov / 2))
    cx = width / 2
    cy = height / 2
    
    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'width': width,
        'height': height,
    }


def load_depth_anything_v2(model_size: str = 'small', device: str = 'mps'):
    """Load Depth Anything V2 from HuggingFace"""
    print(f"[Depth] Loading Depth Anything V2 ({model_size}) from HuggingFace...")
    
    try:
        import torch
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        
        # Map model sizes to HuggingFace model IDs
        model_id_map = {
            'small': 'depth-anything/Depth-Anything-V2-small-hf',
            'base': 'depth-anything/Depth-Anything-V2-base-hf',
            'large': 'depth-anything/Depth-Anything-V2-large-hf',
        }
        
        model_id = model_id_map.get(model_size, 'depth-anything/Depth-Anything-V2-small-hf')
        
        print(f"[Depth] Loading processor and model from {model_id}...")
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        model.to(device)
        model.eval()
        
        print(f"[Depth] ✓ Depth Anything V2 ({model_size}) loaded successfully")
        return {'model': model, 'processor': processor, 'device': device}
        
    except Exception as e:
        print(f"[Depth] ✗ Failed to load: {str(e)[:150]}")
        print(f"[Depth] Using dummy depth instead")
        return None


def estimate_depth_dummy(frame: np.ndarray) -> np.ndarray:
    """Generate dummy depth map for testing (when model fails to load)"""
    h, w = frame.shape[:2]
    # Create realistic-looking depth with some variation
    depth = np.zeros((h, w), dtype=np.float32)
    # Add horizontal gradient (left=near, right=far) and vertical variation
    for y in range(h):
        for x in range(w):
            # Horizontal gradient: 1m to 5m
            dist_x = 1.0 + 4.0 * (x / w)
            # Vertical variation: objects at different heights
            dist_y = 0.5 + 1.5 * np.sin(np.pi * y / h)
            depth[y, x] = (dist_x + dist_y) * 500  # Convert to mm, scale down slightly
    return depth


@torch.no_grad()
def estimate_depth(frame: np.ndarray, model_dict, device: str = 'mps') -> tuple:
    """Estimate depth using Depth Anything V2 from HuggingFace
    
    Returns: (depth_map, is_real) where is_real indicates if real depth or dummy
    """
    if model_dict is None:
        return estimate_depth_dummy(frame), False
    
    h, w = frame.shape[:2]
    
    # HuggingFace model dict
    if isinstance(model_dict, dict) and 'model' in model_dict:
        model = model_dict['model']
        processor = model_dict['processor']
        device = model_dict['device']
        
        try:
            from PIL import Image
            
            # Convert BGR to RGB for PIL
            pil_image = Image.fromarray(frame[:, :, ::-1])
            
            # Preprocess
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            
            # Inference
            outputs = model(**inputs)
            post_processed_output = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            
            # Convert to numpy and normalize
            depth_np = post_processed_output.squeeze().cpu().numpy()
            
            # Normalize [0, 1]
            depth_min = depth_np.min()
            depth_max = depth_np.max()
            if depth_max - depth_min > 1e-6:
                depth_normalized = (depth_np - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.ones_like(depth_np) * 0.5
            
            # Scale to realistic range: 300mm to 10000mm (30cm to 10m)
            depth_map = 300.0 + depth_normalized * 9700.0
            return depth_map.astype(np.float32), True  # Real depth
            
        except Exception as e:
            import traceback
            print(f"[Depth] ✗ Inference error: {str(e)[:100]}")
            print(f"[Depth] Traceback: {traceback.format_exc()[:200]}")
            return estimate_depth_dummy(frame), False  # Fallback to dummy
    
    # Fallback to dummy
    return estimate_depth_dummy(frame), False


def save_intrinsics_image(intrinsics: dict, frame_num: int):
    """Save camera intrinsics visualization"""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    K = np.array([
        [intrinsics['fx'], 0, intrinsics['cx']],
        [0, intrinsics['fy'], intrinsics['cy']],
        [0, 0, 1]
    ])
    
    cv2.putText(img, "CAMERA INTRINSICS K", (20, 40),
               cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
    
    y = 100
    cv2.putText(img, f"Resolution: {int(intrinsics['width'])}x{int(intrinsics['height'])}", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    y += 50
    for i, row in enumerate(K):
        text = f"[ {row[0]:8.2f}  {row[1]:8.2f}  {row[2]:8.2f} ]"
        cv2.putText(img, text, (40, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)
        y += 35
    
    y += 20
    cv2.putText(img, f"fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    path = OUTPUT_DIR / f"00_intrinsics_frame_{frame_num:03d}.png"
    cv2.imwrite(str(path), img)
    print(f"  ✓ {path.name}")


def save_depth_heatmap(depth_map: np.ndarray, frame_num: int, is_real: bool = True):
    """Save depth heatmap with indicator if it's real or dummy depth"""
    if depth_map.size == 0:
        return
    
    valid = depth_map[depth_map > 0]
    if len(valid) == 0:
        return
    
    depth_norm = np.clip(depth_map, valid.min(), valid.max())
    depth_norm = (depth_norm - valid.min()) / (valid.max() - valid.min() + 1e-8)
    depth_norm = (depth_norm * 255).astype(np.uint8)
    
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    
    # Add indicator if dummy
    if not is_real:
        cv2.putText(depth_colored, "DUMMY DEPTH (Fallback)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    path = OUTPUT_DIR / f"01_depth_heatmap_frame_{frame_num:03d}.png"
    cv2.imwrite(str(path), depth_colored)
    print(f"  ✓ {path.name} {'(DUMMY)' if not is_real else '(REAL)'}")


def save_yolo_detections(frame: np.ndarray, results, frame_num: int):
    """Save YOLO detections"""
    if not results or len(results) == 0:
        return
    
    annotated = frame.copy()
    result = results[0]
    boxes = result.boxes
    
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0]
        
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                     (0, 255, 0), 3)
        
        label = f"{result.names[cls_id]} {conf:.2f}"
        cv2.putText(annotated, label, (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    path = OUTPUT_DIR / f"02_yolo_detections_frame_{frame_num:03d}.png"
    cv2.imwrite(str(path), annotated)
    print(f"  ✓ {path.name}")


def save_depth_distribution(depth_map: np.ndarray, frame_num: int):
    """Save depth distribution"""
    valid = depth_map[depth_map > 0]
    if len(valid) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valid / 1000.0, bins=100, color='steelblue', alpha=0.7)
    ax.set_xlabel('Depth (meters)')
    ax.set_ylabel('Pixel Count')
    ax.set_title(f'Depth Distribution - Frame {frame_num}')
    ax.grid(True, alpha=0.3)
    
    path = OUTPUT_DIR / f"03_depth_distribution_frame_{frame_num:03d}.png"
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path.name}")


def save_point_cloud(depth_map: np.ndarray, intrinsics: dict, frame_num: int):
    """Save 3D point cloud visualization"""
    h, w = depth_map.shape
    
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # Back-project
    yy, xx = np.mgrid[0:h, 0:w]
    Z = depth_map / 1000.0  # mm to m
    X = (xx - cx) * Z / fx
    Y = (yy - cy) * Z / fy
    
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    valid_mask = depth_map.reshape(-1) > 0
    points = points[valid_mask]
    
    if len(points) == 0:
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample for visualization
    sample_idx = np.random.choice(len(points), min(5000, len(points)), replace=False)
    sample = points[sample_idx]
    Z_vals = sample[:, 2]
    colors = plt.cm.viridis((Z_vals - Z_vals.min()) / (Z_vals.max() - Z_vals.min() + 1e-8))
    
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=colors, s=1, alpha=0.6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Point Cloud - Frame {frame_num} ({len(points):,} points)')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 5])
    
    path = OUTPUT_DIR / f"04_point_cloud_frame_{frame_num:03d}.png"
    plt.savefig(path, dpi=80, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path.name}")


def save_reprojection(depth_map: np.ndarray, results, intrinsics: dict, frame_num: int):
    """Save 2D reprojection of 3D points"""
    h, w = depth_map.shape
    
    # Same back-projection
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    yy, xx = np.mgrid[0:h, 0:w]
    Z = depth_map / 1000.0
    X = (xx - cx) * Z / fx
    Y = (yy - cy) * Z / fy
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    
    # Reproject to 2D
    points_proj = np.zeros_like(points)
    points_proj[:, 0] = (points[:, 0] * fx / points[:, 2]) + cx
    points_proj[:, 1] = (points[:, 1] * fy / points[:, 2]) + cy
    
    proj_img = np.ones((h, w, 3), dtype=np.uint8) * 50
    
    # Draw dots
    sample_idx = np.random.choice(len(points_proj), min(3000, len(points_proj)), replace=False)
    for idx in sample_idx:
        x, y = points_proj[idx, :2]
        if 0 <= int(x) < w and 0 <= int(y) < h:
            cv2.circle(proj_img, (int(x), int(y)), 1, (0, 255, 255), -1)
    
    # Draw YOLO boxes
    if results and len(results) > 0:
        result = results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(proj_img, (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 255, 0), 2)
    
    path = OUTPUT_DIR / f"05_reprojection_dots_frame_{frame_num:03d}.png"
    cv2.imwrite(str(path), proj_img)
    print(f"  ✓ {path.name}")


def main():
    parser = argparse.ArgumentParser(description="Lightweight spatial mapping test")
    parser.add_argument("--video", type=str, default="data/examples/room.mp4")
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=30)
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"❌ Video not found: {args.video}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("🔬 SPATIAL MAPPING TEST - LIGHTWEIGHT")
    print("="*80)
    
    # Load video
    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video: {args.video}")
    print(f"   Resolution: {width}x{height} @ {fps:.1f} fps")
    print(f"   Total: {total_frames} frames ({total_frames/fps:.1f}s)")
    
    print(f"\n[1/3] Initializing...")
    yolo = YOLO("yolo11n.pt")
    print("  ✅ YOLO loaded")
    
    # Try to load depth model
    depth_model = load_depth_anything_v2(model_size='small', device='mps')
    
    intrinsics = auto_estimate_intrinsics(width, height)
    print("  ✅ Camera intrinsics estimated")
    
    # Save intrinsics
    save_intrinsics_image(intrinsics, 0)
    
    print(f"\n[2/3] Processing frames (every {args.sample_rate}th, max {args.max_frames})...")
    
    frame_idx = 0
    processed = 0
    total_time = 0
    
    while frame_idx < total_frames and processed < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % args.sample_rate != 0:
            frame_idx += 1
            continue
        
        start = time.time()
        processed += 1
        
        print(f"\n  Frame {frame_idx} (#{processed}):")
        
        # YOLO
        results = yolo(frame, verbose=False)
        
        # Depth - returns (depth_map, is_real_depth)
        depth_map, is_real = estimate_depth(frame, depth_model, device='mps')
        
        # Save all visualizations
        save_intrinsics_image(intrinsics, processed)
        save_depth_heatmap(depth_map, processed, is_real=is_real)
        save_yolo_detections(frame, results, processed)
        save_depth_distribution(depth_map, processed)
        save_point_cloud(depth_map, intrinsics, processed)
        save_reprojection(depth_map, results, intrinsics, processed)
        
        elapsed = time.time() - start
        total_time += elapsed
        print(f"    ⏱️  {elapsed:.2f}s")
        
        frame_idx += 1
    
    cap.release()
    
    print("\n" + "="*80)
    print("[3/3] COMPLETE")
    print("="*80)
    print(f"\n✅ Processed {processed} frames in {total_time:.1f}s ({total_time/processed:.2f}s per frame)")
    print(f"📁 Output: {OUTPUT_DIR.absolute()}/")
    
    files = sorted(OUTPUT_DIR.glob("*.png"))
    print(f"\n📊 Generated {len(files)} images:")
    for f in files:
        print(f"   {f.name}")


if __name__ == "__main__":
    main()
