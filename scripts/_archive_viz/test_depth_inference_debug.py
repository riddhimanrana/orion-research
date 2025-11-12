#!/usr/bin/env python3
"""
Debug script to test depth inference frame-by-frame
Shows exactly which frames use real vs dummy depth
"""
import cv2
import numpy as np
import torch
from pathlib import Path
import sys
import time

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

OUTPUT_DIR = Path("/Users/riddhiman.rana/Desktop/Coding/Orion/orion-research/depth_debug_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_depth_anything_v2(model_size: str = 'small', device: str = 'mps'):
    """Load Depth Anything V2 from HuggingFace"""
    print(f"[Depth] Loading Depth Anything V2 ({model_size}) from HuggingFace...")
    
    try:
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        
        # Map model sizes to HuggingFace model IDs
        model_id_map = {
            'small': 'depth-anything/Depth-Anything-V2-small-hf',
            'base': 'depth-anything/Depth-Anything-V2-base-hf',
            'large': 'depth-anything/Depth-Anything-V2-large-hf',
        }
        
        model_id = model_id_map.get(model_size, 'depth-anything/Depth-Anything-V2-small-hf')
        
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        model.to(device)
        model.eval()
        
        print(f"[Depth] ✓ Depth Anything V2 ({model_size}) loaded successfully")
        return {'model': model, 'processor': processor, 'device': device}
        
    except Exception as e:
        print(f"[Depth] ✗ HuggingFace load failed: {str(e)[:100]}...")
        print(f"[Depth] Falling back to dummy depth")
        return None


@torch.no_grad()
def estimate_depth(frame: np.ndarray, model_dict, device: str = 'mps'):
    """Estimate depth and return stats"""
    if model_dict is None:
        h, w = frame.shape[:2]
        depth = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            depth[y, :] = (2.0 + 2.0 * y / h) * 1000
        return depth, False, "DUMMY"
    
    h, w = frame.shape[:2]
    
    if isinstance(model_dict, dict) and 'model' in model_dict:
        model = model_dict['model']
        processor = model_dict['processor']
        device = model_dict['device']
        
        try:
            from PIL import Image
            
            # Convert BGR to RGB
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
            
            # Convert to numpy
            depth_np = post_processed_output.squeeze().cpu().numpy()
            
            # Normalize
            depth_min = depth_np.min()
            depth_max = depth_np.max()
            if depth_max - depth_min > 1e-6:
                depth_normalized = (depth_np - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.ones_like(depth_np) * 0.5
            
            # Scale to mm
            depth_map = 300.0 + depth_normalized * 9700.0
            
            # Compute stats
            mean_depth = depth_map.mean()
            std_depth = depth_map.std()
            
            return depth_map.astype(np.float32), True, f"REAL (μ={mean_depth:.0f}, σ={std_depth:.0f})"
            
        except Exception as e:
            import traceback
            print(f"[Depth] ✗ Error: {str(e)[:100]}")
            traceback.print_exc()
            
            # Return dummy
            depth = np.zeros((h, w), dtype=np.float32)
            for y in range(h):
                depth[y, :] = (2.0 + 2.0 * y / h) * 1000
            return depth, False, f"ERROR: {str(e)[:50]}"
    
    h, w = frame.shape[:2]
    depth = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        depth[y, :] = (2.0 + 2.0 * y / h) * 1000
    return depth, False, "DUMMY (no model)"


def save_depth_debug_image(depth_map, frame_idx, is_real, status):
    """Save depth map with detailed analysis overlay"""
    h, w = depth_map.shape
    
    # Normalize for visualization
    valid = depth_map[depth_map > 0]
    if len(valid) > 0:
        depth_norm = np.clip(depth_map, valid.min(), valid.max())
        depth_norm = (depth_norm - valid.min()) / (valid.max() - valid.min() + 1e-8)
    else:
        depth_norm = depth_map
    
    depth_norm = (depth_norm * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    
    # Add text overlay
    if is_real:
        color = (0, 255, 0)  # Green
        text = f"✓ REAL DEPTH - {status}"
    else:
        color = (0, 0, 255)  # Red
        text = f"✗ FALLBACK/DUMMY - {status}"
    
    cv2.putText(depth_colored, text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Add statistics
    mean_d = depth_map.mean()
    std_d = depth_map.std()
    min_d = depth_map.min()
    max_d = depth_map.max()
    
    stats_text = f"Mean: {mean_d:.0f}mm | Std: {std_d:.0f}mm | Range: [{min_d:.0f}, {max_d:.0f}]mm"
    cv2.putText(depth_colored, stats_text, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Check if it's just a gradient (bad)
    vertical_gradient = np.abs(np.diff(depth_map, axis=0)).mean()
    is_gradient = vertical_gradient < 50  # If very low variation, likely a gradient
    
    if is_gradient:
        cv2.putText(depth_colored, "⚠ WARNING: Looks like gradient (bad)", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    path = OUTPUT_DIR / f"frame_{frame_idx:03d}_depth.png"
    cv2.imwrite(str(path), depth_colored)
    
    return is_gradient


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug depth inference")
    parser.add_argument("--video", type=str, default="data/examples/room.mp4")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--skip", type=int, default=10)
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {args.video}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("🔍 DEPTH INFERENCE DEBUG")
    print("="*80)
    
    # Load video
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video: {args.video}")
    print(f"   Resolution: {width}x{height} @ {fps:.1f} fps")
    print(f"   Total frames: {total_frames}")
    
    # Load model
    print(f"\n[1/2] Loading model...")
    depth_model = load_depth_anything_v2(model_size='small', device='mps')
    
    # Test frames
    print(f"\n[2/2] Testing depth inference (every {args.skip}th frame, max {args.frames})...")
    
    frame_idx = 0
    processed = 0
    gradient_count = 0
    real_count = 0
    dummy_count = 0
    
    while processed < args.frames and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue
        
        print(f"\n  Frame #{processed+1} (video frame {frame_idx}):")
        start = time.time()
        
        depth_map, is_real, status = estimate_depth(frame, depth_model, device='mps')
        is_gradient = save_depth_debug_image(depth_map, processed, is_real, status)
        
        elapsed = time.time() - start
        
        if is_real:
            real_count += 1
            icon = "✓"
        else:
            dummy_count += 1
            icon = "✗"
        
        if is_gradient:
            gradient_count += 1
            gradient_icon = "⚠"
        else:
            gradient_icon = "✓"
        
        print(f"    {icon} {status}")
        print(f"    {gradient_icon} Gradient check: {'GRADIENT DETECTED' if is_gradient else 'Looks OK'}")
        print(f"    ⏱️  {elapsed:.2f}s")
        
        processed += 1
        frame_idx += 1
    
    cap.release()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Real depth frames: {real_count}")
    print(f"Dummy/Fallback frames: {dummy_count}")
    print(f"Gradient-like frames: {gradient_count}")
    print(f"\n✅ Output saved to: {OUTPUT_DIR}/")
    print(f"📊 Generated {processed} debug images")


if __name__ == "__main__":
    main()
