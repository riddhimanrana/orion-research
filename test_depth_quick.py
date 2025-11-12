#!/usr/bin/env python3
"""Quick test of Depth Anything V2 on a single frame"""

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import time

def test_depth():
    print("🧪 QUICK DEPTH TEST")
    print("=" * 60)
    
    # Load video
    video_path = "data/examples/video_short.mp4"
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        return
    
    h, w = frame.shape[:2]
    print(f"✓ Loaded frame: {w}x{h}")
    
    # Load model
    print("\n[1/3] Loading Depth Anything V2...")
    try:
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        
        processor = AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-small-hf')
        model = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-small-hf')
        model.to('mps')
        model.eval()
        print("✓ Model loaded")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Run inference
    print("\n[2/3] Running depth estimation...")
    try:
        with torch.no_grad():
            start = time.time()
            
            # Preprocess
            pil_image = Image.fromarray(frame[:, :, ::-1])
            inputs = processor(images=pil_image, return_tensors="pt").to('mps')
            
            # Infer
            outputs = model(**inputs)
            
            # Postprocess
            post_processed_output = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            
            depth_np = post_processed_output.squeeze().cpu().numpy()
            elapsed = time.time() - start
            
        print(f"✓ Inference completed in {elapsed:.2f}s")
        print(f"  Depth shape: {depth_np.shape}")
        print(f"  Depth range: [{depth_np.min():.4f}, {depth_np.max():.4f}]")
        print(f"  Depth mean: {depth_np.mean():.4f}")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Normalize and save
    print("\n[3/3] Saving visualization...")
    try:
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        depth_normalized = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        
        output_path = "test_depth_output.png"
        cv2.imwrite(output_path, depth_colored)
        print(f"✓ Saved to {output_path}")
        
        # Show statistics
        print(f"\n📊 Depth Statistics:")
        print(f"  Min: {depth_min:.4f}")
        print(f"  Max: {depth_max:.4f}")
        print(f"  Mean: {depth_np.mean():.4f}")
        print(f"  Std: {depth_np.std():.4f}")
        print(f"  25th percentile: {np.percentile(depth_np, 25):.4f}")
        print(f"  50th percentile: {np.percentile(depth_np, 50):.4f}")
        print(f"  75th percentile: {np.percentile(depth_np, 75):.4f}")
        
    except Exception as e:
        print(f"❌ Saving failed: {e}")
        return
    
    print("\n✅ DEPTH TEST COMPLETE")

if __name__ == '__main__':
    test_depth()
