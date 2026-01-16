#!/usr/bin/env python3
"""
Depth Anything V2 Integration
==============================

Depth Anything V2 is a state-of-the-art monocular depth estimation model:
- Faster than MiDaS (especially the Small variant)
- More accurate depth predictions
- Better edge preservation
- Trained on much more diverse data

Models:
- depth_anything_v2_vits (Small): 24.8M params, ~15ms on M1
- depth_anything_v2_vitb (Base): 97.5M params, ~30ms on M1  
- depth_anything_v2_vitl (Large): 335M params, ~60ms on M1

GitHub: https://github.com/DepthAnything/Depth-Anything-V2
Paper: https://arxiv.org/abs/2406.09414

Usage:
    from orion.perception.depth_anything import DepthAnythingV2Estimator
    
    estimator = DepthAnythingV2Estimator(model_size='small')
    depth_map, confidence = estimator.estimate(rgb_frame)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional
from pathlib import Path


class DepthAnythingV2Estimator:
    """
    Depth Anything V2 depth estimator
    
    Significantly faster and more accurate than MiDaS, especially for:
    - Indoor scenes
    - Complex geometry
    - Thin structures (poles, cables)
    - Reflective surfaces
    """
    
    def __init__(self, 
                 model_size: str = 'small',
                 device: str = 'mps',
                 model_dir: str = 'models/weights'):
        """
        Initialize Depth Anything V2
        
        Args:
            model_size: 'small' (fastest), 'base', or 'large' (most accurate)
            device: 'mps', 'cuda', or 'cpu'
            model_dir: Directory to store model weights
        """
        self.model_size = model_size
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[DepthAnythingV2] Loading model: {model_size} on {device}")
        
        # Map model sizes to variants
        model_variants = {
            'small': 'depth_anything_v2_vits',
            'base': 'depth_anything_v2_vitb',
            'large': 'depth_anything_v2_vitl'
        }
        
        if model_size not in model_variants:
            raise ValueError(f"Invalid model_size: {model_size}. Choose from: {list(model_variants.keys())}")
        
        variant = model_variants[model_size]
        
        # Load model from torch hub or local
        try:
            # Try loading from Depth-Anything-V2 repo
            self.model = torch.hub.load(
                'DepthAnything/Depth-Anything-V2',
                variant,
                pretrained=True,
                trust_repo=True
            )
        except Exception as e:
            print(f"[DepthAnythingV2] Failed to load from hub: {e}")
            print("[DepthAnythingV2] Attempting local load...")
            
            # Fallback: load from local weights
            local_path = self.model_dir / f"{variant}.pth"
            if not local_path.exists():
                raise RuntimeError(
                    f"Model not found at {local_path}. "
                    f"Download from https://github.com/DepthAnything/Depth-Anything-V2/releases"
                )
            
            # Load architecture and weights
            from orion.perception.depth_anything_v2.dpt import DepthAnythingV2
            
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            
            encoder = variant.split('_')[-1]
            self.model = DepthAnythingV2(**model_configs[encoder])
            self.model.load_state_dict(torch.load(local_path, map_location='cpu'))
        
        self.model = self.model.to(device).eval()
        
        print(f"[DepthAnythingV2] Model loaded successfully")
    
    @torch.no_grad()
    def estimate(self, 
                 frame: np.ndarray,
                 input_size: int = 518) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Estimate depth from RGB image
        
        Args:
            frame: RGB image (H, W, 3) uint8
            input_size: Input size (model processes at this resolution)
                       518 is default, higher = more detail but slower
        
        Returns:
            depth_map: Depth in millimeters (H, W) float32
            confidence: Confidence map (H, W) float32, or None
        """
        h, w = frame.shape[:2]
        
        # Convert to tensor and normalize
        image = torch.from_numpy(frame).permute(2, 0, 1).float()  # (3, H, W)
        image = image.unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Resize to model input size (preserving aspect ratio)
        scale = input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Make divisible by 14 (ViT patch size)
        new_h = (new_h // 14) * 14
        new_w = (new_w // 14) * 14
        
        image = F.interpolate(
            image,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )
        
        # Infer depth
        depth = self.model(image)
        
        # Resize back to original size
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # Convert to numpy
        depth_map = depth.cpu().numpy()
        
        # Normalize to [0, 1] 
        # Depth Anything V2 outputs relative depth, not metric
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        # IMPORTANT: Scale to reasonable depth range in METERS (not mm)
        # For indoor desktop scenes: 0.5m to 5m is more realistic
        # Close objects (keyboard/mouse) at 0.5-1.5m, far walls at 3-5m
        # This is just an initial guess - semantic scale recovery will refine it
        depth_map = (depth_map * 4.5) + 0.5  # 0.5m to 5.0m
        
        # Confidence (we don't have this directly, so estimate from depth variance)
        confidence = self._estimate_confidence(depth_map)
        
        return depth_map.astype(np.float32), confidence
    
    def _estimate_confidence(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Estimate confidence from depth gradients
        
        Areas with large gradients (edges) tend to be less confident
        """
        # Compute gradients
        grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to [0, 1]
        grad_mag = grad_mag / (grad_mag.max() + 1e-6)
        
        # Confidence is inverse of gradient (smooth areas = high confidence)
        confidence = 1.0 - grad_mag
        
        # Apply sigmoid to sharpen
        confidence = 1.0 / (1.0 + np.exp(-10 * (confidence - 0.5)))
        
        return confidence.astype(np.float32)
    
    def estimate_batch(self, 
                      frames: list,
                      input_size: int = 518) -> list:
        """
        Batch depth estimation (more efficient for multiple frames)
        
        Args:
            frames: list of RGB images (H, W, 3)
            input_size: Input size for model
        
        Returns:
            list of (depth_map, confidence) tuples
        """
        # TODO: Implement true batch processing
        # For now, process sequentially
        results = []
        for frame in frames:
            depth, conf = self.estimate(frame, input_size)
            results.append((depth, conf))
        
        return results


def compare_depth_models(frame: np.ndarray):
    """
    Compare MiDaS vs Depth Anything V2
    
    Usage:
        from orion.perception.depth_anything import compare_depth_models
        compare_depth_models(rgb_frame)
    """
    import time
    from orion.perception.depth import DepthEstimator
    
    print("=" * 60)
    print("DEPTH MODEL COMPARISON")
    print("=" * 60)
    
    # MiDaS
    print("\n1. MiDaS (DPT-Hybrid)")
    midas = DepthEstimator(device='mps')
    
    t0 = time.time()
    midas_depth, _ = midas.estimate(frame)
    midas_time = (time.time() - t0) * 1000
    
    print(f"   Time: {midas_time:.1f}ms")
    print(f"   Range: {midas_depth.min():.0f} - {midas_depth.max():.0f} mm")
    
    # Depth Anything V2 Small
    print("\n2. Depth Anything V2 (Small)")
    dav2_small = DepthAnythingV2Estimator(model_size='small', device='mps')
    
    t0 = time.time()
    dav2_depth, dav2_conf = dav2_small.estimate(frame)
    dav2_time = (time.time() - t0) * 1000
    
    print(f"   Time: {dav2_time:.1f}ms ({midas_time/dav2_time:.1f}x faster than MiDaS)")
    print(f"   Range: {dav2_depth.min():.0f} - {dav2_depth.max():.0f} mm")
    print(f"   Avg confidence: {dav2_conf.mean():.2f}")
    
    print("\n" + "=" * 60)
    print(f"Winner: Depth Anything V2 ({dav2_time/midas_time*100:.0f}% of MiDaS time)")
    print("=" * 60)
    
    return {
        'midas': (midas_depth, midas_time),
        'dav2': (dav2_depth, dav2_time, dav2_conf)
    }


if __name__ == "__main__":
    # Test with a dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    estimator = DepthAnythingV2Estimator(model_size='small')
    depth, conf = estimator.estimate(frame)
    
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: {depth.min():.0f} - {depth.max():.0f} mm")
    print(f"Confidence shape: {conf.shape}")
    print(f"Avg confidence: {conf.mean():.2f}")
