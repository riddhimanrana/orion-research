"""
Depth estimation module using ONLY Depth Anything V2.

Depth Anything V2 is a state-of-the-art monocular depth model:
- Faster than MiDaS (15-60ms depending on size)
- More accurate predictions
- Better edge preservation
- Diverse training data

This module ONLY uses Depth Anything V2 (no MiDaS, no ZoeDepth).
"""

import time
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


class DepthEstimator:
    """
    Monocular depth estimation using ONLY Depth Anything V2.
    
    Supports multiple sizes:
        - 'small' (24.8M params, fastest, ~15ms)
        - 'base' (97.5M params, balanced, ~30ms)
        - 'large' (335M params, most accurate, ~60ms)
    """
    
    def __init__(
        self,
        model_name: str = "depth_anything_v2",
        model_size: str = "small",
        device: Optional[str] = None,
        half_precision: bool = False,
    ):
        """
        Initialize depth estimator with Depth Anything V2.
        
        Args:
            model_name: Must be "depth_anything_v2" (for compatibility)
            model_size: "small", "base", or "large"
            device: Device to run on ("cuda", "mps", "cpu", or None for auto-detect)
            half_precision: Use FP16 for faster inference (GPU only)
        """
        if model_name != "depth_anything_v2":
            raise ValueError(
                f"DepthEstimator now ONLY supports Depth Anything V2. "
                f"Got: {model_name}. Use model_name='depth_anything_v2'"
            )
        
        self.model_name = model_name
        self.model_size = model_size
        self.half_precision = half_precision
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"[DepthEstimator] Using Depth Anything V2 ({model_size}) on {self.device}")
        
        # Load model
        self.model = self._load_depth_anything_v2()
        self.model.eval()
        
        if self.half_precision and self.device.type in ["cuda", "mps"]:
            self.model.half()
            print(f"[DepthEstimator] Using FP16 precision")
    
    def _load_depth_anything_v2(self) -> torch.nn.Module:
        """Load Depth Anything V2 model from torch hub or local fallback."""
        try:
            print("[DepthEstimator] Loading Depth Anything V2 from torch hub...")
            
            # Map size to variant
            size_to_variant = {
                'small': 'depth_anything_v2_vits',
                'base': 'depth_anything_v2_vitb',
                'large': 'depth_anything_v2_vitl',
            }
            
            if self.model_size not in size_to_variant:
                raise ValueError(
                    f"Invalid model_size: {self.model_size}. "
                    f"Choose from: {list(size_to_variant.keys())}"
                )
            
            variant = size_to_variant[self.model_size]
            
            # Load from official Depth Anything V2 repository
            model: torch.nn.Module = torch.hub.load(
                'DepthAnything/Depth-Anything-V2',
                variant,
                pretrained=True,
                trust_repo=True
            )
            
            model = model.to(self.device)
            print(f"[DepthEstimator] Depth Anything V2 ({self.model_size}) loaded successfully from torch hub")
            return model
            
        except Exception as e:
            print(f"[DepthEstimator] Torch hub load failed: {e}")
            print(f"[DepthEstimator] Trying local DepthAnythingV2Estimator fallback...")
            
            try:
                # Fallback to local implementation
                from orion.perception.depth_anything import DepthAnythingV2Estimator
                fallback = DepthAnythingV2Estimator(model_size=self.model_size, device=str(self.device))
                print(f"[DepthEstimator] Using local DepthAnythingV2Estimator")
                return fallback.model
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load Depth Anything V2 from both torch hub and local implementation. "
                    f"Torch hub error: {e}\n"
                    f"Local fallback error: {e2}"
                )
    
    @torch.no_grad()
    def estimate(
        self,
        frame: np.ndarray,
        return_confidence: bool = False,
        input_size: int = 518,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Estimate depth for a frame using Depth Anything V2.
        
        Args:
            frame: RGB frame (H, W, 3) as uint8 numpy array
            return_confidence: If True, return None (not supported by v2)
            input_size: Input resolution for inference (higher = more detail, slower)
                       Default 518, can be 256-768
            
        Returns:
            depth_map: (H, W) depth in millimeters
            confidence_map: Always None (Depth Anything V2 doesn't provide confidence)
        """
        start_time = time.time()
        
        h, w = frame.shape[:2]
        
        # Preprocess
        input_tensor = self._preprocess(frame, input_size)
        
        # Inference
        if self.half_precision:
            input_tensor = input_tensor.half()
        
        depth_tensor = self.model(input_tensor)
        
        # Postprocess
        depth_map = self._postprocess(depth_tensor, (h, w))
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Confidence not provided by Depth Anything V2
        confidence_map = None
        
        return depth_map, confidence_map
    
    def _preprocess(self, frame: np.ndarray, input_size: int = 518) -> torch.Tensor:
        """
        Preprocess frame for Depth Anything V2.
        
        Args:
            frame: RGB frame (H, W, 3) as uint8
            input_size: Target input size (will be adjusted to multiple of 14)
            
        Returns:
            Preprocessed tensor (1, 3, H', W') normalized
        """
        h, w = frame.shape[:2]
        
        # Convert to tensor and normalize
        image = torch.from_numpy(frame).permute(2, 0, 1).float()  # (3, H, W)
        image = image.unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Resize to input_size (preserving aspect ratio)
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
        
        return image
    
    def _postprocess(
        self,
        depth_tensor: torch.Tensor,
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Postprocess depth tensor to match original frame size.
        
        Args:
            depth_tensor: Model output tensor (1, H', W') or (H', W')
            original_shape: (H, W) of original frame
            
        Returns:
            depth_map: (H, W) depth in millimeters, float32
        """
        # Ensure 4D tensor for interpolation
        if depth_tensor.dim() == 3:
            depth_tensor = depth_tensor.unsqueeze(0)
        
        # Squeeze batch dimension if needed
        if depth_tensor.shape[0] == 1:
            depth_tensor = depth_tensor.squeeze(0)  # (1, H', W') -> (H', W')
        
        # Ensure 3D
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)  # (H', W') -> (1, H', W')
        
        # Interpolate to original size
        depth_tensor = F.interpolate(
            depth_tensor.unsqueeze(0),
            size=original_shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # (1, 1, H, W) -> (H, W)
        
        # Convert to numpy
        depth_np = depth_tensor.cpu().numpy()
        
        # Depth Anything V2 outputs metric depth in meters
        # Convert to millimeters
        depth_map = depth_np * 1000.0
        
        # Clamp to reasonable range (100mm to 10m)
        depth_map = np.clip(depth_map, 100.0, 10000.0)
        
        return depth_map.astype(np.float32)
    
    def cleanup(self) -> None:
        """Release GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
