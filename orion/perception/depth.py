"""
Depth estimation module using ZoeDepth or MiDaS.
"""

import time
from typing import Tuple, Optional, Literal
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


class DepthEstimator:
    """
    Monocular depth estimation for egocentric video.
    
    Supports:
        - ZoeDepth (default, optimized for near-field)
        - MiDaS (fallback, proven baseline)
    """
    
    def __init__(
        self,
        model_name: Literal["zoe", "midas"] = "zoe",
        device: Optional[str] = None,
        half_precision: bool = False,
    ):
        """
        Initialize depth estimator.
        
        Args:
            model_name: "zoe" for ZoeDepth or "midas" for MiDaS
            device: Device to run on ("cuda", "mps", "cpu", or None for auto-detect)
            half_precision: Use FP16 for faster inference (GPU only)
        """
        self.model_name = model_name
        self.half_precision = half_precision and model_name == "zoe"  # Only ZoeDepth supports FP16 well
        
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
        
        print(f"[DepthEstimator] Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        if self.half_precision:
            self.model.half()
            print(f"[DepthEstimator] Using FP16 precision")
    
    def _load_model(self) -> torch.nn.Module:
        """Load the depth estimation model."""
        if self.model_name == "zoe":
            return self._load_zoedepth()
        elif self.model_name == "midas":
            return self._load_midas()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _load_zoedepth(self) -> torch.nn.Module:
        """Load ZoeDepth model."""
        try:
            # Try to import ZoeDepth
            # Note: This requires the ZoeDepth repo or package installed
            # For now, we'll use torch.hub as fallback
            print("[DepthEstimator] Loading ZoeDepth model...")
            
            # Load from torch hub
            model = torch.hub.load(
                "isl-org/ZoeDepth",
                "ZoeD_N",
                pretrained=True,
                trust_repo=True
            )
            
            model = model.to(self.device)
            print("[DepthEstimator] ZoeDepth loaded successfully")
            return model
            
        except Exception as e:
            print(f"[DepthEstimator] Failed to load ZoeDepth: {e}")
            print("[DepthEstimator] Falling back to MiDaS...")
            self.model_name = "midas"  # Update model name for postprocessing
            return self._load_midas()
    
    def _load_midas(self) -> torch.nn.Module:
        """Load MiDaS model."""
        print("[DepthEstimator] Loading MiDaS model...")
        
        # Load MiDaS small (fast, reasonable quality)
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        model = model.to(self.device)
        
        print("[DepthEstimator] MiDaS loaded successfully")
        return model
    
    @torch.no_grad()
    def estimate(
        self,
        frame: np.ndarray,
        return_confidence: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Estimate depth for a frame.
        
        Args:
            frame: RGB frame (H, W, 3) as uint8 numpy array
            return_confidence: If True, return confidence map (if available)
            
        Returns:
            depth_map: (H, W) depth in millimeters
            confidence_map: Optional (H, W) confidence [0, 1] or None
        """
        start_time = time.time()
        
        # Preprocess
        input_tensor = self._preprocess(frame)
        
        # Inference
        if self.half_precision:
            input_tensor = input_tensor.half()
        
        depth_tensor = self.model(input_tensor)
        
        # Postprocess
        depth_map = self._postprocess(depth_tensor, frame.shape[:2])
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Note: Confidence not available in base models, would need custom implementation
        confidence_map = None
        
        return depth_map, confidence_map
    
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for depth model.
        
        Args:
            frame: RGB frame (H, W, 3) as uint8
            
        Returns:
            Preprocessed tensor (1, 3, H', W')
        """
        # Convert to PIL for easier resizing
        pil_image = Image.fromarray(frame)
        
        # Resize to model input size (typically 384x384 or 518x518)
        if self.model_name == "zoe":
            target_size = (384, 512)  # ZoeDepth preferred size
        else:
            target_size = (384, 384)  # MiDaS small size
        
        pil_image = pil_image.resize(target_size, Image.BILINEAR)
        
        # Convert to tensor and normalize
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, H, W)
        img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.to(self.device)
    
    def _postprocess(
        self,
        depth_tensor: torch.Tensor,
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Postprocess depth tensor to match original frame size.
        
        Args:
            depth_tensor: Model output tensor
            original_shape: (H, W) of original frame
            
        Returns:
            depth_map: (H, W) depth in millimeters
        """
        # Get depth map from tensor
        if isinstance(depth_tensor, dict):
            # ZoeDepth returns dict with 'metric_depth'
            depth = depth_tensor.get('metric_depth', depth_tensor.get('depth'))
        else:
            depth = depth_tensor
        
        # Squeeze and convert to numpy
        depth = depth.squeeze().cpu().numpy()
        
        # Convert to millimeters FIRST (models typically output meters)
        # ZoeDepth outputs metric depth in meters
        # MiDaS outputs inverse depth (needs rescaling)
        if self.model_name == "zoe":
            depth_map = depth * 1000.0  # meters to mm
        else:
            # MiDaS outputs relative/inverse depth, normalize to reasonable range
            # Assume typical indoor range: 0.5m to 5m
            if depth.max() > depth.min():
                # Normalize to 0-1 range
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                # Map to 500mm - 5000mm range (inverted because MiDaS outputs inverse depth)
                depth_map = (1.0 - depth_normalized) * 4500.0 + 500.0
            else:
                # Fallback if depth is uniform
                depth_map = np.full_like(depth, 2000.0)
        
        # Resize to original shape AFTER depth conversion
        depth_map = cv2.resize(depth_map, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        
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
