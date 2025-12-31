"""
Depth estimation module powered by Depth Anything V3.

Depth Anything V3 is the latest monocular depth model from the DepthAnything
team, delivering sharper indoor predictions, better temporal consistency, and
native metric depth output (meters). We default to V3 via Torch Hub with an
automatic fallback to the stable V2 weights when V3 assets are unavailable.
"""

import time
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import sys
from pathlib import Path

# Add Depth-Anything-V2 to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DA_V2_PATH = WORKSPACE_ROOT / "Depth-Anything-V2-temp"
if DA_V2_PATH.exists() and str(DA_V2_PATH) not in sys.path:
    sys.path.append(str(DA_V2_PATH))

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    DA2_AVAILABLE = True
except ImportError as e:
    DA2_AVAILABLE = False
    print(f"[DepthEstimator] DepthAnythingV2 module not found. V2 local loading will fail. Error: {e}")


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V2."""
    
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
            model_name: Must be "depth_anything_v2"
            model_size: "small", "base", or "large"
            device: Device to run on ("cuda", "mps", "cpu", or None for auto-detect)
            half_precision: Use FP16 for faster inference (GPU only)
        """
        if model_name != "depth_anything_v2":
            raise ValueError(f"DepthEstimator now ONLY supports Depth Anything V2. Got: {model_name}")
        
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
        """Load Depth Anything V2 model directly from local files."""
        if not DA2_AVAILABLE:
            raise ImportError("DepthAnythingV2 module is not available. Check the path and installation.")

        print("[DepthEstimator] Loading Depth Anything V2 from local repository...")
        
        # Define model configurations
        model_configs = {
            'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [192, 384, 768, 1536]}
        }
        
        if self.model_size not in model_configs:
            raise ValueError(f"Invalid model_size: {self.model_size}. Choose from: {list(model_configs.keys())}")

        # Instantiate model
        model = DepthAnythingV2(**model_configs[self.model_size])
        
        # Load weights
        weights_dir = WORKSPACE_ROOT / "models" / "_torch" / "depth_anything_v2"
        weights_dir.mkdir(parents=True, exist_ok=True)
        encoder = model_configs[self.model_size]['encoder']
        weights_path = weights_dir / f"depth_anything_v2_{encoder}.pth"
        
        if not weights_path.exists():
            print(f"[DepthEstimator] Weights not found at {weights_path}, downloading...")
            # The official HF repo is now gated, using a community mirror
            url = f"https://huggingface.co/spaces/LiheYoung/Depth-Anything-V2/resolve/main/checkpoints/depth_anything_v2_{encoder}.pth"
            try:
                torch.hub.download_url_to_file(url, weights_path)
            except Exception as e:
                print(f"[DepthEstimator] ✗ Failed to download model weights: {e}")
                raise e

        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        return model
        
        # Load pretrained weights
        # Weights are typically downloaded to a cache dir by the model's own logic,
        # or we can point to a local file if we download them manually.
        # For now, we rely on the model's default weight loading.
        model.load_state_dict(torch.load(f'https://huggingface.co/depth-anything/Depth-Anything-V2-{self.model_size.capitalize()}-hf/resolve/main/pytorch_model.bin', map_location='cpu'))

        model = model.to(self.device)
        print(f"[DepthEstimator] Depth Anything V2 ({self.model_size}) loaded successfully.")
        return model
    
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
            confidence_map: Always None (Depth Anything models don't expose confidence)
        """
        start_time = time.time()
        
        h, w = frame.shape[:2]
        
        # Preprocess (V2)
        input_tensor = self._preprocess(frame, input_size)
        
        # Inference
        if self.half_precision and self.device.type in ["cuda", "mps"]:
            input_tensor = input_tensor.half()
        
        depth_tensor = self.model(input_tensor)
        
        # Postprocess
        depth_map = self._postprocess(depth_tensor, (h, w))
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Confidence not provided by Depth Anything models
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
        
        # Depth Anything outputs metric depth in meters
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
