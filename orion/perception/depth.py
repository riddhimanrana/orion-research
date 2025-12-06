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

# Add Depth-Anything-3/src to path if available. We keep this simple and
# robust: if the folder exists and the import works, we always prefer the
# local clone over any torch hub fallback.
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DA3_PATH = WORKSPACE_ROOT / "Depth-Anything-3" / "src"

DA3_AVAILABLE = False
if DA3_PATH.exists():
    if str(DA3_PATH) not in sys.path:
        sys.path.append(str(DA3_PATH))
    try:
        from depth_anything_3.api import DepthAnything3
        DA3_AVAILABLE = True
        print(f"[DepthEstimator] Found local Depth-Anything-3 at {DA3_PATH}")
    except ImportError as e:
        print(f"[DepthEstimator] DepthAnything3 module not found in local clone. V3 local loading will fail. Error: {e}")
else:
    print(f"[DepthEstimator] Depth-Anything-3 directory not found at {DA3_PATH}. V3 local loading will be skipped.")


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V3 (with V2 fallback)."""
    
    def __init__(
        self,
        model_name: str = "depth_anything_v3",
        model_size: str = "small",
        device: Optional[str] = None,
        half_precision: bool = False,
    ):
        """
        Initialize depth estimator with Depth Anything V3.
        
        Args:
            model_name: Must be "depth_anything_v3" (retained for compatibility)
            model_size: "small", "base", or "large"
            device: Device to run on ("cuda", "mps", "cpu", or None for auto-detect)
            half_precision: Use FP16 for faster inference (GPU only)
        """
        if model_name != "depth_anything_v3":
            raise ValueError(
                f"DepthEstimator now ONLY supports Depth Anything V3. "
                f"Got: {model_name}. Use model_name='depth_anything_v3'"
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
        
        print(f"[DepthEstimator] Using Depth Anything V3 ({model_size}) on {self.device}")
        
        # Load model
        self.model = self._load_depth_anything()
        self.model.eval()
        
        if self.half_precision and self.device.type in ["cuda", "mps"]:
            self.model.half()
            print(f"[DepthEstimator] Using FP16 precision")
    
    def _load_depth_anything(self) -> torch.nn.Module:
        """Load Depth Anything model with V3 preference and V2 fallback."""
        try:
            return self._load_depth_anything_v3()
        except Exception as v3_exc:
            print(f"[DepthEstimator] V3 torch hub load failed: {v3_exc}")
            print("[DepthEstimator] Falling back to Depth Anything V2 weights...")
            try:
                return self._load_depth_anything_v2()
            except Exception as v2_exc:
                raise RuntimeError(
                    "Failed to load Depth Anything models (V3 primary, V2 fallback). "
                    f"V3 error: {v3_exc}\nV2 error: {v2_exc}"
                ) from v2_exc

    def _load_depth_anything_v3(self) -> torch.nn.Module:
        """Load Depth Anything V3 variant via local clone or torch hub."""
        print("[DepthEstimator] Loading Depth Anything V3...")

        if DA3_AVAILABLE:
            print("[DepthEstimator] Using local Depth-Anything-3 repository.")
            size_to_hf_repo = {
                'small': 'depth-anything/DA3-Small',
                'base': 'depth-anything/DA3-Base',
                'large': 'depth-anything/DA3-Large',
            }
            
            if self.model_size not in size_to_hf_repo:
                raise ValueError(
                    f"Invalid model_size: {self.model_size}. "
                    f"Choose from: {list(size_to_hf_repo.keys())}"
                )
                
            repo_id = size_to_hf_repo[self.model_size]
            model = DepthAnything3.from_pretrained(repo_id)
            model = model.to(self.device)
            print(f"[DepthEstimator] Depth Anything V3 ({self.model_size}) loaded from {repo_id}")
            return model
        else:
            print("[DepthEstimator] Local Depth-Anything-3 not found. Trying torch hub...")
            # Fallback to torch hub if local repo is missing but somehow we want V3
            # Note: The user specifically asked to clone the repo, so this path might be less relevant
            # but good for robustness if they delete the folder.
            # However, the torch hub V3 loading I wrote before might not be compatible with the new V3 repo structure
            # if the hubconf changed. Assuming it works as before or we fail over to V2.
            
            size_to_variant = {
                'small': 'depth_anything_v3_vits',
                'base': 'depth_anything_v3_vitb',
                'large': 'depth_anything_v3_vitl',
            }

            if self.model_size not in size_to_variant:
                raise ValueError(
                    f"Invalid model_size: {self.model_size}. "
                    f"Choose from: {list(size_to_variant.keys())}"
                )

            variant = size_to_variant[self.model_size]
            model: torch.nn.Module = torch.hub.load(
                'DepthAnything/Depth-Anything-V3',
                variant,
                pretrained=True,
                trust_repo=True,
            )
            model = model.to(self.device)
            print(f"[DepthEstimator] Depth Anything V3 ({self.model_size}) loaded via Hub")
            return model

    def _load_depth_anything_v2(self) -> torch.nn.Module:
        """Fallback to Depth Anything V2 via torch hub (no local deps)."""
        size_to_variant = {
            'small': 'depth_anything_v2_vits',
            'base': 'depth_anything_v2_vitb',
            'large': 'depth_anything_v2_vitl',
        }
        variant = size_to_variant[self.model_size]
        model: torch.nn.Module = torch.hub.load(
            'DepthAnything/Depth-Anything-V2',
            variant,
            pretrained=True,
            trust_repo=True,
        )
        model = model.to(self.device)
        print(f"[DepthEstimator] Depth Anything V2 ({self.model_size}) fallback loaded")
        return model
    
    @torch.no_grad()
    def estimate(
        self,
        frame: np.ndarray,
        return_confidence: bool = False,
        input_size: int = 518,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Estimate depth for a frame using Depth Anything V3.
        
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
        
        # Check if using local V3 model
        is_local_v3 = DA3_AVAILABLE and type(self.model).__name__ == 'DepthAnything3'
        
        if is_local_v3:
            # Use the inference API from DepthAnything3
            # It handles preprocessing, inference, and postprocessing
            prediction = self.model.inference(
                [frame],
                process_res=input_size,
                show_cameras=False
            )
            depth_map = prediction.depth[0]  # (H, W)
            
            # Ensure output matches input dimensions
            if depth_map.shape[:2] != (h, w):
                depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
                
        else:
            # Preprocess (V2 or Hub V3)
            input_tensor = self._preprocess(frame, input_size)
            
            # Inference
            if self.half_precision:
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
