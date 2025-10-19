"""
Unified Model Manager for Orion
================================

Single point of access for all models in the Orion pipeline.
Handles lazy loading, memory management, and device placement.

Architecture:
    YOLO11x    → Detection (what + where)
    CLIP       → Embeddings (re-ID + verification)
    FastVLM    → Descriptions (rich text)
    Gemma3:4b  → Q&A (via Ollama, not managed here)

Author: Orion Research Team
Date: October 2025
"""

import logging
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton manager for all Orion models.
    
    Benefits:
    - Lazy loading (models only loaded when needed)
    - Memory efficient (shared instances)
    - Device management (auto MPS/CUDA/CPU)
    - Easy to swap backends
    
    Usage:
        manager = ModelManager.get_instance()
        
        # Get YOLO detector
        yolo = manager.yolo
        results = yolo(image)
        
        # Get CLIP embedder
        clip = manager.clip
        embedding = clip.encode_image(image)
        
        # Get FastVLM describer
        fastvlm = manager.fastvlm
        description = fastvlm.generate(image, prompt)
    """
    
    _instance: Optional['ModelManager'] = None
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize model manager.
        
        Args:
            models_dir: Path to models directory (default: repo/models/)
        """
        if models_dir is None:
            # Auto-detect models directory
            try:
                from .models import AssetManager
                self.asset_manager = AssetManager()
            except ImportError:
                # Fallback: try from parent directory
                try:
                    import sys
                    import os
                    # Add parent directory to path
                    parent_dir = Path(__file__).parent.parent.parent
                    if str(parent_dir) not in sys.path:
                        sys.path.insert(0, str(parent_dir))
                    from models import ModelManager as AssetManager
                    self.asset_manager = AssetManager()
                except ImportError:
                    logger.warning("Could not import AssetManager, using default paths")
                    self.asset_manager = None
                    self.models_dir = Path.home() / ".cache" / "orion" / "models"
        else:
            self.models_dir = models_dir
            self.asset_manager = None
        
        # Model instances (lazy loaded)
        self._yolo: Optional[Any] = None
        self._clip: Optional[Any] = None
        self._fastvlm: Optional[Any] = None
        
        # LLM for contextual understanding
        self._ollama_client: Optional[Any] = None
        
        # Device management
        self.device = self._detect_device()
        logger.info(f"ModelManager initialized (device: {self.device})")
    
    @classmethod
    def get_instance(cls, models_dir: Optional[Path] = None) -> 'ModelManager':
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = cls(models_dir)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)"""
        if cls._instance is not None:
            cls._instance.cleanup()
            cls._instance = None
    
    def _detect_device(self) -> str:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    # ========================================================================
    # YOLO11x Detector
    # ========================================================================
    
    @property
    def yolo(self) -> Any:
        """
        Get YOLO11x detector (lazy loaded).
        
        Returns:
            YOLO model instance
        """
        if self._yolo is None:
            self._yolo = self._load_yolo()
        return self._yolo
    
    def _load_yolo(self) -> Any:
        """Load YOLO11x model"""
        try:
            from ultralytics import YOLO
            
            logger.info("Loading YOLO11x detector...")
            
            # Get model path from asset manager
            if self.asset_manager:
                asset_dir = self.asset_manager.ensure_asset("yolo11x")
                model_path = asset_dir / "yolo11x.pt"
            else:
                model_path = self.models_dir / "weights" / "yolo11x.pt"
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"YOLO11x weights not found at {model_path}. "
                    "Run: python scripts/init.py"
                )
            
            model = YOLO(str(model_path))
            logger.info(f"✓ YOLO11x loaded from {model_path}")
            logger.info("  56.9M params, 80 COCO classes")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load YOLO11x: {e}")
            raise
    
    # ========================================================================
    # CLIP Embedder
    # ========================================================================
    
    @property
    def clip(self) -> Any:
        """
        Get CLIP embedder (lazy loaded).
        
        Returns:
            CLIPEmbedder instance
        """
        if self._clip is None:
            self._clip = self._load_clip()
        return self._clip
    
    def _load_clip(self) -> Any:
        """Load CLIP model"""
        try:
            from .backends.clip_backend import CLIPEmbedder
            
            logger.info("Loading CLIP embedder...")
            
            embedder = CLIPEmbedder(
                model_name="openai/clip-vit-base-patch32",
                device=self.device
            )
            
            logger.info("✓ CLIP loaded (512-dim embeddings)")
            logger.info("  Multimodal: vision + text")
            
            return embedder
            
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            raise
    
    # ========================================================================
    # FastVLM Describer
    # ========================================================================
    
    @property
    def fastvlm(self) -> Any:
        """
        Get FastVLM describer (lazy loaded).
        
        Returns:
            FastVLM backend instance
        """
        if self._fastvlm is None:
            self._fastvlm = self._load_fastvlm()
        return self._fastvlm
    
    def _load_fastvlm(self) -> Any:
        """Load FastVLM model"""
        try:
            # Import runtime to select backend
            from .runtime import get_active_backend, select_backend
            
            logger.info("Loading FastVLM describer...")
            
            # Auto-select backend
            backend = get_active_backend()
            if backend is None:
                backend = select_backend()
            
            logger.info(f"  Backend: {backend}")
            
            # Load appropriate backend
            if backend == "mlx":
                from .backends.mlx_fastvlm import FastVLMMLXWrapper
                model = FastVLMMLXWrapper()
            else:
                from .backends.torch_fastvlm import FastVLMTorchWrapper
                model = FastVLMTorchWrapper()
            
            logger.info("✓ FastVLM loaded (0.5B params)")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load FastVLM: {e}")
            raise
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def cleanup(self):
        """Clean up all loaded models and free memory"""
        logger.info("Cleaning up models...")
        
        if self._yolo is not None:
            del self._yolo
            self._yolo = None
        
        if self._clip is not None:
            del self._clip
            self._clip = None
        
        if self._fastvlm is not None:
            del self._fastvlm
            self._fastvlm = None
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        logger.info("✓ Cleanup complete")
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage"""
        usage = {
            "yolo_loaded": self._yolo is not None,
            "clip_loaded": self._clip is not None,
            "fastvlm_loaded": self._fastvlm is not None,
        }
        
        if torch.cuda.is_available():
            usage["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
            usage["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3
        
        return usage
    
    def generate_with_ollama(
        self,
        prompt: str,
        model: str = "gemma3:4b",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate text using Ollama LLM.
        
        This is a convenience method for calling Ollama models.
        Requires ollama to be installed and running.
        
        Args:
            prompt: The prompt to send to the LLM
            model: Ollama model name (default: gemma3:4b)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 1000)
            
        Returns:
            Generated text response
            
        Raises:
            ImportError: If ollama package is not installed
            RuntimeError: If ollama service is not running
        """
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "Ollama package not installed. Install with: pip install ollama"
            )
        
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )
            return response['message']['content']
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def __repr__(self) -> str:
        return (
            f"ModelManager(device={self.device}, "
            f"yolo={'loaded' if self._yolo else 'not loaded'}, "
            f"clip={'loaded' if self._clip else 'not loaded'}, "
            f"fastvlm={'loaded' if self._fastvlm else 'not loaded'})"
        )


# Convenience function for backward compatibility
def get_model_manager() -> ModelManager:
    """Get singleton ModelManager instance"""
    return ModelManager.get_instance()
