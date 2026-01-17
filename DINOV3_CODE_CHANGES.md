# DINOv3 Implementation - Specific Code Changes

This document contains copy-paste ready code for the 6 missing pieces.

---

## File 1: orion/perception/config.py

### Change 1A: Update EmbeddingConfig dataclass (lines 435-520)

Replace the entire `EmbeddingConfig` class with:

```python
@dataclass
class EmbeddingConfig:
    """Re-ID embedding configuration.
    
    V-JEPA2 is the canonical Re-ID backbone for Orion v2.
    It provides 3D-aware video embeddings that handle viewpoint changes
    better than 2D encoders (CLIP/DINO).
    
    DINOv3 and DINOv2 are available as alternatives:
    - DINOv3: Vision Transformer with DINOv3 features (gated access, manual download)
    - DINOv2: Public DINOv2 model (automatic download)
    - V-JEPA2: Default, 3D-aware, best for Re-ID
    """
    
    # Backend selection
    backend: str = "vjepa2"
    """Embedding backend: 'vjepa2' (default), 'dinov2' (public), 'dinov3' (gated)."""
    
    # V-JEPA2 model (the default Re-ID backbone)
    model: str = "facebook/vjepa2-vitl-fpc64-256"
    """V-JEPA2 model name from HuggingFace."""
    
    # DINOv3 local weights path (required if backend='dinov3')
    dinov3_weights_dir: Optional[str] = None
    """Path to DINOv3 local weights directory (e.g., 'models/dinov3-vitb16')."""
    
    embedding_dim: int = 1024
    """Output embedding dimension (V-JEPA2 vitl = 1024, DINOv2/v3 = 768)."""

    # Cluster / memory efficiency settings
    use_cluster_embeddings: bool = False
    """If True, aggregate overlapping detections per frame into cluster embeddings to reduce memory."""

    cluster_similarity_threshold: float = 0.65
    """IoU threshold (0-1) to merge detections into same cluster before embedding extraction."""

    max_embeddings_per_entity: int = 25
    """Cap number of stored observation embeddings per entity (older ones downsampled)."""

    # Debug / verbosity
    reid_debug: bool = False
    """If True, print detailed pairwise similarity and merge decisions in Re-ID phase."""
    
    # Device selection
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    """Device for embedding model (auto/cuda/mps/cpu)"""

    # Batch processing
    batch_size: int = 16
    """Embeddings per batch. V-JEPA2 is heavier than CLIP; default lowered to 16."""
    
    def __post_init__(self):
        """Validate embedding config."""
        # Validate backend
        valid_backends = {"vjepa2", "dinov2", "dinov3"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"backend must be one of {valid_backends}, got {self.backend}"
            )
        
        # Validate DINOv3 weights path if using DINOv3
        if self.backend == "dinov3":
            if not self.dinov3_weights_dir:
                raise ValueError(
                    "backend='dinov3' requires dinov3_weights_dir. "
                    "Download from: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/"
                )
            weights_path = Path(self.dinov3_weights_dir)
            if not weights_path.exists():
                logger.warning(
                    f"DINOv3 weights directory not found: {self.dinov3_weights_dir}. "
                    f"This will fail at runtime."
                )
        
        # Auto-adjust embedding_dim based on backend
        if self.backend in {"dinov2", "dinov3"}:
            # DINO models use 768-dim embeddings
            self.embedding_dim = 768
        elif self.backend == "vjepa2":
            # V-JEPA2 uses 1024-dim embeddings
            self.embedding_dim = 1024
        
        # Validate other fields
        valid_dims = {768, 1024}
        if self.embedding_dim not in valid_dims:
            raise ValueError(
                f"embedding_dim must be one of {valid_dims}, got {self.embedding_dim}"
            )

        if not (0.0 <= self.cluster_similarity_threshold <= 1.0):
            raise ValueError(
                f"cluster_similarity_threshold must be in [0,1], got {self.cluster_similarity_threshold}"
            )
        if self.max_embeddings_per_entity < 1:
            raise ValueError(
                f"max_embeddings_per_entity must be >=1, got {self.max_embeddings_per_entity}"
            )
        
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.batch_size > 64:
            logger.warning(f"Large batch_size: {self.batch_size}. May cause OOM.")
        
        valid_devices = {"auto", "cuda", "mps", "cpu"}
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got {self.device}")
        
        logger.debug(
            f"EmbeddingConfig validated: backend={self.backend}, "
            f"model={self.model}, dim={self.embedding_dim}, "
            f"batch_size={self.batch_size}, device={self.device}"
        )
```

### Change 1B: Add imports at top of config.py (line 1-30)

Make sure `Optional` and `Path` are imported:

```python
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
```

### Change 1C: Add DINOv3 presets to PerceptionConfig class

Add these static methods to the `PerceptionConfig` class (after `accurate_preset` method):

```python
    @staticmethod
    def dinov3_preset() -> "PerceptionConfig":
        """DINOv3 (gated access) Re-ID preset.
        
        Requires manual download of DINOv3 weights from Meta:
        https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
        
        Extract to: models/dinov3-vitb16/
        
        Usage:
            config = PerceptionConfig.dinov3_preset()
            config.embedding.dinov3_weights_dir = "models/dinov3-vitb16"
            engine = PerceptionEngine(config=config)
        """
        return PerceptionConfig(
            mode="custom",
            name="dinov3",
            detection=DetectionConfig(
                backend="yolo",
                model="yolo11m",
                confidence_threshold=0.25,
                device="auto",
                enable_adaptive_confidence=True,
            ),
            embedding=EmbeddingConfig(
                backend="dinov3",
                dinov3_weights_dir="models/dinov3-vitb16",
                batch_size=32,  # DINOv3 lighter than V-JEPA2
                device="auto",
            ),
            tracking=TrackingConfig(
                enable_temporal_reid=True,
                reid_similarity_threshold=0.5,
                device="auto",
            ),
            depth=DepthConfig(enabled=True, model="depth_anything_v2", model_size="small"),
        )
    
    @staticmethod
    def dinov2_preset() -> "PerceptionConfig":
        """DINOv2 (public) Re-ID preset.
        
        Automatically downloads DINOv2 weights from Hugging Face.
        Faster inference than V-JEPA2, slightly lower Re-ID accuracy.
        """
        return PerceptionConfig(
            mode="custom",
            name="dinov2",
            detection=DetectionConfig(
                backend="yolo",
                model="yolo11m",
                confidence_threshold=0.25,
                device="auto",
                enable_adaptive_confidence=True,
            ),
            embedding=EmbeddingConfig(
                backend="dinov2",
                model="facebook/dinov2-base",
                batch_size=32,
                device="auto",
            ),
            tracking=TrackingConfig(
                enable_temporal_reid=True,
                reid_similarity_threshold=0.5,
                device="auto",
            ),
            depth=DepthConfig(enabled=True, model="depth_anything_v2", model_size="small"),
        )
```

---

## File 2: orion/perception/embedder.py

### Replace the `__init__` and add backend factory method

Replace lines 35-70 with:

```python
    def __init__(
        self,
        clip_model=None,  # Unused; kept for backward compat signature
        config: Optional[EmbeddingConfig] = None,
    ):
        """Initialize visual embedder with configurable backend.
        
        Args:
            clip_model: Unused, kept for backward compatibility
            config: EmbeddingConfig specifying backend and hyperparameters
        """
        self.config = config or EmbeddingConfig()
        self.backend = None
        
        # Resolve device
        import torch
        device = self.config.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device
        
        # Initialize backend based on config
        self._init_backend(device)
        logger.info(
            f"VisualEmbedder initialized: {self.config.backend} "
            f"(device={device}, dim={self.config.embedding_dim})"
        )
    
    def _init_backend(self, device: str):
        """Initialize embedding backend based on config.backend."""
        if self.config.backend == "vjepa2":
            from orion.backends.vjepa2_backend import VJepa2Embedder
            self.backend = VJepa2Embedder(
                model_name=self.config.model,
                device=device,
            )
            logger.debug("Initialized V-JEPA2 backend (3D-aware video embeddings)")
        
        elif self.config.backend == "dinov2":
            from orion.backends.dino_backend import DINOEmbedder
            self.backend = DINOEmbedder(
                model_name="facebook/dinov2-base",
                device=device,
            )
            logger.debug("Initialized DINOv2 backend (public, visual-only)")
        
        elif self.config.backend == "dinov3":
            from orion.backends.dino_backend import DINOEmbedder
            if not self.config.dinov3_weights_dir:
                raise ValueError(
                    "backend='dinov3' requires dinov3_weights_dir. "
                    "Download from: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/"
                )
            self.backend = DINOEmbedder(
                local_weights_dir=self.config.dinov3_weights_dir,
                device=device,
            )
            logger.debug("Initialized DINOv3 backend (gated access, visual-only)")
        
        else:
            raise ValueError(
                f"Unknown embedding backend: {self.config.backend}. "
                f"Valid options: vjepa2, dinov2, dinov3"
            )
        
        if self.backend is None:
            raise RuntimeError("Failed to initialize embedding backend")
```

### Update the `_embed_batch` method to use backend interface

Replace the `_embed_batch` method (lines ~150-180) with:

```python
    def _embed_batch(self, batch: List[dict]) -> List[np.ndarray]:
        """Embed a batch of detection crops using the configured backend.
        
        Args:
            batch: List of detection dicts with 'crop' field
            
        Returns:
            List of normalized embedding vectors
        """
        crops = []
        valid_indices = []
        
        for i, detection in enumerate(batch):
            if 'crop' in detection and detection['crop'] is not None:
                crops.append(detection['crop'])
                valid_indices.append(i)
        
        if not crops:
            # Return zero embeddings if no valid crops
            return [np.zeros(self.config.embedding_dim, dtype=np.float32) for _ in batch]
        
        # Use batch encoding if available, else fall back to single encoding
        if hasattr(self.backend, 'encode_images_batch') and len(crops) > 1:
            embeddings = self.backend.encode_images_batch(crops, normalize=True)
        else:
            embeddings = [self.backend.encode_image(crop, normalize=True) for crop in crops]
        
        # Reconstruct full batch with zero embeddings for invalid crops
        result = []
        emb_idx = 0
        for i in range(len(batch)):
            if i in valid_indices:
                result.append(embeddings[emb_idx])
                emb_idx += 1
            else:
                result.append(np.zeros(self.config.embedding_dim, dtype=np.float32))
        
        return result
```

---

## File 3: orion/cli/run_showcase.py

### Add CLI arguments (find the argument parser section)

Add these lines to the argument parser (after existing `--detection-backend` argument):

```python
    # Embedding backend selection
    parser.add_argument(
        "--embedding-backend",
        choices=["vjepa2", "dinov2", "dinov3"],
        default="vjepa2",
        help="Visual embedding backend for Re-ID. "
             "vjepa2: 3D-aware (default), dinov2: public DINO, dinov3: gated DINO",
    )
    
    parser.add_argument(
        "--dinov3-weights",
        type=str,
        default=None,
        help="Path to DINOv3 local weights (required if --embedding-backend=dinov3). "
             "Download from: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/",
    )
```

### Apply backend choice in main() function

Find the section where `PerceptionConfig` is created and add:

```python
    # Apply embedding backend selection
    if args.embedding_backend == "dinov3":
        if not args.dinov3_weights:
            raise ValueError(
                "dinov3 backend requires --dinov3-weights argument. "
                "Download from: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/"
            )
        config.embedding.backend = "dinov3"
        config.embedding.dinov3_weights_dir = args.dinov3_weights
    elif args.embedding_backend == "dinov2":
        config.embedding.backend = "dinov2"
    elif args.embedding_backend == "vjepa2":
        config.embedding.backend = "vjepa2"
    
    logger.info(f"Using embedding backend: {config.embedding.backend}")
```

---

## File 4: scripts/setup_dinov3.py (NEW FILE)

Create new file with:

```python
#!/usr/bin/env python3
"""
DINOv3 Setup Verification Script

Verifies that DINOv3 weights are properly downloaded and located.
"""

import sys
import json
from pathlib import Path


def verify_dinov3_weights(weights_dir: str) -> bool:
    """Verify DINOv3 weights structure and file sizes."""
    path = Path(weights_dir)
    
    print(f"\nVerifying DINOv3 weights at: {path}")
    
    if not path.exists():
        print(f"❌ Weights directory not found: {weights_dir}")
        return False
    
    print(f"✅ Directory exists")
    
    # Check required files
    required_files = {
        "pytorch_model.bin": (300 * 1024**2, 400 * 1024**2),  # 300-400 MB for ViT-B
        "config.json": (1024, 10 * 1024),  # 1KB - 10KB
        "preprocessor_config.json": (1024, 10 * 1024),  # 1KB - 10KB
    }
    
    all_valid = True
    for filename, (min_size, max_size) in required_files.items():
        filepath = path / filename
        if not filepath.exists():
            print(f"❌ Missing: {filename}")
            all_valid = False
        else:
            size = filepath.stat().st_size
            if not (min_size <= size <= max_size):
                print(f"⚠️  {filename}: {size / 1024**2:.1f}MB (expected {min_size / 1024**2:.0f}-{max_size / 1024**2:.0f}MB)")
            else:
                print(f"✅ {filename}: {size / 1024**2:.1f}MB")
    
    if not all_valid:
        return False
    
    # Try to load config
    try:
        with open(path / "config.json") as f:
            config = json.load(f)
        print(f"✅ Config loaded successfully")
        
        # Print architecture info
        if "hidden_size" in config:
            print(f"   Architecture: ViT with {config['hidden_size']} hidden dim")
        if "num_hidden_layers" in config:
            print(f"   Layers: {config['num_hidden_layers']}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False
    
    return True


def main():
    print("=" * 80)
    print("DINOv3 SETUP VERIFICATION")
    print("=" * 80)
    
    weights_dir = "models/dinov3-vitb16"
    
    if verify_dinov3_weights(weights_dir):
        print("\n" + "=" * 80)
        print("✅ DINOv3 IS READY TO USE!")
        print("=" * 80)
        print("\nUsage examples:")
        print("\n1. CLI:")
        print(f"   python -m orion.cli.run_showcase \\")
        print(f"     --embedding-backend dinov3 \\")
        print(f"     --dinov3-weights {weights_dir} \\")
        print(f"     --episode my_video --video video.mp4")
        
        print("\n2. Python:")
        print(f"   from orion.perception.config import PerceptionConfig")
        print(f"   config = PerceptionConfig.dinov3_preset()")
        print(f"   config.embedding.dinov3_weights_dir = '{weights_dir}'")
        print(f"   engine = PerceptionEngine(config=config)")
        
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ DINOv3 SETUP INCOMPLETE")
        print("=" * 80)
        print("\nSetup instructions:")
        print("1. Download DINOv3 weights from:")
        print("   https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/")
        print("\n2. Extract to:")
        print(f"   {weights_dir}/")
        print("\n3. Verify with:")
        print(f"   python scripts/setup_dinov3.py")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## File 5: scripts/test_dinov3_reid.py (NEW FILE)

Create new file with:

```python
#!/usr/bin/env python3
"""
Test DINOv3 Re-ID end-to-end.

Tests:
- DINOv3 backend initialization
- Image encoding
- Batch processing
- Similarity matching
"""

import sys
import numpy as np
from pathlib import Path


def test_dinov3_backend():
    """Test DINOv3 backend initialization."""
    print("\n" + "=" * 80)
    print("TEST 1: DINOv3 Backend Initialization")
    print("=" * 80)
    
    from orion.backends.dino_backend import DINOEmbedder
    
    weights_dir = "models/dinov3-vitb16"
    if not Path(weights_dir).exists():
        print(f"❌ SKIP: DINOv3 weights not found at {weights_dir}")
        return False
    
    try:
        embedder = DINOEmbedder(local_weights_dir=weights_dir, device="mps")
        print(f"✅ DINOEmbedder initialized")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dinov3_encoding():
    """Test single image encoding."""
    print("\n" + "=" * 80)
    print("TEST 2: Single Image Encoding")
    print("=" * 80)
    
    from orion.backends.dino_backend import DINOEmbedder
    
    weights_dir = "models/dinov3-vitb16"
    if not Path(weights_dir).exists():
        print(f"❌ SKIP: DINOv3 weights not found")
        return False
    
    try:
        embedder = DINOEmbedder(local_weights_dir=weights_dir, device="mps")
        
        # Create dummy image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Encode
        embedding = embedder.encode_image(test_image, normalize=True)
        
        print(f"✅ Encoding successful")
        print(f"   Shape: {embedding.shape}")
        print(f"   Dtype: {embedding.dtype}")
        print(f"   L2 norm: {np.linalg.norm(embedding):.6f} (should be ~1.0)")
        
        # Validate
        assert embedding.shape == (768,), f"Expected (768,), got {embedding.shape}"
        assert 0.99 <= np.linalg.norm(embedding) <= 1.01, "Not L2-normalized"
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dinov3_batch():
    """Test batch encoding."""
    print("\n" + "=" * 80)
    print("TEST 3: Batch Encoding")
    print("=" * 80)
    
    from orion.backends.dino_backend import DINOEmbedder
    
    weights_dir = "models/dinov3-vitb16"
    if not Path(weights_dir).exists():
        print(f"❌ SKIP: DINOv3 weights not found")
        return False
    
    try:
        embedder = DINOEmbedder(local_weights_dir=weights_dir, device="mps")
        
        # Create dummy batch
        batch_size = 4
        images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(batch_size)]
        
        # Encode batch
        embeddings = embedder.encode_images_batch(images, normalize=True)
        
        print(f"✅ Batch encoding successful")
        print(f"   Batch size: {len(embeddings)}")
        print(f"   Individual shapes: {[e.shape for e in embeddings[:2]]} ...")
        
        # Validate
        assert len(embeddings) == batch_size
        assert all(e.shape == (768,) for e in embeddings)
        assert all(0.99 <= np.linalg.norm(e) <= 1.01 for e in embeddings)
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dinov3_reid():
    """Test Re-ID similarity matching."""
    print("\n" + "=" * 80)
    print("TEST 4: Re-ID Similarity Matching")
    print("=" * 80)
    
    from orion.perception.config import EmbeddingConfig
    from orion.perception.embedder import VisualEmbedder
    
    weights_dir = "models/dinov3-vitb16"
    if not Path(weights_dir).exists():
        print(f"❌ SKIP: DINOv3 weights not found")
        return False
    
    try:
        # Initialize embedder with DINOv3
        config = EmbeddingConfig(
            backend="dinov3",
            dinov3_weights_dir=weights_dir,
        )
        embedder = VisualEmbedder(config=config)
        print(f"✅ VisualEmbedder initialized with DINOv3")
        
        # Create detections with crops
        detections = [
            {"crop": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8), "id": i}
            for i in range(3)
        ]
        
        # Embed
        detections = embedder.embed_detections(detections)
        print(f"✅ Embedded {len(detections)} detections")
        
        # Compute similarities
        emb1 = detections[0]['embedding']
        emb2 = detections[1]['embedding']
        sim = np.dot(emb1, emb2)
        
        print(f"✅ Cosine similarity: {sim:.4f}")
        print(f"   (Random crops should have ~0.0 similarity)")
        
        # Validate
        assert emb1.shape == (768,)
        assert -1.0 <= sim <= 1.0
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("DINOv3 RE-ID TESTING SUITE")
    print("=" * 80)
    
    results = {
        "Backend Initialization": test_dinov3_backend(),
        "Single Image Encoding": test_dinov3_encoding(),
        "Batch Encoding": test_dinov3_batch(),
        "Re-ID Matching": test_dinov3_reid(),
    }
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED - DINOv3 RE-ID IS WORKING!")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ SOME TESTS FAILED - CHECK ABOVE FOR DETAILS")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## Summary of Changes

| File | Lines | Changes |
|------|-------|---------|
| `orion/perception/config.py` | ~100 | Add `backend` & `dinov3_weights_dir` to `EmbeddingConfig`, add DINOv3/v2 presets |
| `orion/perception/embedder.py` | ~80 | Add backend factory in `__init__`, update `_embed_batch` |
| `orion/cli/run_showcase.py` | ~20 | Add `--embedding-backend` and `--dinov3-weights` arguments |
| `scripts/setup_dinov3.py` | NEW | 100 lines, verify DINOv3 weights |
| `scripts/test_dinov3_reid.py` | NEW | 200 lines, test DINOv3 end-to-end |

**Total**: ~500 lines (mostly new, minimal breaking changes)

