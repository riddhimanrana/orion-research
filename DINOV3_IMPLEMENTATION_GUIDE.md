# DINOv3 Full Implementation Guide

## Status: 85% Complete

DINOv3 backend framework exists but **requires final integration steps** to be production-ready.

---

## What's Already Done ✅

### 1. **DINOEmbedder Backend** (`orion/backends/dino_backend.py`)
- ✅ Loads DINOv3 from local weights (transformer-based)
- ✅ Fallback to DINOv2 via Hugging Face (public model)
- ✅ Fallback to timm (if needed)
- ✅ Image encoding with L2 normalization
- ✅ Batch processing support
- ✅ Device auto-detection (cuda/mps/cpu)
- ✅ Feature map extraction for spatial pooling
- ✅ Region pooling for bbox-level embeddings

### 2. **Configuration** (`orion/perception/config.py`)
- ✅ `EmbeddingConfig` dataclass exists
- ✅ Validation logic in place
- ✅ Default to V-JEPA2
- ✅ Device selection (auto/cuda/mps/cpu)
- ✅ Batch size configuration

### 3. **Visual Embedder** (`orion/perception/embedder.py`)
- ✅ Uses V-JEPA2 for Re-ID
- ✅ Batch processing logic
- ✅ Detection embedding pipeline

---

## What's Missing ❌ & How to Fix It

### **Missing Piece 1: EmbeddingConfig Backend Selection**

**Problem**: `EmbeddingConfig` hardcodes V-JEPA2. No way to switch to DINOv3.

**Solution**: Add `backend` field to `EmbeddingConfig`:

```python
@dataclass
class EmbeddingConfig:
    """Re-ID embedding configuration."""
    
    # Backend selection
    backend: str = "vjepa2"  # "vjepa2" | "dinov2" | "dinov3"
    """Embedding backend: vjepa2 (3D-aware), dinov2 (public), dinov3 (gated)."""
    
    model: str = "facebook/vjepa2-vitl-fpc64-256"
    """Model name/path. For dinov3, use local_weights_dir instead."""
    
    # DINOv3 local weights
    dinov3_weights_dir: Optional[str] = None
    """Path to DINOv3 local weights (e.g., models/dinov3-vitb16)."""
    
    embedding_dim: int = 1024
    """Output embedding dimension."""
    
    def __post_init__(self):
        """Validate and normalize config."""
        valid_backends = {"vjepa2", "dinov2", "dinov3"}
        if self.backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}, got {self.backend}")
        
        # Auto-adjust embedding_dim based on model
        if self.backend == "dinov2":
            # DINOv2-base: 768 dims
            self.embedding_dim = 768
        elif self.backend == "dinov3":
            # DINOv3-ViT-B/16: 768 dims
            self.embedding_dim = 768
        # else V-JEPA2: keep 1024
```

**File**: `orion/perception/config.py` (around line 435)

---

### **Missing Piece 2: VisualEmbedder Backend Factory**

**Problem**: `VisualEmbedder` only uses V-JEPA2. No factory pattern to switch backends.

**Solution**: Add backend factory:

```python
class VisualEmbedder:
    """Generates visual embeddings using configurable backend."""

    def __init__(
        self,
        clip_model=None,  # Unused; kept for backward compat
        config: Optional[EmbeddingConfig] = None,
    ):
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
        
        # Initialize backend
        self._init_backend(device)
        logger.info(f"VisualEmbedder initialized: {self.config.backend} (device={device}, dim={self.config.embedding_dim})")
    
    def _init_backend(self, device: str):
        """Initialize embedding backend based on config."""
        if self.config.backend == "vjepa2":
            from orion.backends.vjepa2_backend import VJepa2Embedder
            self.backend = VJepa2Embedder(
                model_name=self.config.model,
                device=device,
            )
        elif self.config.backend == "dinov2":
            from orion.backends.dino_backend import DINOEmbedder
            self.backend = DINOEmbedder(
                model_name="facebook/dinov2-base",
                device=device,
            )
        elif self.config.backend == "dinov3":
            from orion.backends.dino_backend import DINOEmbedder
            if not self.config.dinov3_weights_dir:
                raise ValueError(
                    "dinov3 backend requires dinov3_weights_dir. "
                    "Download from: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/"
                )
            self.backend = DINOEmbedder(
                local_weights_dir=self.config.dinov3_weights_dir,
                device=device,
            )
        else:
            raise ValueError(f"Unknown embedding backend: {self.config.backend}")
    
    def embed_detections(self, detections: List[dict]) -> List[dict]:
        """Delegate to backend."""
        if not hasattr(self.backend, 'encode_image'):
            raise RuntimeError("Backend does not have encode_image method")
        
        # Batch processing
        batch_size = self.config.batch_size
        for batch_idx in range(0, len(detections), batch_size):
            batch = detections[batch_idx:batch_idx + batch_size]
            embeddings = self._embed_batch(batch)
            for detection, embedding in zip(batch, embeddings):
                detection["embedding"] = embedding
        
        return detections
    
    def _embed_batch(self, batch: List[dict]) -> List[np.ndarray]:
        """Embed a batch using backend."""
        crops = [d['crop'] for d in batch if 'crop' in d]
        if not crops:
            return [np.zeros(self.config.embedding_dim) for _ in batch]
        
        if hasattr(self.backend, 'encode_images_batch'):
            # Use batch method if available
            embeddings = self.backend.encode_images_batch(crops, normalize=True)
        else:
            # Fallback to single encoding
            embeddings = [self.backend.encode_image(crop, normalize=True) for crop in crops]
        
        return embeddings
```

**File**: `orion/perception/embedder.py` (replace lines 35-70)

---

### **Missing Piece 3: EmbeddingConfig Presets with DINOv3**

**Problem**: `PerceptionConfig` presets don't offer DINOv3 variants.

**Solution**: Add DINOv3 presets in config.py:

```python
@staticmethod
def dinov3_preset() -> PerceptionConfig:
    """DINOv3 (gated) visual-only Re-ID preset.
    
    Requires local weights from Meta.
    """
    return PerceptionConfig(
        mode="custom",  # Not auto; manual setup required
        detection=DetectionConfig(
            backend="yolo",
            model="yolo11m",
            confidence_threshold=0.25,
        ),
        embedding=EmbeddingConfig(
            backend="dinov3",
            dinov3_weights_dir="models/dinov3-vitb16",
            embedding_dim=768,
            batch_size=32,  # DINOv3 is lighter than V-JEPA2
        ),
        # ... rest of config
    )

@staticmethod
def dinov2_preset() -> PerceptionConfig:
    """DINOv2 (public) visual-only Re-ID preset."""
    return PerceptionConfig(
        mode="custom",
        detection=DetectionConfig(backend="yolo", model="yolo11m"),
        embedding=EmbeddingConfig(
            backend="dinov2",
            model="facebook/dinov2-base",
            embedding_dim=768,
            batch_size=32,
        ),
    )
```

**File**: `orion/perception/config.py` (add to PerceptionConfig class)

---

### **Missing Piece 4: CLI Support for DINOv3**

**Problem**: `run_showcase` CLI doesn't expose embedding backend selection.

**Solution**: Add CLI argument in `orion/cli/run_showcase.py`:

```python
import argparse

parser.add_argument(
    "--embedding-backend",
    choices=["vjepa2", "dinov2", "dinov3"],
    default="vjepa2",
    help="Visual embedding backend for Re-ID: vjepa2 (default, 3D-aware), dinov2 (public), dinov3 (gated)",
)

parser.add_argument(
    "--dinov3-weights",
    type=str,
    default="models/dinov3-vitb16",
    help="Path to DINOv3 local weights (required if --embedding-backend=dinov3)",
)

# In main():
if args.embedding_backend == "dinov3":
    config.embedding.backend = "dinov3"
    config.embedding.dinov3_weights_dir = args.dinov3_weights
elif args.embedding_backend == "dinov2":
    config.embedding.backend = "dinov2"
elif args.embedding_backend == "vjepa2":
    config.embedding.backend = "vjepa2"
```

---

### **Missing Piece 5: Download & Setup Instructions**

**Problem**: Users don't know how to get DINOv3 weights.

**Solution**: Create `scripts/setup_dinov3.py`:

```python
#!/usr/bin/env python3
"""
DINOv3 Setup Helper

DINOv3 requires manual download from Meta (gated access).

Steps:
1. Go to: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
2. Request access (free, instant approval usually)
3. Download ViT-B/16 or ViT-L/14 model
4. Extract to: models/dinov3-vitb16/ or models/dinov3-vitl14/
5. Run: python scripts/test_dinov3_setup.py
"""

import os
import json
from pathlib import Path

def verify_dinov3_weights(weights_dir: str) -> bool:
    """Verify DINOv3 weights structure."""
    path = Path(weights_dir)
    
    if not path.exists():
        print(f"❌ Weights directory not found: {weights_dir}")
        return False
    
    required_files = ["pytorch_model.bin", "config.json", "preprocessor_config.json"]
    missing = [f for f in required_files if not (path / f).exists()]
    
    if missing:
        print(f"❌ Missing files: {', '.join(missing)}")
        return False
    
    # Check file sizes (rough validation)
    pytorch_size = (path / "pytorch_model.bin").stat().st_size
    if pytorch_size < 300 * 1024 * 1024:  # Should be ~330MB for ViT-B
        print(f"⚠️  pytorch_model.bin seems small ({pytorch_size / 1024**2:.0f}MB). May be corrupted.")
        return False
    
    print(f"✅ DINOv3 weights verified at {weights_dir}")
    return True

def main():
    print("=" * 80)
    print("DINOv3 SETUP HELPER")
    print("=" * 80)
    
    weights_dir = "models/dinov3-vitb16"
    
    if verify_dinov3_weights(weights_dir):
        print("\n✅ DINOv3 is ready to use!")
        print("\nUsage:")
        print(f"  python -m orion.cli.run_showcase --embedding-backend dinov3 \\")
        print(f"    --dinov3-weights {weights_dir} --episode test_video --video video.mp4")
    else:
        print("\n❌ DINOv3 weights not found or invalid")
        print("\nSetup instructions:")
        print("1. Download from: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/")
        print(f"2. Extract to: {weights_dir}/")
        print("3. Run: python scripts/setup_dinov3.py")

if __name__ == "__main__":
    main()
```

---

### **Missing Piece 6: Tests & Validation**

**Problem**: No validation that DINOv3 works end-to-end.

**Solution**: Create `scripts/test_dinov3_reid.py`:

```python
#!/usr/bin/env python3
"""
Test DINOv3 Re-ID end-to-end:
- Embedding generation
- Similarity matching
- Memory clustering
"""

import numpy as np
from pathlib import Path
from orion.perception.config import EmbeddingConfig
from orion.perception.embedder import VisualEmbedder

def test_dinov3_backend():
    """Test DINOv3 backend initialization and encoding."""
    print("\n" + "=" * 80)
    print("TEST: DINOv3 Backend")
    print("=" * 80)
    
    config = EmbeddingConfig(
        backend="dinov3",
        dinov3_weights_dir="models/dinov3-vitb16",
    )
    
    embedder = VisualEmbedder(config=config)
    
    # Test encoding
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    embedding = embedder.backend.encode_image(test_image)
    
    print(f"✅ DINOv3 embedding shape: {embedding.shape}")
    print(f"✅ L2 norm: {np.linalg.norm(embedding):.4f} (should be ~1.0)")
    
    assert embedding.shape == (768,), f"Expected shape (768,), got {embedding.shape}"
    assert 0.99 <= np.linalg.norm(embedding) <= 1.01, "Embedding not L2-normalized"
    
    return True

def test_dinov3_reid():
    """Test Re-ID with DINOv3."""
    print("\n" + "=" * 80)
    print("TEST: DINOv3 Re-ID Matching")
    print("=" * 80)
    
    config = EmbeddingConfig(backend="dinov3", dinov3_weights_dir="models/dinov3-vitb16")
    embedder = VisualEmbedder(config=config)
    
    # Create fake detections with crops
    detections = [
        {"crop": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8), "id": 1}
        for _ in range(5)
    ]
    
    # Embed detections
    detections = embedder.embed_detections(detections)
    
    print(f"✅ Embedded {len(detections)} detections")
    print(f"✅ Sample embedding shape: {detections[0]['embedding'].shape}")
    
    # Test similarity
    emb1 = detections[0]['embedding']
    emb2 = detections[1]['embedding']
    sim = np.dot(emb1, emb2)
    print(f"✅ Cosine similarity between two random crops: {sim:.4f}")
    
    return True

def main():
    try:
        # Check weights exist
        weights_path = Path("models/dinov3-vitb16")
        if not weights_path.exists():
            print(f"❌ DINOv3 weights not found at {weights_path}")
            print("Download from: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/")
            return False
        
        # Run tests
        test_dinov3_backend()
        test_dinov3_reid()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        return True
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
```

---

## Implementation Checklist

- [ ] **Step 1**: Add `backend` field to `EmbeddingConfig` (config.py, ~20 lines)
- [ ] **Step 2**: Add backend factory to `VisualEmbedder` (embedder.py, ~60 lines)
- [ ] **Step 3**: Add DINOv3 presets to `PerceptionConfig` (config.py, ~30 lines)
- [ ] **Step 4**: Add CLI arguments in `run_showcase.py` (~15 lines)
- [ ] **Step 5**: Create setup helper script (scripts/setup_dinov3.py, ~50 lines)
- [ ] **Step 6**: Create test script (scripts/test_dinov3_reid.py, ~100 lines)
- [ ] **Step 7**: Update README with DINOv3 instructions

**Total Implementation Time**: ~2 hours  
**Total Code Changes**: ~300 lines (mostly additions, no breaking changes)

---

## Usage After Implementation

### Option A: Direct Python
```python
from orion.perception.config import PerceptionConfig
from orion.perception.engine import PerceptionEngine

config = PerceptionConfig.dinov3_preset()
config.embedding.dinov3_weights_dir = "models/dinov3-vitb16"

engine = PerceptionEngine(config=config)
results = engine.process_video(video_path)
```

### Option B: CLI
```bash
python -m orion.cli.run_showcase \
  --episode my_video \
  --video video.mp4 \
  --embedding-backend dinov3 \
  --dinov3-weights models/dinov3-vitb16
```

### Option C: Config File
```yaml
perception:
  embedding:
    backend: dinov3
    dinov3_weights_dir: models/dinov3-vitb16
    batch_size: 32
```

---

## Comparison: V-JEPA2 vs DINOv2 vs DINOv3

| Feature | V-JEPA2 | DINOv2 | DINOv3 |
|---------|---------|--------|--------|
| **3D-aware** | ✅ Yes | ❌ No | ✅ Yes |
| **Video-native** | ✅ Yes | ❌ No | ✅ Video-optimized |
| **Public** | ⚠️ Research | ✅ Yes | ❌ Gated |
| **Dim** | 1024 | 768 | 768 |
| **Speed** | Medium | Fast | Fast |
| **Re-ID Accuracy** | Highest | Medium | High |
| **Setup** | Auto-download | Auto-download | Manual download |

**Recommendation**: V-JEPA2 (default) for best Re-ID. DINOv3 as alternative if gated access approved.

