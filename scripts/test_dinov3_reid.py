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
