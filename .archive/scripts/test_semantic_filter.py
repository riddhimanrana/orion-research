#!/usr/bin/env python3
"""
Test script for SemanticFilter validation.

Tests the semantic filtering pipeline:
1. Load test images/crops
2. Run through SemanticFilter
3. Validate results

Usage:
    python scripts/test_semantic_filter.py [--device cuda|mps|cpu]
"""

import argparse
import logging
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.filters import (
    SemanticFilter,
    SemanticFilterConfig,
    create_semantic_filter,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_crops() -> list[tuple[Image.Image, str]]:
    """Create synthetic test crops for validation.
    
    Returns list of (image, expected_label) tuples.
    """
    crops = []
    
    # Simple colored rectangles as stand-ins
    # In real usage, these would be actual object crops
    
    # Create a simple "chair-like" brown image
    chair_img = Image.new('RGB', (224, 224), color=(139, 90, 43))
    crops.append((chair_img, "chair"))
    
    # Create a "monitor-like" dark gray image with blue center
    monitor_img = Image.new('RGB', (224, 224), color=(50, 50, 50))
    # Add blue "screen" center
    screen = Image.new('RGB', (180, 120), color=(30, 100, 180))
    monitor_img.paste(screen, (22, 52))
    crops.append((monitor_img, "monitor"))
    
    # Create a "person-like" image (skin tone + clothing)
    person_img = Image.new('RGB', (224, 224), color=(100, 80, 200))  # Purple shirt
    crops.append((person_img, "person"))
    
    return crops


def test_sentence_transformer_only():
    """Test that sentence transformer loads and embeds correctly."""
    logger.info("=" * 60)
    logger.info("TEST 1: Sentence Transformer Embedding")
    logger.info("=" * 60)
    
    from orion.perception.filters import get_sentence_transformer
    
    model = get_sentence_transformer()
    
    # Test embeddings
    test_texts = [
        "A wooden chair in a room",
        "A chair made of wood",  # Should be very similar
        "A large dog running",    # Should be different
    ]
    
    embeddings = model.encode(test_texts, convert_to_numpy=True)
    
    # Compute cosine similarities
    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    sim_chair_chair = cosine_sim(embeddings[0], embeddings[1])
    sim_chair_dog = cosine_sim(embeddings[0], embeddings[2])
    
    logger.info(f"  Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"  'wooden chair' vs 'chair made of wood': {sim_chair_chair:.3f}")
    logger.info(f"  'wooden chair' vs 'dog running': {sim_chair_dog:.3f}")
    
    # Validate
    assert sim_chair_chair > 0.7, f"Similar texts should have high similarity, got {sim_chair_chair}"
    assert sim_chair_dog < 0.5, f"Different texts should have low similarity, got {sim_chair_dog}"
    
    logger.info("  ✓ Sentence transformer test passed!")
    return True


def test_label_expansion():
    """Test label embedding with expansion."""
    logger.info("=" * 60)
    logger.info("TEST 2: Label Expansion and Caching")
    logger.info("=" * 60)
    
    filter = SemanticFilter()
    
    # Get embeddings for various labels
    labels = ["chair", "person", "monitor", "table"]
    
    for label in labels:
        emb = filter.get_label_embedding(label)
        logger.info(f"  {label}: shape={emb.shape}")
        
    # Test cache hit
    emb1 = filter.get_label_embedding("chair")
    emb2 = filter.get_label_embedding("chair")
    assert np.allclose(emb1, emb2), "Cache should return same embedding"
    
    logger.info("  ✓ Label expansion test passed!")
    return True


def test_similarity_computation():
    """Test cosine similarity computation."""
    logger.info("=" * 60)
    logger.info("TEST 3: Similarity Computation")
    logger.info("=" * 60)
    
    filter = SemanticFilter()
    
    # Test with known embeddings
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    d = np.array([-1.0, 0.0, 0.0])
    
    sim_same = filter.cosine_similarity(a, b)
    sim_ortho = filter.cosine_similarity(a, c)
    sim_opposite = filter.cosine_similarity(a, d)
    
    logger.info(f"  Same vectors: {sim_same:.3f} (expected 1.0)")
    logger.info(f"  Orthogonal vectors: {sim_ortho:.3f} (expected 0.0)")
    logger.info(f"  Opposite vectors: {sim_opposite:.3f} (expected -1.0)")
    
    assert abs(sim_same - 1.0) < 0.001, f"Same vectors should have sim=1.0, got {sim_same}"
    assert abs(sim_ortho - 0.0) < 0.001, f"Orthogonal should have sim=0.0, got {sim_ortho}"
    assert abs(sim_opposite - (-1.0)) < 0.001, f"Opposite should have sim=-1.0, got {sim_opposite}"
    
    logger.info("  ✓ Similarity computation test passed!")
    return True


def test_vlm_description(device: str = "cuda"):
    """Test VLM description generation."""
    logger.info("=" * 60)
    logger.info(f"TEST 4: VLM Description Generation (device={device})")
    logger.info("=" * 60)
    
    config = SemanticFilterConfig(device=device)
    filter = SemanticFilter(config)
    
    # Create a simple test image
    test_img = Image.new('RGB', (224, 224), color=(139, 90, 43))  # Brown
    
    logger.info("  Generating description...")
    desc = filter.generate_description(test_img)
    logger.info(f"  Description: '{desc}'")
    
    assert len(desc) > 0, "Description should not be empty"
    
    logger.info("  ✓ VLM description test passed!")
    return True


def test_filter_single(device: str = "cuda"):
    """Test filtering a single detection."""
    logger.info("=" * 60)
    logger.info(f"TEST 5: Single Detection Filtering (device={device})")
    logger.info("=" * 60)
    
    filter = create_semantic_filter(device=device, similarity_threshold=0.2)
    
    # Create test image
    test_img = Image.new('RGB', (224, 224), color=(139, 90, 43))
    
    result = filter.filter_single(
        crop=test_img,
        label="wooden object",
        track_id=1,
        confidence=0.85,
    )
    
    logger.info(f"  Track ID: {result.track_id}")
    logger.info(f"  Label: {result.label}")
    logger.info(f"  Description: {result.description}")
    logger.info(f"  Similarity: {result.similarity:.3f}")
    logger.info(f"  Is Valid: {result.is_valid}")
    logger.info(f"  Reason: {result.reason or 'N/A'}")
    
    logger.info("  ✓ Single filter test passed!")
    return True


def test_filter_batch(device: str = "cuda"):
    """Test batch filtering."""
    logger.info("=" * 60)
    logger.info(f"TEST 6: Batch Filtering (device={device})")
    logger.info("=" * 60)
    
    filter = create_semantic_filter(device=device, similarity_threshold=0.2)
    
    # Create test data
    n = 3
    crops = [Image.new('RGB', (224, 224), color=(100 + i*30, 80, 50)) for i in range(n)]
    labels = ["furniture", "object", "item"]
    track_ids = list(range(n))
    confidences = [0.9, 0.8, 0.7]
    
    results = filter.filter_batch(
        crops=crops,
        labels=labels,
        track_ids=track_ids,
        confidences=confidences,
        show_progress=True,
    )
    
    logger.info(f"  Processed {len(results)} detections:")
    for r in results:
        status = "✓" if r.is_valid else "✗"
        logger.info(f"    [{status}] Track {r.track_id}: {r.label} -> '{r.description[:40]}...' (sim={r.similarity:.2f})")
    
    valid = sum(1 for r in results if r.is_valid)
    logger.info(f"  Valid: {valid}/{n}")
    
    logger.info("  ✓ Batch filter test passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test SemanticFilter")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"],
                       help="Device for VLM inference")
    parser.add_argument("--skip-vlm", action="store_true",
                       help="Skip VLM tests (useful for quick embedding-only tests)")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("SemanticFilter Test Suite")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: Sentence transformer
    try:
        results["sentence_transformer"] = test_sentence_transformer_only()
    except Exception as e:
        logger.error(f"  ✗ Sentence transformer test failed: {e}")
        results["sentence_transformer"] = False
    
    # Test 2: Label expansion
    try:
        results["label_expansion"] = test_label_expansion()
    except Exception as e:
        logger.error(f"  ✗ Label expansion test failed: {e}")
        results["label_expansion"] = False
    
    # Test 3: Similarity computation
    try:
        results["similarity"] = test_similarity_computation()
    except Exception as e:
        logger.error(f"  ✗ Similarity computation test failed: {e}")
        results["similarity"] = False
    
    if not args.skip_vlm:
        # Test 4: VLM description
        try:
            results["vlm_description"] = test_vlm_description(args.device)
        except Exception as e:
            logger.error(f"  ✗ VLM description test failed: {e}")
            results["vlm_description"] = False
        
        # Test 5: Single filter
        try:
            results["filter_single"] = test_filter_single(args.device)
        except Exception as e:
            logger.error(f"  ✗ Single filter test failed: {e}")
            results["filter_single"] = False
        
        # Test 6: Batch filter
        try:
            results["filter_batch"] = test_filter_batch(args.device)
        except Exception as e:
            logger.error(f"  ✗ Batch filter test failed: {e}")
            results["filter_batch"] = False
    else:
        logger.info("Skipping VLM tests (--skip-vlm)")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"  {name}: {status}")
    
    logger.info("=" * 60)
    logger.info(f"Result: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
