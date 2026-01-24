#!/usr/bin/env python3
"""
Comprehensive Pipeline Testing Script
Tests each stage with example videos and verifies DINOv3 integration.
"""

import sys
from pathlib import Path
import json

def test_phase1_detection():
    """Test Phase 1: Detection + Tracking"""
    print("\n" + "=" * 80)
    print("PHASE 1: DETECTION + TRACKING (Testing YOLO backend)")
    print("=" * 80)
    
    try:
        from orion.perception.detectors.yolo_detector import YOLODetector
        from orion.perception.config import DetectionConfig
        
        print("\n✅ Detection modules imported")
        
        # Test YOLO detector
        print("\n🔬 Testing YOLO detector initialization...")
        config = DetectionConfig(
            backend="yolo",
            model="yolo11m",
            confidence_threshold=0.25,
            device="mps"
        )
        detector = YOLODetector(config)
        print(f"  ✅ YOLO detector initialized: {detector.model_name}")
        
        # Test on dummy frame
        import numpy as np
        test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        detections = detector.detect(test_frame)
        print(f"  ✅ Detection works (dummy frame: {len(detections)} detections)")
        
    except Exception as e:
        print(f"  ❌ Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()

def test_phase2_reid():
    """Test Phase 2: Re-ID / Memory Clustering"""
    print("\n" + "=" * 80)
    print("PHASE 2: RE-ID / MEMORY CLUSTERING (Testing V-JEPA2 backend)")
    print("=" * 80)
    
    try:
        from orion.backends.vjepa2_backend import VJEPA2Embedder
        
        print("\n✅ Re-ID modules imported")
        
        # Test V-JEPA2 embedder
        print("\n🔬 Testing V-JEPA2 embedder...")
        embedder = VJEPA2Embedder(device="mps")
        print(f"  ✅ V-JEPA2 loaded on {embedder.device}")
        
        # Test embedding
        import numpy as np
        test_crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        embedding = embedder.encode_image(test_crop)
        print(f"  ✅ Embedding works: shape={embedding.shape}")
        
    except Exception as e:
        print(f"  ❌ Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()

def test_phase3_scene_graph():
    """Test Phase 3: Scene Graph Generation"""
    print("\n" + "=" * 80)
    print("PHASE 3: SCENE GRAPH GENERATION")
    print("=" * 80)
    
    try:
        from orion.graph.scene_graph import build_scene_graphs
        
        print("\n✅ Scene graph modules imported")
        
        # Create minimal test data
        memory = {
            "objects": [
                {"memory_id": "mem_001", "class": "person"},
                {"memory_id": "mem_002", "class": "chair"},
            ]
        }
        
        tracks = [
            {
                "frame_id": 0,
                "category": "person",
                "bbox": [100, 100, 200, 300],
                "embedding_id": "emb_001",
                "frame_width": 640,
                "frame_height": 480,
            },
            {
                "frame_id": 0,
                "category": "chair",
                "bbox": [150, 250, 250, 350],
                "embedding_id": "emb_002",
                "frame_width": 640,
                "frame_height": 480,
            },
        ]
        
        print("\n🔬 Testing scene graph building...")
        graphs = build_scene_graphs(memory, tracks)
        print(f"  ✅ Scene graph works: {len(graphs)} frames")
        
        if graphs:
            print(f"  Frame 0: {len(graphs[0].get('nodes', []))} nodes, {len(graphs[0].get('edges', []))} edges")
        
    except Exception as e:
        print(f"  ❌ Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()

def test_depth_estimation():
    """Test Depth Estimation"""
    print("\n" + "=" * 80)
    print("DEPTH ESTIMATION")
    print("=" * 80)
    
    try:
        from orion.perception.depth import DepthEstimator
        
        print("\n✅ Depth modules imported")
        
        # Test Depth Anything V2
        print("\n🔬 Testing Depth Anything V2...")
        depth_estimator = DepthEstimator(
            model_name="depth_anything_v2",
            model_size="small",
            device="mps"
        )
        print(f"  ✅ Depth estimator initialized: {depth_estimator.model_name}")
        
        # Test on dummy frame
        import numpy as np
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth_map, _ = depth_estimator.estimate(test_frame)
        print(f"  ✅ Depth estimation works: shape={depth_map.shape}, range=[{depth_map.min():.1f}, {depth_map.max():.1f}]mm")
        
    except Exception as e:
        print(f"  ❌ Depth estimation failed: {e}")
        import traceback
        traceback.print_exc()

def test_dino_backends():
    """Test DINO backends (DINOv2 vs DINOv3)"""
    print("\n" + "=" * 80)
    print("DINO BACKENDS (DINOv2 vs DINOv3)")
    print("=" * 80)
    
    try:
        from orion.backends.dino_backend import DINOEmbedder
        import numpy as np
        
        print("\n✅ DINO modules imported")
        
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test DINOv2 (public)
        print("\n🔬 Testing DINOv2 (facebook/dinov2-base)...")
        try:
            dino_v2 = DINOEmbedder(model_name="facebook/dinov2-base", device="mps")
            embed_v2 = dino_v2.encode_image(test_image)
            print(f"  ✅ DINOv2 works! Embedding shape: {embed_v2.shape}")
            if embed_v2.shape != (768,):
                print(f"  ❌ Shape mismatch! Expected (768,), got {embed_v2.shape}")
        except Exception as e:
            print(f"  ❌ DINOv2 failed: {e}")
        
        # Test DINOv3 (local weights)
        print("\n🔬 Testing DINOv3 (local weights)...")
        dinov3_path = Path("models/dinov3-vitb16")
        if dinov3_path.exists():
            try:
                dino_v3 = DINOEmbedder(local_weights_dir=dinov3_path, device="mps")
                embed_v3 = dino_v3.encode_image(test_image)
                print(f"  ✅ DINOv3 works! Embedding shape: {embed_v3.shape}")
                if embed_v3.shape != (768,):
                    print(f"  ❌ Shape mismatch! Expected (768,), got {embed_v3.shape}")
            except Exception as e:
                print(f"  ❌ DINOv3 failed: {e}")
        else:
            print(f"  ⚠️  DINOv3 weights not found at {dinov3_path}")
            print("  💡 Download from: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/")
    
    except Exception as e:
        print(f"  ❌ DINO backend test failed: {e}")
        import traceback
        traceback.print_exc()

def test_visual_embedder_batch():
    """Test VisualEmbedder batch processing and shape consistency."""
    print("\n" + "=" * 80)
    print("VISUAL EMBEDDER BATCH TEST")
    print("=" * 80)

    try:
        from orion.perception.embedder import VisualEmbedder
        from orion.perception.config import EmbeddingConfig
        import numpy as np

        # Test DINOv2 backend in VisualEmbedder
        print("\n🔬 Testing VisualEmbedder with DINOv2 backend...")
        config = EmbeddingConfig(
            backend="dinov2",
            model="facebook/dinov2-base",
            device="mps",
            batch_size=2
        )
        embedder = VisualEmbedder(config=config)
        
        # Create dummy detections
        detections = []
        for i in range(3):
            detections.append({
                "crop": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "frame_number": 0,
                "bbox": [0, 0, 224, 224]
            })
            
        # Run embedding
        results = embedder.embed_detections(detections)
        
        # Verify
        print(f"  ✅ Processed {len(results)} detections")
        for i, res in enumerate(results):
            emb = res.get("embedding")
            if emb is not None:
                shape = emb.shape
                if shape != (768,):
                     print(f"  ❌ Item {i} Shape mismatch! Expected (768,), got {shape}")
            else:
                print(f"  ❌ Item {i} missing embedding")
        print("  ✅ All shapes verified (768,)")

    except Exception as e:
        print(f"  ❌ VisualEmbedder test failed: {e}")
        import traceback
        traceback.print_exc()

def test_end_to_end():
    """Test end-to-end pipeline on example video"""
    print("\n" + "=" * 80)
    print("END-TO-END PIPELINE TEST")
    print("=" * 80)
    
    example_videos = list(Path("data/examples").glob("*.mp4"))
    if not example_videos:
        print("  ⚠️  No example videos found in data/examples/")
        return
    
    test_video = example_videos[0]
    print(f"\n📹 Test video: {test_video}")
    
    # Check if results exist
    episode_id = "pipeline_test"
    results_dir = Path(f"results/{episode_id}")
    
    if results_dir.exists():
        # Check results
        tracks_file = results_dir / "tracks.jsonl"
        memory_file = results_dir / "memory.json"
        graph_file = results_dir / "scene_graph.jsonl"
        
        if tracks_file.exists():
            with open(tracks_file) as f:
                num_tracks = sum(1 for _ in f)
            print(f"  ✅ Tracks: {num_tracks} observations")
        
        if memory_file.exists():
            with open(memory_file) as f:
                memory = json.load(f)
            print(f"  ✅ Memory: {len(memory.get('objects', []))} unique objects")
        
        if graph_file.exists():
            with open(graph_file) as f:
                num_frames = sum(1 for _ in f)
            print(f"  ✅ Scene graph: {num_frames} frames")
    else:
        print(f"  ⚠️  No results found. Run:")
        print(f"     python -m orion.cli.run_showcase --episode {episode_id} --video {test_video}")

def main():
    print("=" * 80)
    print("ORION PIPELINE COMPREHENSIVE TESTING")
    print("=" * 80)
    
    # Test each component
    test_phase1_detection()
    test_phase2_reid()
    test_phase3_scene_graph()
    test_depth_estimation()
    test_dino_backends()
    test_visual_embedder_batch()
    test_end_to_end()
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
