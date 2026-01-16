#!/usr/bin/env python3
"""
Cleanup dead code and test core pipeline components.
Part a) Remove unused/legacy files
Part b) Test each stage with example videos
"""

import os
import sys
from pathlib import Path
import json
import shutil

# Dead/unused code to remove (not imported anywhere in core pipeline)
DEAD_CODE = {
    "orion": [
        "depth_anything.py",  # Legacy, replaced by orion/perception/depth.py
        "corrector.py",  # Not used
        "reid_matcher.py",  # Moved to orion/perception/reid/matcher.py
        "spatial_map_builder.py",  # Experimental, not in core pipeline
        "settings.py",  # Old config, replaced by orion/perception/config.py
    ],
    "scripts": [
        # SGG exploration scripts (keep eval_sgg_recall.py, remove experiments)
        "aggressive_sgg.py",
        "analyze_scene_graph.py",
        "context_aware_boost.py",
        "eval_fuzzy_matching.py",
        "eval_sgg_filtered.py",
        "eval_track_based_sgg.py",
        "fast_class_remap.py",
        "rebuild_aggressive.py",
        "rebuild_all_graphs.py",
        "reprocess_all_yoloworld.py",
        "reprocess_with_vocab.sh",
        "speculative_sgg.py",
        "vlm_relations.py",
        "batch_recluster_memory.sh",
        "batch_recluster_memory.py",
        # Other experimental/one-off scripts
        "print_reid_metrics.py",
        "gemini_accuracy_evaluation.py",
        "validate_phase2_gemini.py",
    ],
    "docs": [
        # Keep only the core docs
        # Remove: interim status reports, optimization plans
    ],
    "root": [
        # Top-level clutter
        "yolo11m.pt",  # Model weight should be in models/
        "qa_comparison_results.json",  # One-off result file
        "sgg_recall_results.json",
        "sgg_recall_filtered_results.json",
        "SGG_EVALUATION_REPORT.md",
        "SGG_OPTIMIZATION_PLAN.md",
        "SGG_INTERIM_STATUS.md",
    ]
}

# Core pipeline files to KEEP and test
CORE_PIPELINE = {
    "detection": "orion/perception/detectors/",
    "tracking": "orion/perception/trackers/",
    "reid": "orion/perception/reid/",
    "scene_graph": "orion/graph/scene_graph.py",
    "depth": "orion/perception/depth.py",
    "backends": "orion/backends/",
    "cli": "orion/cli/run_showcase.py",
}

def backup_and_remove(file_path: Path, dry_run: bool = True):
    """Backup file to _archive/ and remove."""
    if not file_path.exists():
        print(f"  ⚠️  Not found: {file_path}")
        return
    
    # Create archive directory
    archive_dir = file_path.parent / "_archive"
    archive_dir.mkdir(exist_ok=True)
    
    if dry_run:
        print(f"  🔍 Would archive: {file_path.name}")
    else:
        backup_path = archive_dir / file_path.name
        shutil.move(str(file_path), str(backup_path))
        print(f"  ✅ Archived: {file_path.name} -> {backup_path}")

def cleanup_dead_code(dry_run: bool = True):
    """Remove dead/unused code files."""
    repo_root = Path(__file__).resolve().parents[1]
    
    print("=" * 80)
    print("PART A: CLEANUP DEAD CODE")
    print("=" * 80)
    print(f"Repo: {repo_root}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will delete)'}\n")
    
    # Cleanup orion/ root files
    print("📁 orion/ (root-level legacy files):")
    for filename in DEAD_CODE["orion"]:
        file_path = repo_root / "orion" / filename
        backup_and_remove(file_path, dry_run)
    
    # Cleanup scripts/
    print("\n📁 scripts/ (experimental/one-off scripts):")
    for filename in DEAD_CODE["scripts"]:
        file_path = repo_root / "scripts" / filename
        backup_and_remove(file_path, dry_run)
    
    # Cleanup root
    print("\n📁 root/ (top-level clutter):")
    for filename in DEAD_CODE["root"]:
        file_path = repo_root / filename
        backup_and_remove(file_path, dry_run)
    
    print("\n✅ Cleanup complete!" if not dry_run else "\n🔍 Dry run complete!")

def test_dino_backends():
    """Test DINOv2 vs DINOv3 embeddings."""
    print("\n" + "=" * 80)
    print("PART B: TEST DINO BACKENDS")
    print("=" * 80)
    
    try:
        from orion.backends.dino_backend import DINOEmbedder
        import numpy as np
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test DINOv2 (public)
        print("\n🧪 Testing DINOv2 (facebook/dinov2-base)...")
        try:
            dino_v2 = DINOEmbedder(model_name="facebook/dinov2-base", device="mps")
            embed_v2 = dino_v2.encode_image(test_image)
            print(f"  ✅ DINOv2 works! Embedding shape: {embed_v2.shape}")
        except Exception as e:
            print(f"  ❌ DINOv2 failed: {e}")
        
        # Test DINOv3 (local weights if available)
        print("\n🧪 Testing DINOv3 (local weights)...")
        dinov3_path = Path("models/dinov3-vitb16")
        if dinov3_path.exists():
            try:
                dino_v3 = DINOEmbedder(local_weights_dir=dinov3_path, device="mps")
                embed_v3 = dino_v3.encode_image(test_image)
                print(f"  ✅ DINOv3 works! Embedding shape: {embed_v3.shape}")
            except Exception as e:
                print(f"  ❌ DINOv3 failed: {e}")
        else:
            print(f"  ⚠️  DINOv3 weights not found at {dinov3_path}")
            print("  💡 Download from: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/")
    
    except ImportError as e:
        print(f"  ❌ Cannot import DINO backend: {e}")

def test_pipeline_stages():
    """Test each pipeline stage with example video."""
    print("\n" + "=" * 80)
    print("PART B: TEST PIPELINE STAGES")
    print("=" * 80)
    
    # Check if example video exists
    example_video = Path("data/examples/test.mp4")
    if not example_video.exists():
        print(f"  ⚠️  Example video not found: {example_video}")
        print("  💡 Place a short test video at data/examples/test.mp4")
        return
    
    print(f"\n📹 Test video: {example_video}")
    
    # Test Phase 1: Detection + Tracking
    print("\n🔬 Phase 1: Detection + Tracking")
    try:
        from orion.cli.run_tracks import process_video_to_tracks
        
        results_dir = Path("results/test_cleanup")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print("  Running detection...")
        # This will test: YOLO detection, tracking, embeddings
        # process_video_to_tracks(example_video, results_dir, ...)
        print("  ⚠️  Skipped (requires full config setup)")
    
    except Exception as e:
        print(f"  ❌ Phase 1 failed: {e}")
    
    # Test Phase 2: Memory/Re-ID
    print("\n🔬 Phase 2: Re-ID / Memory Clustering")
    try:
        from orion.perception.reid.matcher import build_memory_from_tracks
        print("  ✅ Re-ID module importable")
    except Exception as e:
        print(f"  ❌ Re-ID import failed: {e}")
    
    # Test Phase 3: Scene Graph
    print("\n🔬 Phase 3: Scene Graph Generation")
    try:
        from orion.graph.scene_graph import build_scene_graphs
        print("  ✅ Scene graph module importable")
    except Exception as e:
        print(f"  ❌ Scene graph import failed: {e}")
    
    # Test Depth
    print("\n🔬 Depth Estimation")
    try:
        from orion.perception.depth import DepthEstimator
        print("  Testing DepthEstimator initialization...")
        depth_estimator = DepthEstimator(model_name="depth_anything_v2", model_size="small")
        print(f"  ✅ Depth estimator works! Using {depth_estimator.model_name}")
    except Exception as e:
        print(f"  ❌ Depth estimator failed: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cleanup and test Orion pipeline")
    parser.add_argument("--cleanup", action="store_true", help="Run cleanup (dry run by default)")
    parser.add_argument("--cleanup-live", action="store_true", help="Run cleanup (ACTUALLY DELETE)")
    parser.add_argument("--test", action="store_true", help="Run pipeline tests")
    parser.add_argument("--test-dino", action="store_true", help="Test DINO backends")
    args = parser.parse_args()
    
    if args.cleanup or args.cleanup_live:
        cleanup_dead_code(dry_run=not args.cleanup_live)
    
    if args.test_dino:
        test_dino_backends()
    
    if args.test:
        test_pipeline_stages()
    
    if not (args.cleanup or args.cleanup_live or args.test or args.test_dino):
        print("Usage:")
        print("  python scripts/cleanup_and_test.py --cleanup        # Dry run (preview)")
        print("  python scripts/cleanup_and_test.py --cleanup-live   # Actually delete")
        print("  python scripts/cleanup_and_test.py --test           # Test pipeline")
        print("  python scripts/cleanup_and_test.py --test-dino      # Test DINO backends")

if __name__ == "__main__":
    main()
