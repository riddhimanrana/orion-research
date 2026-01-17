#!/usr/bin/env python3
"""
Safe workspace prune - EXCLUDES orion/evaluation/ folder.

Removes:
  - Legacy orion/ root files (old trackers, pipelines, etc.)
  - Empty directories (semantic/, video_qa/, slam/)
  - Duplicate overlay versions
  - Old results (>7 days)
  - Build artifacts

PRESERVES:
  - orion/evaluation/ (UNTOUCHED)
  - All active modules
"""
import shutil
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Legacy files in orion/ root - NEW: explicitly exclude evaluation folder
LEGACY_FILES = [
    "orion/appearance_extractor.py",
    "orion/appearance_reid.py",
    "orion/clip_reid.py",
    "orion/fastvlm_reid.py",
    "orion/geometric_reid.py",
    "orion/reid_matcher.py",
    "orion/tracker_reid.py",
    "orion/advanced_detection.py",
    "orion/advanced_reid.py",
    "orion/object_tracker.py",
    "orion/entity_tracker.py",
    "orion/temporal_tracker.py",
    "orion/tracking.py",
    "orion/enhanced_tracker_adapter.py",
    "orion/pipeline.py",
    "orion/pipeline_adapter.py",
    "orion/unified_pipeline.py",
    "orion/scene_graph.py",
    "orion/scene_assembler.py",
    "orion/scene_classifier.py",
    "orion/scene_understanding.py",
    "orion/spatial_analyzer.py",
    "orion/spatial_map_builder.py",
    "orion/spatial_nlg.py",
    "orion/world_coordinate_tracker.py",
    "orion/projection_3d.py",
    "orion/reconstruction_3d.py",
    "orion/semantic_scale.py",
    "orion/semantic_slam.py",
    "orion/slam_fusion.py",
    "orion/enhanced_spatial_reasoning.py",
    "orion/depth_anything.py",
    "orion/depth_odometry.py",
    "orion/rich_captioning.py",
    "orion/smart_caption_prioritizer.py",
    "orion/strategic_captioner.py",
    "orion/corrector.py",
    "orion/query_intelligence.py",
    "orion/visualization.py",
]

LEGACY_DIRS = [
    "orion/_archive",
    "orion/semantic",
    "orion/video_qa",
    "orion/slam",
    "Depth-Anything-V2-temp",
    "Depth-Anything-3",
]

OVERLAY_VERSIONS = [
    "orion/perception/viz_overlay_v2.py",
    "orion/perception/viz_overlay_v3.py",
    "orion/perception/viz_overlay_v4.py",
]

BUILD_ARTIFACTS = [
    "orion.egg-info",
    "mlx-vlm/build",
]

def find_old_results(days: int):
    results_dir = REPO_ROOT / "results"
    if not results_dir.exists():
        return []
    cutoff = datetime.now() - timedelta(days=days)
    out = []
    for d in results_dir.iterdir():
        if d.is_dir():
            try:
                mtime = datetime.fromtimestamp(d.stat().st_mtime)
                if mtime < cutoff:
                    out.append(d)
            except:
                continue
    return out

def remove_items(items, dry_run=False):
    count = 0
    for p in items:
        p = REPO_ROOT / p if isinstance(p, str) else p
        if not p.exists():
            continue
        
        # SAFETY: Never touch evaluation folder
        if "evaluation" in str(p):
            print(f"⚠️  SKIPPED (protected): {p}")
            continue
        
        try:
            if dry_run:
                print(f"DRY-RUN: Would remove {p}")
            else:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                print(f"✓ Removed: {p}")
                count += 1
        except Exception as e:
            print(f"✗ Error removing {p}: {e}")
    return count

def main():
    import sys
    dry_run = "--confirm" not in sys.argv
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         SAFE PRUNE (Evaluation Folder Protected)             ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"\nMode: {'DRY-RUN' if dry_run else '🔴 APPLYING'}\n")
    
    all_items = (
        LEGACY_FILES +
        LEGACY_DIRS +
        OVERLAY_VERSIONS +
        BUILD_ARTIFACTS +
        find_old_results(7)
    )
    
    print(f"Items to process: {len(all_items)}")
    print("─" * 60)
    
    removed = remove_items(all_items, dry_run)
    
    print("\n" + "─" * 60)
    if dry_run:
        print(f"DRY-RUN: {removed} items would be removed")
        print("\nRun with --confirm to apply changes")
    else:
        print(f"✓ PRUNE COMPLETE: {removed} items removed")
        print("✓ orion/evaluation/ preserved")

if __name__ == "__main__":
    main()
