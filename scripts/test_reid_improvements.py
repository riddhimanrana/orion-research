#!/usr/bin/env python3
"""
Test Phase 2 Re-ID improvements with label normalization.

Runs the embed command with and without label normalization to compare results.

Usage:
    python scripts/test_reid_improvements.py --episode pipeline_test_fixed
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


def run_embed(episode: str, normalize: bool = True) -> dict:
    """Run embed command and return results."""
    cmd = [
        "python", "-m", "orion.cli.commands.embed",
        "--episode", episode,
        "--similarity", "0.75"
    ]
    
    # Note: normalization is now always on, but we can compare before/after
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Embed failed: {result.stderr}")
        return {}
    
    # Load results
    results_dir = Path(f"results/{episode}")
    memory_file = results_dir / "memory.json"
    
    if memory_file.exists():
        return json.load(open(memory_file))
    return {}


def analyze_results(episode: str):
    """Analyze embedding results."""
    results_dir = Path(f"results/{episode}")
    memory_file = results_dir / "memory.json"
    tracks_file = results_dir / "tracks.jsonl"
    
    if not memory_file.exists():
        print(f"No memory.json found in {results_dir}")
        return
    
    memory = json.load(open(memory_file))
    
    print("\n" + "="*60)
    print("PHASE 2 RE-ID RESULTS")
    print("="*60)
    
    meta = memory.get('metadata', {})
    print(f"\nMetadata:")
    print(f"  Tracks before: {meta.get('total_tracks_before', 'N/A')}")
    print(f"  Tracks embedded: {meta.get('tracks_embedded', 'N/A')}")
    print(f"  Objects after: {meta.get('total_objects_after', 'N/A')}")
    print(f"  Fragmentation reduction: {100*meta.get('fragmentation_reduction', 0):.1f}%")
    print(f"  Label normalization: {meta.get('label_normalization', 'N/A')}")
    
    objects = memory.get('objects', [])
    
    # Analyze labels
    label_counts = Counter(obj['label'] for obj in objects)
    
    print(f"\nUnique canonical labels: {len(label_counts)}")
    print(f"\nTop 15 labels:")
    for label, count in label_counts.most_common(15):
        print(f"  {label:<25} {count}")
    
    # Check for label merging
    merged_count = 0
    for obj in objects:
        raw_labels = obj.get('raw_labels', [])
        if len(raw_labels) > 1:
            merged_count += 1
    
    print(f"\nObjects with merged labels: {merged_count}")
    
    # Show examples of merged labels
    print("\nExamples of merged labels:")
    shown = 0
    for obj in objects:
        raw_labels = obj.get('raw_labels', [])
        if len(raw_labels) > 1 and shown < 10:
            print(f"  {obj['label']}: {raw_labels} -> {obj.get('original_track_ids', [])[:5]}...")
            shown += 1
    
    # Analyze cluster sizes
    cluster_sizes = [obj['total_observations'] for obj in objects]
    
    print(f"\nCluster size distribution:")
    print(f"  Min: {min(cluster_sizes)}")
    print(f"  Max: {max(cluster_sizes)}")
    print(f"  Avg: {sum(cluster_sizes)/len(cluster_sizes):.1f}")
    
    # Check singletons
    singletons = sum(1 for s in cluster_sizes if s <= 2)
    print(f"  Singletons (≤2 obs): {singletons} ({100*singletons/len(objects):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Test Phase 2 Re-ID improvements")
    parser.add_argument("--episode", default="pipeline_test_fixed",
                        help="Episode to test")
    parser.add_argument("--run-embed", action="store_true",
                        help="Run embed command (otherwise just analyze existing)")
    args = parser.parse_args()
    
    if args.run_embed:
        print("Running embed with label normalization...")
        run_embed(args.episode, normalize=True)
    
    analyze_results(args.episode)


if __name__ == "__main__":
    main()
