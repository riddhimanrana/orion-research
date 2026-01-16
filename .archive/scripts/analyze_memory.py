#!/usr/bin/env python3
"""Analyze Phase 2 memory.json results."""

import json
import sys
from collections import Counter
from pathlib import Path


def analyze_memory(memory_path: str):
    """Analyze memory.json from Phase 2."""
    with open(memory_path) as f:
        mem = json.load(f)
    
    print(f"=== Phase 2 Memory Summary ===")
    print(f"Total objects: {len(mem['objects'])}")
    
    # Group by label
    labels = Counter([o['label'] for o in mem['objects']])
    print(f"Unique labels: {len(labels)}")
    print()
    print("Top 15 labels by count:")
    for label, count in labels.most_common(15):
        print(f"  {label}: {count}")
    print()
    
    # Separate dynamic and static
    dynamic_objs = [o for o in mem['objects'] if 'tier' not in o]
    static_objs = [o for o in mem['objects'] if o.get('tier') == 'static']
    print(f"Dynamic objects (Re-ID): {len(dynamic_objs)}")
    print(f"Static objects (no Re-ID): {len(static_objs)}")
    
    # Count merged objects
    merged = [o for o in dynamic_objs if len(o.get('original_track_ids', [])) > 1]
    print(f"Objects with merged tracks: {len(merged)}")
    
    if merged:
        print()
        print("Sample merged objects (track consolidation):")
        for o in sorted(merged, key=lambda x: -len(x['original_track_ids']))[:10]:
            lifespan = o['last_seen_frame'] - o['first_seen_frame']
            print(f"  {o['label']} (id={o['object_id']}): {len(o['original_track_ids'])} tracks merged, "
                  f"frames {o['first_seen_frame']}-{o['last_seen_frame']} ({lifespan} span), "
                  f"conf={o['avg_confidence']:.2f}")
    
    # Compute fragmentation improvement
    total_tracks = sum(len(o.get('original_track_ids', [1])) for o in mem['objects'])
    print()
    print(f"Total original tracks represented: {total_tracks}")
    print(f"Unified objects: {len(mem['objects'])}")
    print(f"Consolidation ratio: {total_tracks / len(mem['objects']):.2f}:1")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_memory.py <memory.json>")
        sys.exit(1)
    
    analyze_memory(sys.argv[1])
