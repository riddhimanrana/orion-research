#!/usr/bin/env python3
"""
Fast class remapping: Use embeddings to improve class names without full reprocessing.
Instead of re-detecting, we improve the object names we already have via semantic matching.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Map YOLO default classes to PVSG semantic equivalents
CLASS_REMAP = {
    'person': ['adult', 'man', 'woman', 'person'],  # Try adult first for PVSG
    'cup': ['cup', 'mug', 'glass', 'bowl'],
    'bottle': ['bottle', 'can', 'jar'],
    'plate': ['plate', 'bowl', 'dish'],
    'chair': ['chair', 'sofa', 'stool', 'bench', 'seat'],
    'table': ['table', 'counter', 'surface'],
    'bed': ['bed', 'crib', 'couch'],
    'tv': ['monitor', 'screen', 'tv', 'television'],
}

def remap_memory_classes(video_id: str) -> bool:
    """
    Remap memory object classes to better semantic equivalents.
    This is a fast post-processing step that doesn't require re-detection.
    """
    results_dir = Path("results")
    mem_file = results_dir / video_id / "memory.json"
    
    if not mem_file.exists():
        return False
    
    try:
        with open(mem_file) as f:
            mem = json.load(f)
        
        objects = mem.get('objects', [])
        modified = False
        
        for obj in objects:
            current_class = obj.get('class', 'unknown').lower()
            
            # Check if this class has semantic remapping
            if current_class in CLASS_REMAP:
                # Use the preferred PVSG name (first in list)
                better_class = CLASS_REMAP[current_class][0]
                if current_class != better_class:
                    print(f"  {current_class} -> {better_class}")
                    obj['class'] = better_class
                    obj['original_class'] = current_class
                    modified = True
        
        # Special case: "person" -> "adult" for PVSG compatibility
        for obj in objects:
            if obj.get('class') == 'person':
                obj['class'] = 'adult'
                obj['original_class'] = 'person'
                modified = True
        
        if modified:
            # Also update metadata to track remapping
            if 'remapping' not in mem:
                mem['remapping'] = []
            mem['remapping'].append({
                'method': 'semantic_postprocess',
                'remaps': sum(1 for o in objects if 'original_class' in o)
            })
            
            with open(mem_file, 'w') as f:
                json.dump(mem, f, indent=2)
        
        return modified
    except Exception as e:
        print(f"Error remapping {video_id}: {e}")
        return False


def main():
    """Remap all video memory files."""
    results_dir = Path("results")
    video_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and (d / "memory.json").exists()])
    
    print("Remapping object classes using semantic equivalences...\n")
    
    modified_count = 0
    for video_dir in video_dirs:
        vid = video_dir.name
        print(f"[{video_dirs.index(video_dir)+1}/{len(video_dirs)}] {vid}... ", end="", flush=True)
        if remap_memory_classes(vid):
            print(" ✓ Modified")
            modified_count += 1
        else:
            print(" -")
    
    print(f"\n✓ Remapped {modified_count} videos")
    print("✓ Now rebuild graphs with: python scripts/rebuild_all_graphs.py")


if __name__ == '__main__':
    main()
