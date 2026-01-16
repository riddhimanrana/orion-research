#!/usr/bin/env python3
"""
Context-aware scene understanding: Infer missing objects from scene context.
If we see: {cake, person}, infer likely: {plate, fork, knife}
If we see: {bed}, infer likely: {pillow, blanket}
"""

import json
from pathlib import Path
from collections import defaultdict

# Scene context rules: if we see these objects, likely to also have these
SCENE_CONTEXT = {
    'kitchen': {
        'found': ['oven', 'stove', 'microwave', 'refrigerator', 'sink', 'counter'],
        'infer': ['plate', 'cup', 'knife', 'fork', 'spoon', 'pan', 'pot'],
    },
    'dining': {
        'found': ['table', 'chair'],
        'infer': ['plate', 'cup', 'napkin', 'fork', 'spoon', 'knife'],
    },
    'bedroom': {
        'found': ['bed'],
        'infer': ['pillow', 'blanket', 'nightstand', 'dresser'],
    },
    'cake scene': {
        'found': ['cake'],
        'infer': ['plate', 'fork', 'candle', 'person'],
    },
    'baby': {
        'found': ['baby', 'child'],
        'infer': ['adult', 'toy', 'bottle', 'chair'],
    }
}

def infer_context_objects(detected_classes: set) -> list:
    """Given detected objects, infer likely missing objects."""
    inferred = []
    
    # Detect scenes
    if 'cake' in detected_classes:
        # Likely eating scene
        inferred.extend(['adult', 'child', 'plate', 'candle'])
    
    if 'bed' in detected_classes:
        inferred.extend(['person', 'pillow', 'blanket'])
    
    if 'table' in detected_classes and 'chair' in detected_classes:
        # Dining scene
        inferred.extend(['person', 'plate', 'cup', 'fork', 'knife'])
    
    if any(k in detected_classes for k in ['stove', 'oven', 'microwave', 'refrigerator']):
        # Kitchen scene
        inferred.extend(['adult', 'plate', 'cup', 'pot', 'pan'])
    
    # Baby scenes
    if any(k in detected_classes for k in ['toy', 'bottle', 'crib']):
        inferred.extend(['baby', 'child', 'adult'])
    
    # Remove already detected ones and return unique
    return list(set(inferred) - detected_classes)

def boost_with_contextual_relations(predicted_triplets: list, detected_classes: set) -> list:
    """
    Add contextual relations for inferred objects.
    
    Example: If we infer "plate" from "cake", add (adult, holding, plate) etc.
    """
    boosted = list(predicted_triplets)
    inferred = infer_context_objects(detected_classes)
    
    # For each inferred object, add likely relations
    for obj in inferred:
        if 'adult' in detected_classes or 'person' in detected_classes:
            person_class = 'adult' if 'adult' in detected_classes else 'person'
            
            # Common holding relations
            if obj in ['plate', 'cup', 'knife', 'fork', 'spoon', 'napkin', 'bottle']:
                boosted.append((person_class, 'holding', obj))
            
            # Common looking_at relations (VLM-based, but we approximate)
            if obj in ['cake', 'food', 'tv', 'book']:
                boosted.append((person_class, 'looking_at', obj))
        
        # Object-on-table relations
        if 'table' in detected_classes:
            if obj in ['plate', 'cup', 'knife', 'fork', 'bowl', 'glass', 'candle']:
                boosted.append((obj, 'on', 'table'))
        
        # Person-sitting relations
        if obj == 'chair' and ('adult' in detected_classes or 'person' in detected_classes):
            person = 'adult' if 'adult' in detected_classes else 'person'
            boosted.append((person, 'sitting_on', 'chair'))
    
    return boosted

def boost_single_video(video_id: str) -> tuple:
    """Boost a single video with contextual relations."""
    results_dir = Path("results")
    
    # Load memory
    mem_file = results_dir / video_id / "memory.json"
    if not mem_file.exists():
        return None, None
    
    with open(mem_file) as f:
        mem = json.load(f)
        detected = set(o.get('class', 'unknown').lower() for o in mem.get('objects', []))
    
    # Load scene graph
    sg_file = results_dir / video_id / "scene_graph.jsonl"
    predicted_triplets = []
    
    if sg_file.exists():
        with open(sg_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    for edge in data.get('edges', []):
                        # Extract class names from memory_id
                        # This is complex, so for now just count what we have
                        predicted_triplets.append(None)  # placeholder
                except:
                    pass
    
    # Boost
    # boosted = boost_with_contextual_relations(predicted_triplets, detected)
    inferred = infer_context_objects(detected)
    
    return detected, inferred


def main():
    """Analyze all videos and their inferred objects."""
    results_dir = Path("results")
    video_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and (d / "memory.json").exists()])
    
    print("Analyzing scene context and inferred objects...\n")
    print(f"{'Video':<20} | {'Detected':<30} | {'Inferred':<30}")
    print("-" * 85)
    
    for video_dir in video_dirs[:5]:  # First 5 for testing
        vid = video_dir.name
        detected, inferred = boost_single_video(vid)
        
        if detected:
            det_str = ','.join(sorted(list(detected))[:4])
            inf_str = ','.join(sorted(inferred)[:4]) if inferred else "-"
            print(f"{vid:<20} | {det_str:<30} | {inf_str:<30}")
    
    print("\nThis approach could boost recall if inferred objects match GT.")
    print("However, it still generates hallucinated relations without actual detections.")


if __name__ == '__main__':
    main()
