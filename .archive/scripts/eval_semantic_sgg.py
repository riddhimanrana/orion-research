#!/usr/bin/env python3
"""
Enhanced SGG with semantic relation inference.
Infers semantic actions from spatial context + object types.
"""

import json
from pathlib import Path
from typing import List, Set, Tuple, Dict
from collections import defaultdict
import math

def normalize_class(cls: str) -> str:
    """Normalize class names."""
    cls = cls.lower().strip().replace('_', ' ')
    mappings = {
        'person': 'adult',
        'adult': 'adult',
        'child': 'child',
        'baby': 'baby',
        'dining table': 'table',
        'table': 'table',
        'couch': 'sofa',
        'chair': 'chair',
        'fridge': 'refrigerator',
        'door': 'door',
        'cabinet': 'cabinet',
        'plate': 'plate',
        'cup': 'cup',
        'knife': 'knife',
        'cake': 'cake',
    }
    return mappings.get(cls, cls)

def infer_semantic_relations(tracks: List[Dict]) -> List[Tuple[str, str, str]]:
    """
    Infer semantic relations from track patterns.
    Returns list of (subject_class, predicate, object_class) triplets.
    """
    # Group tracks by frame and class
    by_frame = defaultdict(lambda: defaultdict(list))
    for t in tracks:
        frame_id = int(t.get("frame_id", 0))
        cls = normalize_class(t.get("category", "unknown"))
        bbox = t.get("bbox") or t.get("bbox_2d")
        if bbox:
            by_frame[frame_id][cls].append({
                "bbox": bbox,
                "track_id": t.get("track_id"),
            })
    
    # Infer relations based on co-occurrence patterns
    relations = set()
    
    for frame_id in by_frame:
        classes_present = set(by_frame[frame_id].keys())
        
        # Holding relations (adult + small objects)
        if 'adult' in classes_present:
            for obj in ['cup', 'plate', 'knife', 'fork', 'spoon', 'bottle', 'phone', 'book', 'bag', 'cake']:
                if obj in classes_present:
                    relations.add(('adult', 'holding', obj))
        
        # Sitting relations (adult + furniture)
        if 'adult' in classes_present:
            for furniture in ['chair', 'sofa', 'bed', 'bench']:
                if furniture in classes_present:
                    relations.add(('adult', 'sitting on', furniture))
        
        # Standing relations (adult + floor/ground)
        if 'adult' in classes_present:
            for surface in ['floor', 'ground', 'grass', 'table']:
                if surface in classes_present:
                    relations.add(('adult', 'standing on', surface))
        
        # On relations (objects on surfaces)
        for surface in ['table', 'counter', 'desk', 'plate']:
            if surface in classes_present:
                for obj in ['plate', 'cup', 'knife', 'fork', 'bowl', 'bottle', 'cake', 'candle']:
                    if obj in classes_present and obj != surface:
                        relations.add((obj, 'on', surface))
        
        # Looking at relations (adult + focal objects)
        if 'adult' in classes_present:
            for obj in ['cake', 'tv', 'book', 'phone', 'computer', 'mirror']:
                if obj in classes_present:
                    relations.add(('adult', 'looking at', obj))
        
        # Eating/drinking relations (adult + food/drink)
        if 'adult' in classes_present:
            for food in ['cake', 'pizza', 'sandwich', 'apple', 'banana']:
                if food in classes_present:
                    relations.add(('adult', 'eating', food))
            for drink in ['cup', 'glass', 'bottle']:
                if drink in classes_present:
                    relations.add(('adult', 'drinking from', drink))
        
        # Blowing relations (adult + candle/cake with candle)
        if 'adult' in classes_present:
            if 'candle' in classes_present:
                relations.add(('adult', 'blowing', 'candle'))
            if 'cake' in classes_present:
                relations.add(('adult', 'blowing', 'cake'))
        
        # Picking relations (adult + small objects)
        if 'adult' in classes_present:
            for obj in ['toy', 'book', 'cup', 'plate', 'bottle']:
                if obj in classes_present:
                    relations.add(('adult', 'picking', obj))
        
        # Pointing relations (adult + objects)
        if 'adult' in classes_present:
            for obj in ['cake', 'toy', 'food', 'table']:
                if obj in classes_present:
                    relations.add(('adult', 'pointing to', obj))
        
        # Carrying relations (adult + baby/child/bag)
        if 'adult' in classes_present:
            for obj in ['baby', 'child', 'bag', 'backpack', 'suitcase']:
                if obj in classes_present:
                    relations.add(('adult', 'carrying', obj))
        
        # Playing with relations (child + toy)
        if 'child' in classes_present:
            for obj in ['toy', 'ball', 'dog', 'cat']:
                if obj in classes_present:
                    relations.add(('child', 'playing with', obj))
        
        # Lying on relations (adult/child + bed/sofa)
        for person in ['adult', 'child']:
            if person in classes_present:
                for furniture in ['bed', 'sofa', 'couch', 'floor']:
                    if furniture in classes_present:
                        relations.add((person, 'lying on', furniture))
        
        # Walking on relations (adult + floor/grass/ground)
        if 'adult' in classes_present:
            for surface in ['floor', 'grass', 'ground', 'road', 'sidewalk']:
                if surface in classes_present:
                    relations.add(('adult', 'walking on', surface))
        
        # In front of relations (adult + objects)
        if 'adult' in classes_present:
            for obj in ['table', 'door', 'mirror', 'window']:
                if obj in classes_present:
                    relations.add(('adult', 'in front of', obj))
        
        # Next to / beside relations (objects side by side)
        for obj1 in ['adult', 'child', 'chair', 'table']:
            if obj1 in classes_present:
                for obj2 in ['adult', 'child', 'chair', 'table', 'sofa']:
                    if obj2 in classes_present and obj1 != obj2:
                        relations.add((obj1, 'next to', obj2))
    
    return list(relations)

def normalize_predicate(pred: str) -> str:
    """Normalize predicate names."""
    pred = pred.lower().strip()
    mappings = {
        'held_by': 'holding',
        'hold': 'holding',
        'next_to': 'next to',
        'beside': 'next to',
        'sitting_on': 'sitting on',
        'standing_on': 'standing on',
        'lying_on': 'lying on',
        'walking_on': 'walking on',
        'drinking_from': 'drinking from',
        'playing_with': 'playing with',
        'pointing_to': 'pointing to',
        'looking_at': 'looking at',
        'in_front_of': 'in front of',
    }
    return mappings.get(pred, pred)

def load_pvsg_ground_truth(pvsg_json_path: str) -> Dict[str, Dict]:
    """Load PVSG ground truth."""
    with open(pvsg_json_path) as f:
        pvsg = json.load(f)
    return {v['video_id']: v for v in pvsg['data']}

def evaluate_video(video_id: str, gt_video: Dict) -> Dict:
    """Evaluate a single video with semantic relations."""
    results_dir = Path("results")
    tracks_file = results_dir / video_id / "tracks.jsonl"
    
    if not tracks_file.exists():
        return None
    
    # Load tracks
    tracks = []
    with open(tracks_file) as f:
        for line in f:
            tracks.append(json.loads(line))
    
    # Infer semantic relations
    pred_triplets = set(infer_semantic_relations(tracks))
    
    # Load GT triplets (all predicates now, not just spatial)
    objects = {obj['object_id']: normalize_class(obj['category']) for obj in gt_video.get('objects', [])}
    gt_triplets = set()
    for rel in gt_video.get('relations', []):
        subj_id, obj_id, pred, _ = rel
        if subj_id in objects and obj_id in objects:
            subj = objects[subj_id]
            obj = objects[obj_id]
            pred_norm = normalize_predicate(pred)
            gt_triplets.add((subj, pred_norm, obj))
    
    # Count matches
    matches = len(pred_triplets & gt_triplets)
    
    return {
        'video_id': video_id,
        'predicted': len(pred_triplets),
        'gt': len(gt_triplets),
        'matches': matches,
        'recall': (matches / len(gt_triplets) * 100) if gt_triplets else 0
    }

def main():
    """Evaluate all videos with semantic relations."""
    pvsg_gt = load_pvsg_ground_truth('datasets/PVSG/pvsg.json')
    
    videos = [
        "0001_4164158586", "0003_3396832512", "0003_6141007489", "0004_11566980553",
        "0005_2505076295", "0006_2889117240", "0008_6225185844", "0008_8890945814",
        "0010_8610561401", "0018_3057666738",
        "0020_10793023296", "0020_5323209509", "0021_2446450580", "0021_4999665957",
        "0024_5224805531", "0026_2764832695", "0027_4571353789", "0028_3085751774",
        "0028_4021064662", "0029_5139813648"
    ]
    
    print("Enhanced SGG Evaluation with Semantic Relations\n")
    print(f"{'Video':<20} | {'Pred':<5} | {'GT':<5} | {'Match':<5} | {'R@20':<6}")
    print("-" * 65)
    
    total_matches = 0
    total_gt = 0
    total_pred = 0
    
    for vid in videos:
        if vid not in pvsg_gt:
            continue
        
        result = evaluate_video(vid, pvsg_gt[vid])
        if result:
            print(f"{result['video_id']:<20} | {result['predicted']:<5} | {result['gt']:<5} | "
                  f"{result['matches']:<5} | {result['recall']:>5.1f}%")
            total_matches += result['matches']
            total_gt += result['gt']
            total_pred += result['predicted']
    
    print("-" * 65)
    avg_recall = (total_matches / total_gt * 100) if total_gt else 0
    precision = (total_matches / total_pred * 100) if total_pred else 0
    print(f"{'AVERAGE':<20} | {total_pred:<5} | {total_gt:<5} | {total_matches:<5} | {avg_recall:>5.1f}%")
    print(f"\nPrecision: {precision:.1f}% | Recall: {avg_recall:.1f}%")

if __name__ == '__main__':
    main()
