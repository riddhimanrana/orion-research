#!/usr/bin/env python3
"""
Detailed SGG diagnostic: Object detection vs GT requirements per video
"""

import json
import os
from collections import Counter

vids = [
    "0020_10793023296",
    "0020_5323209509",
    "0021_2446450580",
    "0021_4999665957",
    "0024_5224805531",
    "0026_2764832695",
    "0027_4571353789",
    "0028_3085751774",
    "0028_4021064662",
    "0029_5139813648",
]

gt_file = 'datasets/PVSG/pvsg.json'
with open(gt_file) as f:
    gt_data = json.load(f)

gt_by_vid = {v['video_id']: v for v in gt_data['data']}

print("=" * 100)
print("SGG Diagnostic: Object Detection Failures vs GT Requirements")
print("=" * 100)

total_gt_triplets = 0
total_matchable = 0
total_missing_objs = 0

for vid in vids:
    print(f"\n{vid}")
    print("-" * 100)
    
    # Get Orion detections
    mem_file = f'results/{vid}/memory.json'
    orion_classes = set()
    if os.path.exists(mem_file):
        with open(mem_file) as f:
            memory = json.load(f)
            orion_classes = set([o.get('class', 'unknown') for o in memory.get('objects', [])])
    
    # Get GT requirements
    gt_objs = gt_by_vid[vid].get('objects', [])
    gt_id_to_class = {o['object_id']: o['category'] for o in gt_objs}
    gt_relations = gt_by_vid[vid].get('relations', [])
    gt_classes = set(gt_id_to_class.values())
    
    # Analyze
    matchable = 0
    unmatchable = 0
    blocked_reasons = Counter()
    
    for subj_id, obj_id, pred, frames in gt_relations:
        subj_class = gt_id_to_class.get(subj_id, '?')
        obj_class = gt_id_to_class.get(obj_id, '?')
        
        if subj_class in orion_classes and obj_class in orion_classes:
            matchable += 1
        else:
            unmatchable += 1
            if subj_class not in orion_classes:
                blocked_reasons[f"subject:'{subj_class}'"] += 1
                total_missing_objs += 1
            if obj_class not in orion_classes:
                blocked_reasons[f"object:'{obj_class}'"] += 1
                total_missing_objs += 1
    
    total_gt = len(gt_relations)
    total_gt_triplets += total_gt
    total_matchable += matchable
    
    # Report
    print(f"  Orion detected: {', '.join(sorted(orion_classes))}")
    print(f"  GT requires: {', '.join(sorted(gt_classes))}")
    print(f"  Missing from Orion: {', '.join(sorted(gt_classes - orion_classes))}")
    print(f"  Triplet analysis: {matchable}/{total_gt} matchable ({100*matchable/total_gt:.1f}%)")
    
    if blocked_reasons:
        print(f"  Top blocking reasons:")
        for reason, count in blocked_reasons.most_common(5):
            print(f"    - {reason}: {count} triplets")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"Total GT triplets: {total_gt_triplets}")
print(f"Matchable triplets: {total_matchable} ({100*total_matchable/total_gt_triplets:.1f}%)")
print(f"Blocked triplets: {total_gt_triplets - total_matchable} ({100*(1-total_matchable/total_gt_triplets):.1f}%)")
print(f"Total missing object instances: {total_missing_objs}")
print(f"\n✗ CONCLUSION: Object detection is the PRIMARY blocker for SGG performance")
