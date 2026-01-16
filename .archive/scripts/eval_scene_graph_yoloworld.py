#!/usr/bin/env python3
"""
Evaluate Scene Graph Generation (SGG) results from YOLO-World + Orion pipeline
against PVSG ground truth relations.
"""

import json
import os
from collections import defaultdict
import numpy as np

def main():
    # Load 10 videos with YOLO-World + scene graphs  
    video_ids = [
        "0001_4164158586", "0003_3396832512", "0003_6141007489", "0004_11566980553",
        "0005_2505076295", "0006_2889117240", "0008_6225185844", "0008_8890945814",
        "0010_8610561401", "0018_3057666738"
    ]

    # Load PVSG ground truth
    print("Loading PVSG ground truth...")
    with open('datasets/PVSG/pvsg.json', 'r') as f:
        pvsg = json.load(f)
    gt_videos = {v['video_id']: v for v in pvsg['data']}

    # Relation mapping (normalize relation names)
    RELATION_MAP = {
        'held_by': 'hold',
        'on': 'on',
        'near': 'next_to'
    }

    results = []

    print("\nProcessing videos...")
    for vid in video_ids:
        # Load Orion scene graph
        sg_file = f'results/{vid}/scene_graph.jsonl'
        if not os.path.exists(sg_file):
            print(f"Skip {vid}: no scene graph")
            continue
        
        # Count Orion relations
        orion_relations = []
        with open(sg_file, 'r') as f:
            for line in f:
                frame_sg = json.loads(line)
                for edge in frame_sg.get('edges', []):
                    rel = edge.get('relation', '')
                    if rel in RELATION_MAP:
                        orion_relations.append(RELATION_MAP[rel])
        
        # Load GT relations
        if vid not in gt_videos:
            print(f"Skip {vid}: not in GT")
            continue
        
        gt_video = gt_videos[vid]
        gt_relations = []
        for rel in gt_video.get('relations', []):
            subj_id, obj_id, predicate, frame_ranges = rel
            gt_relations.append(predicate)
        
        # Count by predicate
        orion_counts = defaultdict(int)
        for r in orion_relations:
            orion_counts[r] += 1
        
        gt_counts = defaultdict(int)
        for r in gt_relations:
            gt_counts[r] += 1
        
        result = {
            'video_id': vid,
            'orion_total': len(orion_relations),
            'gt_total': len(gt_relations),
            'orion_by_pred': dict(orion_counts),
            'gt_by_pred': dict(gt_counts)
        }
        results.append(result)
        
        print(f"  {vid}: Orion={len(orion_relations)} relations | GT={len(gt_relations)} relations")

    print("\n" + "="*80)
    print("SCENE GRAPH GENERATION RESULTS (YOLO-World + SGG)")
    print("="*80)
    print(f"{'Video ID':<20} | {'Orion Rels':<12} | {'GT Rels':<12} | Coverage")
    print("-"*80)

    for r in results:
        coverage = (r['orion_total'] / r['gt_total'] * 100) if r['gt_total'] > 0 else 0
        print(f"{r['video_id']:<20} | {r['orion_total']:>10} | {r['gt_total']:>10} | {coverage:>6.1f}%")

    avg_coverage = np.mean([
        (r['orion_total'] / r['gt_total'] * 100) if r['gt_total'] > 0 else 0
        for r in results
    ])
    print("-"*80)
    print(f"{'AVERAGE':<20} | {np.mean([r['orion_total'] for r in results]):>10.1f} | {np.mean([r['gt_total'] for r in results]):>10.1f} | {avg_coverage:>6.1f}%")
    print("="*80)

    # Save results
    with open('scene_graph_results_yoloworld.json', 'w') as f:
        json.dump({'results': results, 'avg_coverage': avg_coverage}, f, indent=2)

    print("\n✓ Saved to scene_graph_results_yoloworld.json")

if __name__ == '__main__':
    main()
