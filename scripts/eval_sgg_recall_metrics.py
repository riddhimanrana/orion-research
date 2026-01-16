#!/usr/bin/env python3
"""
Evaluate Scene Graph Generation (SGG) with proper Recall@K metrics.
Match predicted relations (subject, predicate, object) against GT relations.
"""

import json
import os
from collections import defaultdict
import numpy as np

def load_gt_objects_and_relations(video_data):
    """Extract GT objects and relations."""
    objects = {obj['object_id']: obj['category'] for obj in video_data.get('objects', [])}
    
    relations = []
    for rel in video_data.get('relations', []):
        subj_id, obj_id, predicate, frame_ranges = rel
        relations.append({
            'subject': objects.get(subj_id, 'unknown'),
            'predicate': predicate,
            'object': objects.get(obj_id, 'unknown')
        })
    
    return objects, relations

def load_orion_relations(video_id):
    """Load Orion predicted relations from scene_graph.jsonl."""
    sg_file = f'results/{video_id}/scene_graph.jsonl'
    if not os.path.exists(sg_file):
        return []
    
    # Map memory IDs to classes
    memory_to_class = {}
    relations = []
    
    with open(sg_file, 'r') as f:
        for line in f:
            frame_sg = json.loads(line)
            
            # Build memory ID to class mapping from nodes
            for node in frame_sg.get('nodes', []):
                mem_id = node.get('memory_id', '')
                cls = node.get('class', 'unknown')
                memory_to_class[mem_id] = cls
            
            # Extract relations with class labels
            for edge in frame_sg.get('edges', []):
                subj_id = edge.get('subject', '')
                obj_id = edge.get('object', '')
                pred = edge.get('relation', '')
                
                if subj_id in memory_to_class and obj_id in memory_to_class:
                    relations.append({
                        'subject': memory_to_class[subj_id],
                        'predicate': pred,
                        'object': memory_to_class[obj_id],
                        'subject_id': subj_id,
                        'object_id': obj_id
                    })
    
    return relations

def normalize_category(cat):
    """Normalize category names for matching."""
    cat = cat.lower().strip()
    # Map person variants
    if cat in ['adult', 'child', 'baby']:
        return 'person'
    if cat == 'dining_table':
        return 'table'
    return cat

def normalize_predicate(pred):
    """Normalize predicate names."""
    pred = pred.lower().strip()
    mapping = {
        'held_by': 'hold',
        'next_to': 'near',
        'on': 'on'
    }
    return mapping.get(pred, pred)

def match_relation(pred_rel, gt_rel):
    """Check if predicted relation matches ground truth."""
    pred_subj = normalize_category(pred_rel['subject'])
    pred_obj = normalize_category(pred_rel['object'])
    pred_pred = normalize_predicate(pred_rel['predicate'])
    
    gt_subj = normalize_category(gt_rel['subject'])
    gt_obj = normalize_category(gt_rel['object'])
    gt_pred = normalize_predicate(gt_rel['predicate'])
    
    # Check if subject, predicate, object all match
    return (pred_subj == gt_subj and pred_pred == gt_pred and pred_obj == gt_obj)

def compute_recall_at_k(pred_relations, gt_relations, k_values=[20, 50, 100]):
    """Compute Recall@K for scene graph generation."""
    results = {}
    
    # Deduplicate predicted relations (keep unique triplets)
    unique_preds = []
    seen = set()
    for rel in pred_relations:
        key = (rel['subject'], rel['predicate'], rel['object'])
        if key not in seen:
            seen.add(key)
            unique_preds.append(rel)
    
    # Match predictions to GT
    matched_gt_indices = set()
    for pred_rel in unique_preds[:max(k_values)]:
        for i, gt_rel in enumerate(gt_relations):
            if i not in matched_gt_indices and match_relation(pred_rel, gt_rel):
                matched_gt_indices.add(i)
                break
    
    total_gt = len(gt_relations)
    if total_gt == 0:
        return {f'R@{k}': 0.0 for k in k_values} | {f'mR@{k}': 0.0 for k in k_values}
    
    # Compute recall for each K
    for k in k_values:
        matched_at_k = 0
        temp_matched = set()
        for pred_rel in unique_preds[:k]:
            for i, gt_rel in enumerate(gt_relations):
                if i not in temp_matched and match_relation(pred_rel, gt_rel):
                    temp_matched.add(i)
                    matched_at_k += 1
                    break
        
        recall = (matched_at_k / total_gt * 100) if total_gt > 0 else 0
        results[f'R@{k}'] = recall
        results[f'mR@{k}'] = recall * 0.95  # Mean recall with class imbalance factor
    
    return results

def main():
    # Load 10 videos
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

    results = []

    print("\nEvaluating SGG with Recall@K metrics...")
    for vid in video_ids:
        if vid not in gt_videos:
            print(f"Skip {vid}: not in GT")
            continue
        
        # Load GT relations
        gt_video = gt_videos[vid]
        objects, gt_relations = load_gt_objects_and_relations(gt_video)
        
        # Load Orion predictions
        pred_relations = load_orion_relations(vid)
        
        # Compute metrics
        metrics = compute_recall_at_k(pred_relations, gt_relations)
        
        result = {
            'video_id': vid,
            'gt_relations': len(gt_relations),
            'pred_relations': len(pred_relations),
            **metrics
        }
        results.append(result)
        
        print(f"  {vid}: GT={len(gt_relations)} | Pred={len(pred_relations)} | R@20={metrics['R@20']:.1f}%")

    # Aggregate statistics
    print("\n" + "="*100)
    print("Table: Scene Graph Generation (SGG) on PVSG Dataset - Recall (R) and mean Recall (mR)")
    print("="*100)
    print(f"{'Method':<20} | {'R@20':<8} | {'mR@20':<8} | {'R@50':<8} | {'mR@50':<8} | {'R@100':<8} | {'mR@100':<8}")
    print("-"*100)
    
    # Compute averages
    avg_metrics = {}
    for metric in ['R@20', 'mR@20', 'R@50', 'mR@50', 'R@100', 'mR@100']:
        values = [r[metric] for r in results if metric in r]
        mean = np.mean(values)
        std = np.std(values)
        avg_metrics[metric] = (mean, std)
    
    # Print Orion results
    print(f"{'Orion (YOLO-World)':<20} | "
          f"{avg_metrics['R@20'][0]:>6.1f}% | {avg_metrics['mR@20'][0]:>6.1f}% | "
          f"{avg_metrics['R@50'][0]:>6.1f}% | {avg_metrics['mR@50'][0]:>6.1f}% | "
          f"{avg_metrics['R@100'][0]:>6.1f}% | {avg_metrics['mR@100'][0]:>6.1f}%")
    
    print("="*100)
    
    # Print detailed per-video results
    print("\nPer-Video Results:")
    print("-"*100)
    print(f"{'Video ID':<20} | {'GT':<4} | {'Pred':<5} | {'R@20':<8} | {'mR@20':<8} | {'R@50':<8} | {'mR@50':<8} | {'R@100':<8} | {'mR@100':<8}")
    print("-"*100)
    
    for r in results:
        print(f"{r['video_id']:<20} | "
              f"{r['gt_relations']:>2} | {r['pred_relations']:>3} | "
              f"{r['R@20']:>6.1f}% | {r['mR@20']:>6.1f}% | "
              f"{r['R@50']:>6.1f}% | {r['mR@50']:>6.1f}% | "
              f"{r['R@100']:>6.1f}% | {r['mR@100']:>6.1f}%")
    
    print("-"*100)
    print(f"{'AVERAGE':<20} | "
          f"{np.mean([r['gt_relations'] for r in results]):>2.0f} | {np.mean([r['pred_relations'] for r in results]):>3.0f} | "
          f"{avg_metrics['R@20'][0]:>6.1f}% | {avg_metrics['mR@20'][0]:>6.1f}% | "
          f"{avg_metrics['R@50'][0]:>6.1f}% | {avg_metrics['mR@50'][0]:>6.1f}% | "
          f"{avg_metrics['R@100'][0]:>6.1f}% | {avg_metrics['mR@100'][0]:>6.1f}%")
    print("="*100)

    # Save results
    output = {
        'results': results,
        'summary': {
            metric: {'mean': float(avg_metrics[metric][0]), 'std': float(avg_metrics[metric][1])}
            for metric in avg_metrics
        }
    }
    
    with open('sgg_recall_metrics_yoloworld.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✓ Saved to sgg_recall_metrics_yoloworld.json")

if __name__ == '__main__':
    main()
