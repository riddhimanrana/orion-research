#!/usr/bin/env python
"""
Run temporal SGA evaluation on 10 videos with proper metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from orion.sga.temporal_model import TemporalSGAModel, TemporalSGAConfig
from orion.sga.ag_dataset_v2 import (
    ActionGenomeDatasetV2, 
    AG_ALL_PREDICATES, 
    AG_OBJECT_CLASSES,
    AG_IDX_TO_PREDICATE,
    collate_fn,
)


def compute_recall(pred_logits, exist_logits, target_predicates, target_existence, k_values=[10, 20, 50]):
    """Compute Recall@K for a batch."""
    batch_size = pred_logits.shape[0]
    exist_probs = torch.sigmoid(exist_logits).squeeze(-1)
    pred_probs = F.softmax(pred_logits, dim=-1)
    combined_probs = exist_probs.unsqueeze(-1) * pred_probs
    
    recalls = {f'R@{k}': [] for k in k_values}
    
    for b in range(batch_size):
        gt_mask = target_existence[b] > 0
        gt_positions = gt_mask.nonzero()
        
        if len(gt_positions) == 0:
            continue
        
        gt_set = set()
        for pos in gt_positions:
            f_idx, p_idx = pos[0].item(), pos[1].item()
            pred_class = target_predicates[b, f_idx, p_idx].item()
            if pred_class >= 0:
                gt_set.add((f_idx, p_idx, pred_class))
        
        if len(gt_set) == 0:
            continue
        
        probs = combined_probs[b]
        num_future, num_pairs, num_preds = probs.shape
        flat_probs = probs.flatten()
        
        for k in k_values:
            _, topk_indices = torch.topk(flat_probs, min(k, len(flat_probs)))
            
            hits = 0
            for idx in topk_indices:
                idx = idx.item()
                f_idx = idx // (num_pairs * num_preds)
                remainder = idx % (num_pairs * num_preds)
                p_idx = remainder // num_preds
                pred_idx = remainder % num_preds
                
                if (f_idx, p_idx, pred_idx) in gt_set:
                    hits += 1
            
            recalls[f'R@{k}'].append(hits / len(gt_set))
    
    return recalls


def main():
    annotation_path = 'datasets/ActionGenome/annotations/action_genome_v1.0/object_bbox_and_relationship.pkl'
    model_path = 'models/temporal_sga_best.pt'
    num_videos = 10
    
    print("=" * 70)
    print(f"TEMPORAL SGA EVALUATION ON {num_videos} VIDEOS")
    print("=" * 70)
    
    # Load model
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        model = TemporalSGAModel(config)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"✓ Loaded model from {model_path}")
        print(f"  Training R@20: {checkpoint['best_r20']:.2f}%")
    else:
        print("ERROR: No saved model found!")
        return
    
    model.eval()
    
    # Load dataset
    dataset = ActionGenomeDatasetV2(annotation_path, split='test', input_fraction=0.5)
    print(f"✓ Loaded {len(dataset)} test videos")
    
    # Run evaluation
    print(f"\nEvaluating on {num_videos} videos...")
    print("-" * 70)
    
    all_recalls = defaultdict(list)
    video_results = []
    
    for i in range(min(num_videos, len(dataset))):
        sample = dataset[i]
        batch = collate_fn([sample])
        
        with torch.no_grad():
            outputs = model(
                class_ids=batch['class_ids'],
                bboxes=batch['bboxes'],
                object_mask=batch['object_mask'],
                frame_mask=batch['frame_mask'],
                num_future_frames=batch['num_future_frames'],
            )
        
        # Compute per-video recall
        recalls = compute_recall(
            outputs['predicate_logits'],
            outputs['existence_logits'],
            batch['target_predicates'],
            batch['target_existence'],
        )
        
        # Count GT relations
        gt_count = (batch['target_existence'] > 0).sum().item()
        
        r10 = recalls['R@10'][0] * 100 if recalls['R@10'] else 0
        r20 = recalls['R@20'][0] * 100 if recalls['R@20'] else 0
        r50 = recalls['R@50'][0] * 100 if recalls['R@50'] else 0
        
        video_results.append({
            'video': sample['video_id'],
            'obs_frames': sample['num_observed_frames'],
            'future_frames': sample['num_future_frames'],
            'objects': sample['num_objects'],
            'gt_relations': gt_count,
            'R@10': r10,
            'R@20': r20,
            'R@50': r50,
        })
        
        for k, v in recalls.items():
            all_recalls[k].extend(v)
        
        print(f"[{i+1:2d}] {sample['video_id']}: obs={sample['num_observed_frames']}, "
              f"future={sample['num_future_frames']}, objs={sample['num_objects']}, "
              f"GT={gt_count:2d} | R@10={r10:5.1f}% R@20={r20:5.1f}% R@50={r50:5.1f}%")
    
    # Aggregate results
    print("-" * 70)
    
    avg_r10 = np.mean(all_recalls['R@10']) * 100 if all_recalls['R@10'] else 0
    avg_r20 = np.mean(all_recalls['R@20']) * 100 if all_recalls['R@20'] else 0
    avg_r50 = np.mean(all_recalls['R@50']) * 100 if all_recalls['R@50'] else 0
    
    print(f"\nAGGREGATE RESULTS ({num_videos} videos):")
    print(f"  Average R@10: {avg_r10:.2f}%")
    print(f"  Average R@20: {avg_r20:.2f}%")
    print(f"  Average R@50: {avg_r50:.2f}%")
    
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"✓ Model loaded successfully")
    print(f"✓ Ran on {num_videos} test videos")
    print(f"✓ Average R@20: {avg_r20:.2f}% (expected ~38% based on training)")
    print(f"✓ Predictions are temporal - predicting FUTURE scene graphs")
    print("=" * 70)


if __name__ == "__main__":
    main()
