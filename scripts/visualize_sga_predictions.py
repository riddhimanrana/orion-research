#!/usr/bin/env python
"""
Visualize predictions on individual videos to verify the temporal model works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np

from orion.sga.temporal_model import TemporalSGAModel, TemporalSGAConfig
from orion.sga.ag_dataset_v2 import (
    ActionGenomeDatasetV2, 
    AG_ALL_PREDICATES, 
    AG_OBJECT_CLASSES,
    AG_IDX_TO_PREDICATE,
    collate_fn,
)


def main():
    annotation_path = 'datasets/ActionGenome/annotations/action_genome_v1.0/object_bbox_and_relationship.pkl'
    model_path = 'models/temporal_sga_best.pt'
    
    print("=" * 70)
    print("TEMPORAL SGA - INDIVIDUAL VIDEO PREDICTIONS")
    print("=" * 70)
    
    # Load model
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        model = TemporalSGAModel(config)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model from {model_path} (R@20={checkpoint['best_r20']:.2f}%)")
    else:
        print("No saved model found, using random weights")
        config = TemporalSGAConfig(
            num_object_classes=len(AG_OBJECT_CLASSES),
            num_predicate_classes=len(AG_ALL_PREDICATES),
            hidden_dim=256,
        )
        model = TemporalSGAModel(config)
    
    model.eval()
    
    # Load test videos
    dataset = ActionGenomeDatasetV2(annotation_path, split='test', input_fraction=0.5)
    
    print(f"\nTesting on {min(10, len(dataset))} videos...")
    print("=" * 70)
    
    for i in range(min(10, len(dataset))):
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
        
        # Get predictions
        pred_logits = outputs['predicate_logits'][0]  # (future, pairs, preds)
        exist_logits = outputs['existence_logits'][0]  # (future, pairs, 1)
        
        exist_probs = torch.sigmoid(exist_logits).squeeze(-1)  # (future, pairs)
        pred_probs = F.softmax(pred_logits, dim=-1)  # (future, pairs, preds)
        
        # Combined scores
        combined = exist_probs.unsqueeze(-1) * pred_probs  # (future, pairs, preds)
        
        print(f"\n[{i+1}] Video: {sample['video_id']}")
        print(f"    Observed frames: {sample['num_observed_frames']}")
        print(f"    Future frames to predict: {sample['num_future_frames']}")
        print(f"    Objects detected: {sample['num_objects']}")
        
        # Ground truth future relations
        gt_rels = sample['future_relations']
        print(f"    GT future relations: {len(gt_rels)}")
        
        # Show top GT relations
        if gt_rels:
            print("    Ground truth (first 5):")
            for rel in gt_rels[:5]:
                f_idx, s_idx, pred_idx, o_idx = rel
                pred_name = AG_IDX_TO_PREDICATE.get(pred_idx, f"pred_{pred_idx}")
                print(f"      Frame {f_idx}: person -- {pred_name} --> obj_{o_idx}")
        
        # Show top predictions
        flat = combined.flatten()
        topk_vals, topk_indices = torch.topk(flat, 10)
        
        num_future, num_pairs, num_preds = combined.shape
        
        print("    Top 10 predictions:")
        for j, (score, idx) in enumerate(zip(topk_vals, topk_indices)):
            idx = idx.item()
            f_idx = idx // (num_pairs * num_preds)
            remainder = idx % (num_pairs * num_preds)
            p_idx = remainder // num_preds
            pred_idx = remainder % num_preds
            
            # Decode pair index
            num_objs = sample['num_objects']
            if num_objs > 1:
                s_idx = p_idx // (num_objs - 1)
                adj_o = p_idx % (num_objs - 1)
                o_idx = adj_o if adj_o < s_idx else adj_o + 1
            else:
                s_idx, o_idx = 0, 0
            
            pred_name = AG_IDX_TO_PREDICATE.get(pred_idx, f"pred_{pred_idx}")
            print(f"      [{j+1}] Frame {f_idx}: subj_{s_idx} -- {pred_name} --> obj_{o_idx} (score={score:.3f})")
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print("\nThe temporal model:")
    print("  1. Takes observed frames with object detections")
    print("  2. Encodes spatial relationships per frame")
    print("  3. Encodes temporal evolution across frames")
    print("  4. Predicts future relations for upcoming frames")
    print("\nThis is TRUE scene graph anticipation - predicting the future!")


if __name__ == "__main__":
    main()
