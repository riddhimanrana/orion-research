#!/usr/bin/env python
"""
Final optimized training for Temporal SGA Model.
Uses best hyperparameters found from sweep.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import time
import json
from typing import List, Dict
from torch.utils.data import DataLoader

from orion.sga.temporal_model import TemporalSGAModel, TemporalSGAConfig, SGALoss
from orion.sga.ag_dataset_v2 import (
    ActionGenomeDatasetV2, 
    AG_ALL_PREDICATES, 
    AG_OBJECT_CLASSES,
    collate_fn,
)


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        class_ids = batch['class_ids'].to(device)
        bboxes = batch['bboxes'].to(device)
        object_mask = batch['object_mask'].to(device)
        frame_mask = batch['frame_mask'].to(device)
        target_predicates = batch['target_predicates'].to(device)
        target_existence = batch['target_existence'].to(device)
        num_future = batch['num_future_frames']
        
        outputs = model(
            class_ids=class_ids,
            bboxes=bboxes,
            object_mask=object_mask,
            frame_mask=frame_mask,
            num_future_frames=num_future,
        )
        
        loss_dict = loss_fn(
            predicate_logits=outputs['predicate_logits'],
            existence_logits=outputs['existence_logits'],
            target_predicates=target_predicates,
            target_existence=target_existence,
        )
        loss = loss_dict['total']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def compute_recall(pred_logits, exist_logits, target_predicates, target_existence, k_values=[10, 20, 50]):
    """Compute Recall@K."""
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
    
    return {k: np.mean(v) if v else 0 for k, v in recalls.items()}


def evaluate(model, test_loader, device):
    """Evaluate model."""
    model.eval()
    all_recalls = defaultdict(list)
    
    with torch.no_grad():
        for batch in test_loader:
            class_ids = batch['class_ids'].to(device)
            bboxes = batch['bboxes'].to(device)
            object_mask = batch['object_mask'].to(device)
            frame_mask = batch['frame_mask'].to(device)
            target_predicates = batch['target_predicates'].to(device)
            target_existence = batch['target_existence'].to(device)
            num_future = batch['num_future_frames']
            
            outputs = model(
                class_ids=class_ids,
                bboxes=bboxes,
                object_mask=object_mask,
                frame_mask=frame_mask,
                num_future_frames=num_future,
            )
            
            recalls = compute_recall(
                outputs['predicate_logits'],
                outputs['existence_logits'],
                target_predicates,
                target_existence,
            )
            
            for k, v in recalls.items():
                if v > 0:
                    all_recalls[k].append(v)
    
    return {k: np.mean(v) if v else 0 for k, v in all_recalls.items()}


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-path', default='datasets/ActionGenome/annotations/action_genome_v1.0/object_bbox_and_relationship.pkl')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden-dim', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--input-fraction', type=float, default=0.5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save-model', default='models/temporal_sga_best.pt')
    args = parser.parse_args()
    
    print("=" * 70)
    print("OPTIMIZED TEMPORAL SGA TRAINING")
    print("=" * 70)
    print(f"Config: hidden={args.hidden_dim}, lr={args.lr}, F={args.input_fraction}, epochs={args.epochs}")
    
    # Load data
    train_dataset = ActionGenomeDatasetV2(args.annotation_path, split='train', input_fraction=args.input_fraction)
    test_dataset = ActionGenomeDatasetV2(args.annotation_path, split='test', input_fraction=args.input_fraction)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Model
    config = TemporalSGAConfig(
        num_object_classes=len(AG_OBJECT_CLASSES),
        num_predicate_classes=len(AG_ALL_PREDICATES),
        hidden_dim=args.hidden_dim,
        num_spatial_layers=2,
        num_temporal_layers=3,
        num_decoder_layers=2,
    )
    model = TemporalSGAModel(config).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = SGALoss()
    
    best_recall = 0
    best_state = None
    
    print("\n" + "-" * 70)
    
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, args.device)
        recalls = evaluate(model, test_loader, args.device)
        scheduler.step()
        
        elapsed = time.time() - start
        r10, r20, r50 = recalls.get('R@10', 0) * 100, recalls.get('R@20', 0) * 100, recalls.get('R@50', 0) * 100
        
        improved = r20 > best_recall
        if improved:
            best_recall = r20
            best_state = model.state_dict()
        
        marker = " *" if improved else ""
        print(f"Epoch {epoch+1:2d}/{args.epochs}: loss={train_loss:.4f} | R@10={r10:.2f}% R@20={r20:.2f}% R@50={r50:.2f}% | {elapsed:.1f}s{marker}")
    
    print("-" * 70)
    
    # Save best model
    if best_state and args.save_model:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'config': config,
            'state_dict': best_state,
            'best_r20': best_recall,
        }, args.save_model)
        print(f"\nSaved best model to {args.save_model} (R@20={best_recall:.2f}%)")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Best R@20: {best_recall:.2f}%")
    print(f"  Final R@10: {r10:.2f}%")
    print(f"  Final R@20: {r20:.2f}%")
    print(f"  Final R@50: {r50:.2f}%")


if __name__ == "__main__":
    main()
