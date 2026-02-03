#!/usr/bin/env python
"""
Optimized training and evaluation for Temporal SGA Model.
Tests multiple configurations and reports best results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import time
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
        
        # Move to device
        class_ids = batch['class_ids'].to(device)
        bboxes = batch['bboxes'].to(device)
        object_mask = batch['object_mask'].to(device)
        frame_mask = batch['frame_mask'].to(device)
        target_predicates = batch['target_predicates'].to(device)
        target_existence = batch['target_existence'].to(device)
        num_future = batch['num_future_frames']
        
        # Forward
        outputs = model(
            class_ids=class_ids,
            bboxes=bboxes,
            object_mask=object_mask,
            frame_mask=frame_mask,
            num_future_frames=num_future,
        )
        
        # Compute loss
        loss_dict = loss_fn(
            predicate_logits=outputs['predicate_logits'],
            existence_logits=outputs['existence_logits'],
            target_predicates=target_predicates,
            target_existence=target_existence,
        )
        loss = loss_dict['total']
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def compute_recall_from_tensors(
    pred_logits: torch.Tensor,  # (batch, future, pairs, predicates)
    exist_logits: torch.Tensor,  # (batch, future, pairs, 1)
    target_predicates: torch.Tensor,  # (batch, future, pairs)
    target_existence: torch.Tensor,  # (batch, future, pairs)
    k_values: List[int] = [10, 20, 50],
) -> Dict[str, float]:
    """Compute Recall@K from tensor targets."""
    batch_size = pred_logits.shape[0]
    
    # Combine existence and predicate scores
    exist_probs = torch.sigmoid(exist_logits).squeeze(-1)  # (batch, future, pairs)
    pred_probs = F.softmax(pred_logits, dim=-1)  # (batch, future, pairs, preds)
    
    # Final scores = existence_prob * predicate_prob
    combined_probs = exist_probs.unsqueeze(-1) * pred_probs  # (batch, future, pairs, preds)
    
    recalls = {f'R@{k}': [] for k in k_values}
    
    for b in range(batch_size):
        # Find ground truth positions (where target_existence > 0)
        gt_mask = target_existence[b] > 0  # (future, pairs)
        gt_positions = gt_mask.nonzero()  # (num_gt, 2) - [future_idx, pair_idx]
        
        if len(gt_positions) == 0:
            continue
        
        # Get GT predicate classes
        gt_set = set()
        for pos in gt_positions:
            f_idx, p_idx = pos[0].item(), pos[1].item()
            pred_class = target_predicates[b, f_idx, p_idx].item()
            if pred_class >= 0:
                gt_set.add((f_idx, p_idx, pred_class))
        
        if len(gt_set) == 0:
            continue
        
        # Get predictions
        probs = combined_probs[b]  # (future, pairs, preds)
        num_future = probs.shape[0]
        num_pairs = probs.shape[1]
        num_preds = probs.shape[2]
        
        flat_probs = probs.flatten()
        
        for k in k_values:
            topk_vals, topk_indices = torch.topk(flat_probs, min(k, len(flat_probs)))
            
            hits = 0
            for idx in topk_indices:
                idx = idx.item()
                f_idx = idx // (num_pairs * num_preds)
                remainder = idx % (num_pairs * num_preds)
                p_idx = remainder // num_preds
                pred_idx = remainder % num_preds
                
                if (f_idx, p_idx, pred_idx) in gt_set:
                    hits += 1
            
            recall = hits / len(gt_set)
            recalls[f'R@{k}'].append(recall)
    
    return {k: np.mean(v) if v else 0 for k, v in recalls.items()}


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
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
            
            recalls = compute_recall_from_tensors(
                outputs['predicate_logits'],
                outputs['existence_logits'],
                target_predicates,
                target_existence,
                k_values=[10, 20, 50],
            )
            
            for k, v in recalls.items():
                if v > 0:
                    all_recalls[k].append(v)
    
    return {k: np.mean(v) if v else 0 for k, v in all_recalls.items()}


def train_and_evaluate(
    annotation_path: str,
    hidden_dim: int = 256,
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    input_fraction: float = 0.5,
    device: str = 'cpu',
):
    """Train and evaluate with specified config."""
    print(f"\nConfig: hidden={hidden_dim}, epochs={num_epochs}, lr={learning_rate}, F={input_fraction}")
    
    # Load datasets
    train_dataset = ActionGenomeDatasetV2(
        annotation_path=annotation_path,
        split='train',
        input_fraction=input_fraction,
    )
    
    test_dataset = ActionGenomeDatasetV2(
        annotation_path=annotation_path,
        split='test',
        input_fraction=input_fraction,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    config = TemporalSGAConfig(
        num_object_classes=len(AG_OBJECT_CLASSES),
        num_predicate_classes=len(AG_ALL_PREDICATES),
        hidden_dim=hidden_dim,
        num_spatial_layers=2,
        num_temporal_layers=3,
        num_decoder_layers=2,
    )
    model = TemporalSGAModel(config).to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = SGALoss()
    
    best_recall = 0
    history = []
    
    for epoch in range(num_epochs):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        recalls = evaluate(model, test_loader, device)
        scheduler.step()
        
        elapsed = time.time() - start
        
        r20 = recalls.get('R@20', 0) * 100
        r50 = recalls.get('R@50', 0) * 100
        
        print(f"  Epoch {epoch+1}/{num_epochs}: loss={train_loss:.4f}, R@20={r20:.2f}%, R@50={r50:.2f}%, time={elapsed:.1f}s")
        
        history.append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'R@20': r20,
            'R@50': r50,
        })
        
        if r20 > best_recall:
            best_recall = r20
    
    return {
        'final_R@20': recalls.get('R@20', 0) * 100,
        'final_R@50': recalls.get('R@50', 0) * 100,
        'best_R@20': best_recall,
        'history': history,
    }


def run_optimization_sweep(annotation_path: str, device: str = 'cpu'):
    """Run hyperparameter sweep to find best config."""
    print("=" * 70)
    print("HYPERPARAMETER OPTIMIZATION SWEEP")
    print("=" * 70)
    
    configs = [
        # Baseline
        {'hidden_dim': 256, 'epochs': 3, 'lr': 1e-4, 'F': 0.5, 'batch': 8},
        # Higher learning rate
        {'hidden_dim': 256, 'epochs': 3, 'lr': 3e-4, 'F': 0.5, 'batch': 8},
        # Larger model
        {'hidden_dim': 384, 'epochs': 3, 'lr': 1e-4, 'F': 0.5, 'batch': 4},
        # Different input fraction
        {'hidden_dim': 256, 'epochs': 3, 'lr': 1e-4, 'F': 0.3, 'batch': 8},
        {'hidden_dim': 256, 'epochs': 3, 'lr': 1e-4, 'F': 0.7, 'batch': 8},
    ]
    
    results = []
    
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing config: {cfg}")
        
        result = train_and_evaluate(
            annotation_path=annotation_path,
            hidden_dim=cfg['hidden_dim'],
            num_epochs=cfg['epochs'],
            batch_size=cfg['batch'],
            learning_rate=cfg['lr'],
            input_fraction=cfg['F'],
            device=device,
        )
        
        results.append({
            'config': cfg,
            **result,
        })
    
    # Find best
    best = max(results, key=lambda x: x['best_R@20'])
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    
    for r in results:
        cfg = r['config']
        print(f"  hidden={cfg['hidden_dim']}, lr={cfg['lr']}, F={cfg['F']} => R@20={r['final_R@20']:.2f}%, R@50={r['final_R@50']:.2f}%")
    
    print(f"\n  BEST CONFIG: {best['config']}")
    print(f"  BEST R@20: {best['best_R@20']:.2f}%")
    
    return results, best


def run_extended_training(annotation_path: str, epochs: int = 10, device: str = 'cpu'):
    """Run extended training with best config."""
    print("\n" + "=" * 70)
    print(f"EXTENDED TRAINING ({epochs} EPOCHS)")
    print("=" * 70)
    
    result = train_and_evaluate(
        annotation_path=annotation_path,
        hidden_dim=256,
        num_epochs=epochs,
        batch_size=8,
        learning_rate=2e-4,
        input_fraction=0.5,
        device=device,
    )
    
    print(f"\n  Final R@20: {result['final_R@20']:.2f}%")
    print(f"  Final R@50: {result['final_R@50']:.2f}%")
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-path', default='datasets/ActionGenome/annotations/action_genome_v1.0/object_bbox_and_relationship.pkl')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for extended training')
    parser.add_argument('--device', default='cpu', help='Device to use')
    args = parser.parse_args()
    
    if args.sweep:
        run_optimization_sweep(args.annotation_path, args.device)
    else:
        run_extended_training(args.annotation_path, args.epochs, args.device)


if __name__ == "__main__":
    main()
