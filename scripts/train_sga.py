#!/usr/bin/env python
"""
Train Temporal SGA Model on Action Genome

This script trains a temporal transformer model for Scene Graph Anticipation.
The model learns to predict FUTURE scene graph relations from OBSERVED frames.

Usage:
    python scripts/train_sga.py --epochs 50 --batch_size 8
    
    # With custom fraction
    python scripts/train_sga.py --input_fraction 0.3 --epochs 100
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from orion.sga.temporal_model import TemporalSGAModel, TemporalSGAConfig, SGALoss
from orion.sga.ag_dataset_v2 import (
    ActionGenomeDatasetV2, 
    create_dataloaders,
    AG_ALL_PREDICATES,
    AG_OBJECT_CLASSES,
    AG_IDX_TO_PREDICATE,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_metrics(
    pred_logits: torch.Tensor,
    exist_logits: torch.Tensor,
    target_preds: torch.Tensor,
    target_exist: torch.Tensor,
    k_values: list = [10, 20, 50],
) -> dict:
    """Compute R@K and mR@K metrics."""
    batch_size = pred_logits.size(0)
    
    # Flatten
    pred_flat = pred_logits.view(-1, pred_logits.size(-1))
    target_flat = target_preds.view(-1)
    exist_flat = target_exist.view(-1)
    
    # Get predictions
    pred_probs = F.softmax(pred_flat, dim=-1)
    pred_classes = pred_probs.argmax(dim=-1)
    pred_confs = pred_probs.max(dim=-1).values
    
    # Existence probability
    exist_probs = torch.sigmoid(exist_logits.view(-1))
    
    # Combined confidence
    combined_conf = pred_confs * exist_probs
    
    # Valid positions (where GT exists)
    valid = exist_flat > 0
    
    if valid.sum() == 0:
        return {f'R@{k}': 0.0 for k in k_values}
    
    # Check correctness
    correct = (pred_classes == target_flat) & valid
    
    # Sort by confidence
    sorted_idx = combined_conf.argsort(descending=True)
    sorted_valid = valid[sorted_idx]
    sorted_correct = correct[sorted_idx]
    
    # Only consider valid positions for ranking
    valid_indices = sorted_idx[sorted_valid[sorted_idx]]
    
    metrics = {}
    num_gt = valid.sum().item()
    
    for k in k_values:
        # Get top-k among valid predictions
        top_k_correct = sorted_correct[:k]
        hits = top_k_correct.sum().item()
        recall = (hits / num_gt) * 100 if num_gt > 0 else 0
        metrics[f'R@{k}'] = recall
    
    return metrics


def train_epoch(
    model: TemporalSGAModel,
    loader,
    optimizer,
    loss_fn: SGALoss,
    device: str,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_pred_loss = 0.0
    total_exist_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        # Move to device
        class_ids = batch['class_ids'].to(device)
        bboxes = batch['bboxes'].to(device)
        object_mask = batch['object_mask'].to(device)
        frame_mask = batch['frame_mask'].to(device)
        target_preds = batch['target_predicates'].to(device)
        target_exist = batch['target_existence'].to(device)
        
        # Forward
        outputs = model(
            class_ids=class_ids,
            bboxes=bboxes,
            object_mask=object_mask,
            frame_mask=frame_mask,
            num_future_frames=batch['num_future_frames'],
        )
        
        # Loss
        losses = loss_fn(
            outputs['predicate_logits'],
            outputs['existence_logits'],
            target_preds,
            target_exist,
        )
        
        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track
        total_loss += losses['total'].item()
        total_pred_loss += losses['predicate'].item()
        total_exist_loss += losses['existence'].item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f"{losses['total'].item():.3f}"})
    
    return {
        'loss': total_loss / num_batches,
        'pred_loss': total_pred_loss / num_batches,
        'exist_loss': total_exist_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: TemporalSGAModel,
    loader,
    loss_fn: SGALoss,
    device: str,
) -> dict:
    """Evaluate on validation set."""
    model.eval()
    
    total_loss = 0.0
    all_metrics = []
    num_batches = 0
    
    for batch in tqdm(loader, desc="Evaluating"):
        # Move to device
        class_ids = batch['class_ids'].to(device)
        bboxes = batch['bboxes'].to(device)
        object_mask = batch['object_mask'].to(device)
        frame_mask = batch['frame_mask'].to(device)
        target_preds = batch['target_predicates'].to(device)
        target_exist = batch['target_existence'].to(device)
        
        # Forward
        outputs = model(
            class_ids=class_ids,
            bboxes=bboxes,
            object_mask=object_mask,
            frame_mask=frame_mask,
            num_future_frames=batch['num_future_frames'],
        )
        
        # Loss
        losses = loss_fn(
            outputs['predicate_logits'],
            outputs['existence_logits'],
            target_preds,
            target_exist,
        )
        
        total_loss += losses['total'].item()
        
        # Metrics
        metrics = compute_metrics(
            outputs['predicate_logits'],
            outputs['existence_logits'],
            target_preds,
            target_exist,
        )
        all_metrics.append(metrics)
        num_batches += 1
    
    # Aggregate metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    avg_metrics['loss'] = total_loss / num_batches
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Temporal SGA Model")
    
    # Data
    parser.add_argument(
        '--annotation_path',
        type=str,
        default='datasets/ActionGenome/annotations/action_genome_v1.0/object_bbox_and_relationship.pkl',
    )
    parser.add_argument('--input_fraction', type=float, default=0.5)
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--num_spatial_layers', type=int, default=2)
    parser.add_argument('--num_temporal_layers', type=int, default=3)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints/sga')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, cuda, mps')
    
    args = parser.parse_args()
    
    # Check data exists
    if not Path(args.annotation_path).exists():
        logger.error(f"Annotations not found: {args.annotation_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Loading data...")
    train_loader, val_loader = create_dataloaders(
        annotation_path=args.annotation_path,
        input_fraction=args.input_fraction,
        batch_size=args.batch_size,
        num_workers=0,  # MPS doesn't like multiprocessing
    )
    
    logger.info(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    # Create model
    logger.info("Creating model...")
    config = TemporalSGAConfig(
        num_object_classes=len(AG_OBJECT_CLASSES),
        num_predicate_classes=len(AG_ALL_PREDICATES),
        hidden_dim=args.hidden_dim,
        num_spatial_layers=args.num_spatial_layers,
        num_temporal_layers=args.num_temporal_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    
    model = TemporalSGAModel(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    loss_fn = SGALoss(predicate_weight=1.0, existence_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    best_recall = 0.0
    history = {'train': [], 'val': []}
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)
        history['train'].append(train_metrics)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        history['val'].append(val_metrics)
        
        # Step scheduler
        scheduler.step()
        
        # Log
        logger.info(
            f"Epoch {epoch}/{args.epochs}: "
            f"Train Loss={train_metrics['loss']:.4f}, "
            f"Val Loss={val_metrics['loss']:.4f}, "
            f"R@20={val_metrics.get('R@20', 0):.2f}%, "
            f"R@50={val_metrics.get('R@50', 0):.2f}%"
        )
        
        # Save best model
        r20 = val_metrics.get('R@20', 0)
        if r20 > best_recall:
            best_recall = r20
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'metrics': val_metrics,
            }, output_dir / 'best_model.pt')
            logger.info(f"  → New best! R@20={best_recall:.2f}%")
        
        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
    }, output_dir / 'final_model.pt')
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\nTraining complete!")
    logger.info(f"Best R@20: {best_recall:.2f}%")
    logger.info(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
