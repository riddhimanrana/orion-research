#!/usr/bin/env python
"""Test script for the temporal SGA model."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from orion.sga.temporal_model import TemporalSGAModel, TemporalSGAConfig, SGALoss

def test_temporal_model():
    print("Testing TemporalSGAModel...")
    
    # Create config
    config = TemporalSGAConfig(
        num_object_classes=36,
        num_predicate_classes=23,
        hidden_dim=256,  # Smaller for test
        num_spatial_layers=1,
        num_temporal_layers=2,
        num_decoder_layers=1,
    )
    
    # Create model
    model = TemporalSGAModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Create dummy input
    batch_size = 2
    num_frames = 5
    num_objects = 4
    
    class_ids = torch.randint(0, 36, (batch_size, num_frames, num_objects))
    bboxes = torch.rand(batch_size, num_frames, num_objects, 4)
    object_mask = torch.ones(batch_size, num_frames, num_objects, dtype=torch.bool)
    frame_mask = torch.ones(batch_size, num_frames, dtype=torch.bool)
    
    # Forward pass
    print("Running forward pass...")
    outputs = model(
        class_ids, bboxes,
        object_mask=object_mask,
        frame_mask=frame_mask,
        num_future_frames=2,
    )
    
    print(f"Predicate logits shape: {outputs['predicate_logits'].shape}")
    print(f"Existence logits shape: {outputs['existence_logits'].shape}")
    
    # Test loss
    print("Testing loss computation...")
    loss_fn = SGALoss()
    
    num_pairs = num_objects * (num_objects - 1)
    target_predicates = torch.randint(0, 23, (batch_size, 2, num_pairs))
    target_existence = torch.randint(0, 2, (batch_size, 2, num_pairs)).float()
    
    losses = loss_fn(
        outputs['predicate_logits'],
        outputs['existence_logits'],
        target_predicates,
        target_existence,
    )
    print(f"Total loss: {losses['total'].item():.4f}")
    print(f"Predicate loss: {losses['predicate'].item():.4f}")
    print(f"Existence loss: {losses['existence'].item():.4f}")
    
    # Test prediction
    print("Testing predict method...")
    predictions = model.predict(
        class_ids, bboxes,
        object_mask=object_mask,
        frame_mask=frame_mask,
        num_future_frames=2,
        threshold=0.3,
    )
    print(f"Predictions per batch: {[len(p) for p in predictions]}")
    
    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    test_temporal_model()
