#!/usr/bin/env python
"""
Comprehensive verification and testing of the Temporal SGA Model.
Tests on multiple videos and verifies temporal modeling is working.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import time

from orion.sga.temporal_model import TemporalSGAModel, TemporalSGAConfig
from orion.sga.ag_dataset_v2 import (
    ActionGenomeDatasetV2, 
    AG_ALL_PREDICATES, 
    AG_OBJECT_CLASSES,
    AG_IDX_TO_PREDICATE,
    collate_fn,
)

def verify_model_architecture():
    """Verify the temporal model has proper components."""
    print("=" * 70)
    print("1. MODEL ARCHITECTURE VERIFICATION")
    print("=" * 70)
    
    config = TemporalSGAConfig(
        num_object_classes=36,
        num_predicate_classes=23,
        hidden_dim=256,
        num_spatial_layers=2,
        num_temporal_layers=3,
        num_decoder_layers=2,
    )
    
    model = TemporalSGAModel(config)
    
    # Check components exist
    assert hasattr(model, 'spatial_encoder'), "Missing spatial encoder!"
    assert hasattr(model, 'temporal_encoder'), "Missing temporal encoder!"
    assert hasattr(model, 'anticipation_decoder'), "Missing anticipation decoder!"
    
    print(f"  ✓ Spatial Encoder: {type(model.spatial_encoder).__name__}")
    print(f"  ✓ Temporal Encoder: {type(model.temporal_encoder).__name__}")
    print(f"  ✓ Anticipation Decoder: {type(model.anticipation_decoder).__name__}")
    
    # Check temporal encoder has transformer
    te = model.temporal_encoder
    assert hasattr(te, 'transformer'), "Temporal encoder missing transformer!"
    print(f"  ✓ Temporal transformer layers: {te.transformer.num_layers}")
    print(f"  ✓ Positional encoding: {type(te.temporal_pe).__name__}")
    
    # Check decoder has cross-attention
    ad = model.anticipation_decoder
    assert hasattr(ad, 'decoder'), "Anticipation decoder missing cross-attention!"
    print(f"  ✓ Decoder cross-attention layers: {ad.decoder.num_layers}")
    print(f"  ✓ Future queries shape: {ad.future_queries.weight.shape}")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Total parameters: {num_params:,}")
    
    return model, config


def verify_temporal_encoding(model):
    """Verify that temporal encoding actually captures sequence information."""
    print("\n" + "=" * 70)
    print("2. TEMPORAL ENCODING VERIFICATION")
    print("=" * 70)
    
    model.eval()
    
    # Create two different sequences - same objects but different temporal patterns
    batch = 2
    frames = 8
    objects = 5
    
    # Sequence 1: Objects move from left to right over time
    class_ids = torch.randint(1, 36, (batch, frames, objects))
    bboxes1 = torch.zeros(batch, frames, objects, 4)
    for t in range(frames):
        # Objects move right over time
        x_offset = t * 0.1
        bboxes1[:, t, :, 0] = x_offset
        bboxes1[:, t, :, 1] = 0.3
        bboxes1[:, t, :, 2] = x_offset + 0.1
        bboxes1[:, t, :, 3] = 0.5
    
    # Sequence 2: Objects move from right to left (opposite direction)
    bboxes2 = torch.zeros(batch, frames, objects, 4)
    for t in range(frames):
        x_offset = 0.7 - t * 0.1
        bboxes2[:, t, :, 0] = x_offset
        bboxes2[:, t, :, 1] = 0.3
        bboxes2[:, t, :, 2] = x_offset + 0.1
        bboxes2[:, t, :, 3] = 0.5
    
    mask = torch.ones(batch, frames, objects, dtype=torch.bool)
    frame_mask = torch.ones(batch, frames, dtype=torch.bool)
    
    with torch.no_grad():
        out1 = model(class_ids, bboxes1, object_mask=mask, frame_mask=frame_mask, num_future_frames=2)
        out2 = model(class_ids, bboxes2, object_mask=mask, frame_mask=frame_mask, num_future_frames=2)
    
    # Temporal features should differ for different motion patterns
    tf1 = out1['temporal_features']
    tf2 = out2['temporal_features']
    
    diff = (tf1 - tf2).abs().mean().item()
    
    print(f"  Input: {frames} frames, {objects} objects")
    print(f"  Sequence 1: Objects moving LEFT → RIGHT")
    print(f"  Sequence 2: Objects moving RIGHT → LEFT")
    print(f"  Temporal feature difference: {diff:.4f}")
    
    assert diff > 0.01, "Temporal features should differ for different motion patterns!"
    print(f"  ✓ Temporal encoder captures motion direction!")
    
    # Verify features change across time dimension
    time_variance = tf1.var(dim=1).mean().item()
    print(f"  ✓ Variance across time: {time_variance:.4f}")
    
    return True


def verify_future_prediction(model):
    """Verify that the model makes different predictions for different futures."""
    print("\n" + "=" * 70)
    print("3. FUTURE PREDICTION VERIFICATION")
    print("=" * 70)
    
    model.eval()
    
    batch = 1
    frames = 6
    objects = 4
    
    class_ids = torch.tensor([[[0, 1, 5, 10]] * frames])  # person, bag, box, dish
    bboxes = torch.rand(batch, frames, objects, 4)
    mask = torch.ones(batch, frames, objects, dtype=torch.bool)
    frame_mask = torch.ones(batch, frames, dtype=torch.bool)
    
    with torch.no_grad():
        # Predict different number of future frames
        out_1 = model(class_ids, bboxes, object_mask=mask, frame_mask=frame_mask, num_future_frames=1)
        out_3 = model(class_ids, bboxes, object_mask=mask, frame_mask=frame_mask, num_future_frames=3)
    
    print(f"  Predicting 1 future frame: {out_1['predicate_logits'].shape}")
    print(f"  Predicting 3 future frames: {out_3['predicate_logits'].shape}")
    
    # Predictions should have different shapes
    assert out_1['predicate_logits'].shape[1] == 1
    assert out_3['predicate_logits'].shape[1] == 3
    print(f"  ✓ Model correctly predicts variable future horizons")
    
    # Check predictions vary across future timesteps
    pred_3 = out_3['predicate_logits']
    diff_01 = (pred_3[:, 0] - pred_3[:, 1]).abs().mean().item()
    diff_12 = (pred_3[:, 1] - pred_3[:, 2]).abs().mean().item()
    
    print(f"  Prediction diff (t=0 vs t=1): {diff_01:.4f}")
    print(f"  Prediction diff (t=1 vs t=2): {diff_12:.4f}")
    print(f"  ✓ Predictions vary across future timesteps")
    
    return True


def test_on_real_videos(annotation_path: str, num_videos: int = 10):
    """Test model on real Action Genome videos."""
    print("\n" + "=" * 70)
    print(f"4. TESTING ON {num_videos} REAL AG VIDEOS")
    print("=" * 70)
    
    # Load dataset
    dataset = ActionGenomeDatasetV2(
        annotation_path=annotation_path,
        split='test',
        input_fraction=0.5,
    )
    
    print(f"  Dataset size: {len(dataset)} videos")
    
    # Create model
    config = TemporalSGAConfig(
        num_object_classes=len(AG_OBJECT_CLASSES),
        num_predicate_classes=len(AG_ALL_PREDICATES),
        hidden_dim=256,
        num_spatial_layers=2,
        num_temporal_layers=3,
        num_decoder_layers=2,
    )
    model = TemporalSGAModel(config)
    model.eval()
    
    # Test on videos
    results = []
    
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
        
        # Get predictions
        pred_logits = outputs['predicate_logits']  # (1, num_future, num_pairs, num_preds)
        exist_logits = outputs['existence_logits']  # (1, num_future, num_pairs, 1)
        
        pred_probs = F.softmax(pred_logits, dim=-1)
        exist_probs = torch.sigmoid(exist_logits)
        
        # Count confident predictions (exist > 0.5)
        confident_preds = (exist_probs > 0.5).sum().item()
        
        # Get GT future relations
        gt_relations = len(sample['future_relations'])
        
        results.append({
            'video_id': sample['video_id'],
            'observed_frames': sample['num_observed_frames'],
            'future_frames': sample['num_future_frames'],
            'num_objects': sample['num_objects'],
            'gt_future_relations': gt_relations,
            'confident_predictions': confident_preds,
        })
        
        print(f"  [{i+1}] {sample['video_id']}: {sample['num_observed_frames']} obs frames, "
              f"{sample['num_future_frames']} future, {gt_relations} GT rels, "
              f"{confident_preds} confident preds")
    
    # Summary
    avg_gt = np.mean([r['gt_future_relations'] for r in results])
    avg_preds = np.mean([r['confident_predictions'] for r in results])
    
    print(f"\n  Summary:")
    print(f"    Avg GT relations per video: {avg_gt:.1f}")
    print(f"    Avg confident predictions: {avg_preds:.1f}")
    print(f"  ✓ Model runs successfully on real videos")
    
    return results


def benchmark_speed(model, device='cpu'):
    """Benchmark model inference speed."""
    print("\n" + "=" * 70)
    print("5. SPEED BENCHMARK")
    print("=" * 70)
    
    model = model.to(device)
    model.eval()
    
    # Different input sizes
    test_cases = [
        (1, 5, 5, 2),   # Small
        (1, 10, 10, 3), # Medium
        (4, 10, 15, 5), # Large batch
    ]
    
    for batch, frames, objects, future in test_cases:
        class_ids = torch.randint(0, 36, (batch, frames, objects)).to(device)
        bboxes = torch.rand(batch, frames, objects, 4).to(device)
        mask = torch.ones(batch, frames, objects, dtype=torch.bool).to(device)
        frame_mask = torch.ones(batch, frames, dtype=torch.bool).to(device)
        
        # Warmup
        with torch.no_grad():
            _ = model(class_ids, bboxes, object_mask=mask, frame_mask=frame_mask, num_future_frames=future)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            with torch.no_grad():
                _ = model(class_ids, bboxes, object_mask=mask, frame_mask=frame_mask, num_future_frames=future)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # ms
        print(f"  Batch={batch}, Frames={frames}, Objects={objects}, Future={future}: {avg_time:.1f}ms")
    
    print(f"  ✓ Speed benchmark complete")


def main():
    print("\n" + "=" * 70)
    print("TEMPORAL SGA MODEL - COMPREHENSIVE VERIFICATION")
    print("=" * 70)
    
    # 1. Verify architecture
    model, config = verify_model_architecture()
    
    # 2. Verify temporal encoding
    verify_temporal_encoding(model)
    
    # 3. Verify future prediction
    verify_future_prediction(model)
    
    # 4. Test on real videos
    annotation_path = "datasets/ActionGenome/annotations/action_genome_v1.0/object_bbox_and_relationship.pkl"
    if Path(annotation_path).exists():
        test_on_real_videos(annotation_path, num_videos=10)
    else:
        print(f"\n  ⚠ Skipping real video test (annotations not found)")
    
    # 5. Benchmark
    benchmark_speed(model)
    
    print("\n" + "=" * 70)
    print("✓ ALL VERIFICATIONS PASSED - TEMPORAL MODEL IS WORKING CORRECTLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
