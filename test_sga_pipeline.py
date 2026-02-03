#!/usr/bin/env python
"""
End-to-end SGA Pipeline Test

Tests the full Scene Graph Anticipation pipeline on Action Genome videos.
Uses GT observed relations to predict future relations (GAGS mode).
"""

from orion.sga import (
    load_action_genome, 
    SGAEvaluator, 
    AnticipatedRelation,
)
from collections import Counter

def predict_from_observed_relations(observed_video, future_frame_ids):
    """
    Predict future relations from observed GT relations.
    
    Uses persistence model: if a relation appears in observed frames,
    it's likely to persist in future frames.
    """
    # Count relation occurrences in observed frames
    relation_counts = Counter()
    relation_info = {}
    
    for frame in observed_video.frames.values():
        for rel in frame.relations:
            triplet = rel.as_triplet()
            relation_counts[triplet] += 1
            relation_info[triplet] = {
                'subject_label': triplet[0],
                'predicate': triplet[1], 
                'object_label': triplet[2],
            }
    
    # Total observed frames
    total_frames = len(observed_video.frames)
    
    # Generate predictions with confidence based on persistence
    predictions = []
    for triplet, count in relation_counts.items():
        # Confidence = fraction of frames where relation appeared
        confidence = count / total_frames
        
        info = relation_info[triplet]
        pred = AnticipatedRelation(
            subject_id=hash(triplet[0]) % 10000,
            subject_label=info['subject_label'],
            predicate=info['predicate'],
            object_id=hash(triplet[2]) % 10000,
            object_label=info['object_label'],
            confidence=confidence,
            predicted_frame=future_frame_ids[0] if future_frame_ids else 0,
            source='persistence',
        )
        predictions.append(pred)
    
    # Sort by confidence
    predictions.sort(key=lambda x: -x.confidence)
    return predictions


def main():
    print("=" * 60)
    print("SGA PIPELINE END-TO-END TEST")
    print("=" * 60)
    
    # Load AG data
    print("\n[1] Loading Action Genome data...")
    bundle = load_action_genome('data/ag_ground_truth_full.json', max_videos=1000)
    print(f"    Loaded {bundle.num_videos()} videos")
    print(f"    Total predicates: {len(bundle.predicates)}")
    
    # Initialize evaluator
    print("\n[2] Initializing evaluator...")
    evaluator = SGAEvaluator(top_ks=[10, 20, 50])
    
    # Test with different fractions
    for fraction in [0.3, 0.5, 0.7]:
        print(f"\n{'='*60}")
        print(f"TESTING WITH F={fraction} (observe {int(fraction*100)}%, predict {int((1-fraction)*100)}%)")
        print("="*60)
        
        results = []
        
        for video_id, video in list(bundle.videos.items()):
            observed, future = video.split_by_fraction(fraction)
            
            if future.num_frames() == 0 or observed.num_frames() == 0:
                continue
            
            # Get future frame IDs
            future_frame_ids = list(future.frames.keys())
            
            # Predict from observed relations (persistence model)
            predictions = predict_from_observed_relations(observed, future_frame_ids)
            
            # Evaluate against GT future
            result = evaluator.evaluate_video(predictions, future, fraction)
            results.append(result)
            
            r10 = result.metrics.recall_at_k.get(10, 0)
            r20 = result.metrics.recall_at_k.get(20, 0)
            r50 = result.metrics.recall_at_k.get(50, 0)
            
            print(f"  {video_id}: GT={len(result.gt_triplets):3d}, Pred={len(predictions):3d} | R@10={r10:5.1f}% R@20={r20:5.1f}% R@50={r50:5.1f}%")
        
        # Aggregate results
        if results:
            print(f"\n--- AGGREGATE (F={fraction}, {len(results)} videos) ---")
            
            for k in [10, 20, 50]:
                avg_r = sum(r.metrics.recall_at_k.get(k, 0) for r in results) / len(results)
                avg_mr = sum(r.metrics.mean_recall_at_k.get(k, 0) for r in results) / len(results)
                print(f"  R@{k}: {avg_r:5.2f}%  |  mR@{k}: {avg_mr:5.2f}%")
            
            total_gt = sum(len(r.gt_triplets) for r in results)
            total_pred = sum(len(r.pred_triplets) for r in results)
            print(f"\n  Total GT triplets: {total_gt}")
            print(f"  Total predictions: {total_pred}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
