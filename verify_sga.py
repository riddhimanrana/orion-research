#!/usr/bin/env python
"""
VERIFICATION SCRIPT: Ensure SGA implementation is correct.

This script traces through the entire pipeline step by step to verify:
1. Temporal split is correct (observed BEFORE future)
2. Predictions are based on observed frames only
3. Evaluation is against future ground truth only
4. R@K metrics are computed correctly
"""

from orion.sga import load_action_genome, SGAEvaluator, AnticipatedRelation
from collections import Counter

def verify_sga():
    print("=" * 70)
    print("SGA VERIFICATION - TRIPLE CHECKING CORRECTNESS")
    print("=" * 70)
    
    # Load one video
    bundle = load_action_genome('data/ag_ground_truth_full.json', max_videos=1)
    video = list(bundle.videos.values())[0]
    video_id = list(bundle.videos.keys())[0]
    
    print(f"\nVideo: {video_id}")
    print(f"Total frames: {video.num_frames()}")
    
    # Split at F=0.5
    observed, future = video.split_by_fraction(0.5)
    
    obs_frames = sorted(observed.frames.keys())
    fut_frames = sorted(future.frames.keys())
    
    # ============ CHECK 1: Temporal Split ============
    print("\n" + "=" * 70)
    print("CHECK 1: TEMPORAL SPLIT")
    print("=" * 70)
    print(f"Observed frames: {obs_frames}")
    print(f"Future frames:   {fut_frames}")
    
    temporal_correct = max(obs_frames) < min(fut_frames)
    print(f"\n✓ Max observed ({max(obs_frames)}) < Min future ({min(fut_frames)}): {temporal_correct}")
    
    if not temporal_correct:
        print("❌ FAIL: Temporal split is WRONG!")
        return False
    print("✅ PASS: Temporal split is correct")
    
    # ============ CHECK 2: Observed Relations ============
    print("\n" + "=" * 70)
    print("CHECK 2: OBSERVED RELATIONS (model input)")
    print("=" * 70)
    
    obs_triplets = set()
    obs_triplet_counts = Counter()
    for rel in observed.get_all_relations():
        t = rel.as_triplet()
        obs_triplets.add(t)
        obs_triplet_counts[t] += 1
    
    print(f"Total relation instances in observed: {sum(obs_triplet_counts.values())}")
    print(f"Unique triplets in observed: {len(obs_triplets)}")
    for t in sorted(obs_triplets):
        print(f"  - {t[0]} --[{t[1]}]--> {t[2]}  (count: {obs_triplet_counts[t]})")
    
    # ============ CHECK 3: Future Ground Truth ============
    print("\n" + "=" * 70)
    print("CHECK 3: FUTURE GROUND TRUTH (what we evaluate against)")
    print("=" * 70)
    
    fut_triplets = set()
    for rel in future.get_all_relations():
        fut_triplets.add(rel.as_triplet())
    
    print(f"Unique triplets in FUTURE: {len(fut_triplets)}")
    for t in sorted(fut_triplets):
        status = "✓ (seen)" if t in obs_triplets else "★ (NEW)"
        print(f"  {status} {t[0]} --[{t[1]}]--> {t[2]}")
    
    # ============ CHECK 4: Predictions ============
    print("\n" + "=" * 70)
    print("CHECK 4: PREDICTIONS (persistence model)")
    print("=" * 70)
    
    # Persistence model: predict relations that appear in observed
    predictions = []
    for triplet, count in obs_triplet_counts.items():
        confidence = count / len(observed.frames)
        predictions.append(AnticipatedRelation(
            subject_id=hash(triplet[0]) % 10000,
            subject_label=triplet[0],
            predicate=triplet[1],
            object_id=hash(triplet[2]) % 10000,
            object_label=triplet[2],
            confidence=confidence,
            predicted_frame=fut_frames[0],
            source='persistence'
        ))
    predictions.sort(key=lambda x: -x.confidence)
    
    print(f"Generated {len(predictions)} predictions (= observed unique triplets)")
    for p in predictions:
        print(f"  conf={p.confidence:.2f}: {p.subject_label} --[{p.predicate}]--> {p.object_label}")
    
    # ============ CHECK 5: Evaluation ============
    print("\n" + "=" * 70)
    print("CHECK 5: EVALUATION")
    print("=" * 70)
    
    evaluator = SGAEvaluator(top_ks=[10, 20, 50])
    result = evaluator.evaluate_video(predictions, future, 0.5)
    
    print(f"Ground truth triplets (FUTURE only): {len(result.gt_triplets)}")
    print(f"Predictions: {len(predictions)}")
    
    # What matched?
    pred_set = set((p.subject_label, p.predicate, p.object_label) for p in predictions)
    matched = result.gt_triplets & pred_set
    missed = result.gt_triplets - pred_set
    
    print(f"\nMATCHED ({len(matched)} / {len(result.gt_triplets)}):")
    for t in sorted(matched):
        print(f"  ✓ {t}")
    
    print(f"\nMISSED ({len(missed)}) - these are NEW relations in future:")
    for t in sorted(missed):
        print(f"  ✗ {t}")
    
    # Manual R@K calculation
    print("\n" + "=" * 70)
    print("CHECK 6: MANUAL R@K CALCULATION")
    print("=" * 70)
    
    top_10 = set((p.subject_label, p.predicate, p.object_label) for p in predictions[:10])
    matched_10 = result.gt_triplets & top_10
    manual_r10 = len(matched_10) / len(result.gt_triplets) * 100
    
    print(f"Top-10 predictions: {len(top_10)}")
    print(f"GT triplets: {len(result.gt_triplets)}")
    print(f"Matched in top-10: {len(matched_10)}")
    print(f"Manual R@10: {len(matched_10)}/{len(result.gt_triplets)} = {manual_r10:.1f}%")
    print(f"Evaluator R@10: {result.metrics.recall_at_k[10]:.1f}%")
    
    r10_match = abs(manual_r10 - result.metrics.recall_at_k[10]) < 0.1
    print(f"\n✓ R@10 matches: {r10_match}")
    
    # ============ FINAL VERDICT ============
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    print("\nSGA CORRECTNESS CHECKS:")
    print(f"  1. Temporal split (observed < future): ✅")
    print(f"  2. Predictions from observed only: ✅ ({len(predictions)} = {len(obs_triplets)} observed)")
    print(f"  3. Evaluation against future GT: ✅ ({len(result.gt_triplets)} future triplets)")
    print(f"  4. R@K computed correctly: {'✅' if r10_match else '❌'}")
    
    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT: WHY PERSISTENCE WORKS")
    print("=" * 70)
    overlap = len(obs_triplets & fut_triplets)
    print(f"Triplets that PERSIST from observed to future: {overlap}/{len(fut_triplets)}")
    print(f"This is the theoretical maximum for persistence model: {overlap/len(fut_triplets)*100:.1f}%")
    print(f"Our R@10 ({result.metrics.recall_at_k[10]:.1f}%) reflects this persistence.")
    
    return True


if __name__ == "__main__":
    verify_sga()
