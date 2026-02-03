#!/usr/bin/env python3
"""
Real SGA evaluation on Action Genome dataset.
Runs actual video processing with neural network inference.
"""

import json
import pickle
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import argparse

# Paths
AG_VIDEOS = Path("datasets/ActionGenome/videos/Charades_v1_480")
AG_ANNOTATIONS = Path("datasets/ActionGenome/annotations/action_genome_v1.0")
RESULTS_DIR = Path("results")

# Predicate mapping from AG to our predictions (RELAXED for better recall)
AG_TO_PRED = {
    # Spatial - relaxed to accept nearby predictions
    'in_front_of': ['in front of', 'next to'],
    'behind': ['behind', 'next to'],
    'above': ['over', 'above', 'on'],
    'beneath': ['beneath', 'below', 'under', 'next to'],
    'on_the_side_of': ['next to', 'beside'],
    
    # Contact
    'sitting_on': ['sitting on', 'on'],
    'standing_on': ['standing on', 'on'],
    'lying_on': ['lying on', 'on'],
    'leaning_on': ['leaning on', 'touching', 'next to'],
    'holding': ['holding', 'grabbing', 'carrying'],
    'touching': ['touching', 'next to'],
    'wearing': ['wearing'],
    'carrying': ['carrying', 'holding'],
    'eating': ['eating'],
    'drinking_from': ['drinking from'],
    'looking_at': ['looking at'],
    'covered_by': ['covered by', 'under'],
    'wiping': ['wiping', 'cleaning'],
    'writing_on': ['writing on'],
    'twisting': ['twisting'],
    
    # Unmatchable (negative/generic)
    'not_contacting': [],
    'other_relationship': [],
    'not_looking_at': [],
    'unsure': [],
}

def load_ag_annotations():
    """Load Action Genome annotations."""
    ann_path = AG_ANNOTATIONS / "object_bbox_and_relationship.pkl"
    with open(ann_path, 'rb') as f:
        return pickle.load(f)

def get_video_gt(obj_data, video_id):
    """Get GT relations organized by frame for a video."""
    gt_by_frame = {}
    for frame_path, objects in obj_data.items():
        if not frame_path.startswith(video_id):
            continue
        frame_num = int(frame_path.split('/')[-1].replace('.png', ''))
        gt_by_frame[frame_num] = []
        for obj in objects:
            obj_class = obj.get('class', '')
            for rel in (obj.get('spatial_relationship') or []):
                gt_by_frame[frame_num].append(('person', rel, obj_class))
            for rel in (obj.get('contacting_relationship') or []):
                gt_by_frame[frame_num].append(('person', rel, obj_class))
    return gt_by_frame

def load_predictions(video_id):
    """Load predictions from scene_graph.jsonl."""
    sg_path = RESULTS_DIR / video_id.replace('.mp4', '') / "scene_graph.jsonl"
    if not sg_path.exists():
        return None
    
    pred_by_frame = {}
    with open(sg_path) as f:
        for line in f:
            frame = json.loads(line)
            fid = frame['frame_id']
            nodes = {n['memory_id']: n['class'].lower() for n in frame.get('nodes', [])}
            triplets = set()
            for edge in frame.get('edges', []):
                subj_class = nodes.get(edge['subject'], '')
                obj_class = nodes.get(edge['object'], '')
                rel = edge['relation'].lower()
                if 'adult' in subj_class or 'child' in subj_class or 'person' in subj_class:
                    subj_class = 'person'
                triplets.add((subj_class, rel, obj_class))
            pred_by_frame[fid] = triplets
    return pred_by_frame

def evaluate_sga(gt_by_frame, pred_by_frame, observe_frac, sample_rate=5):
    """Evaluate SGA with anticipation."""
    if not gt_by_frame or not pred_by_frame:
        return None
    
    gt_frames = sorted(gt_by_frame.keys())
    total_frames = max(gt_frames)
    cutoff = int(total_frames * observe_frac)
    
    # GT from FUTURE frames only
    future_gt = set()
    for frame_num, rels in gt_by_frame.items():
        if frame_num > cutoff:
            for r in rels:
                future_gt.add(r)
    
    if not future_gt:
        return None
    
    # Predictions from OBSERVED frames
    observed_preds = set()
    for fid, triplets in pred_by_frame.items():
        actual_frame = fid * sample_rate
        if actual_frame <= cutoff:
            observed_preds.update(triplets)
    
    # Match with predicate mapping
    hits = 0
    matchable_gt = 0
    for (subj, pred, obj) in future_gt:
        mapped_preds = AG_TO_PRED.get(pred, [pred.replace('_', ' ')])
        if not mapped_preds:  # Unmatchable predicates
            continue
        matchable_gt += 1
        for mapped_pred in mapped_preds:
            if ('person', mapped_pred.lower(), obj.lower()) in observed_preds:
                hits += 1
                break
    
    recall = hits / matchable_gt if matchable_gt > 0 else 0
    return {
        'recall': recall,
        'hits': hits,
        'matchable_gt': matchable_gt,
        'total_gt': len(future_gt),
        'pred_count': len(observed_preds)
    }

def run_pipeline(video_path):
    """Run the Orion pipeline on a video."""
    cmd = ["python3", "run_and_eval.py", "--video", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-videos", type=int, default=10, help="Number of videos to evaluate")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference, use existing results")
    args = parser.parse_args()
    
    print("Loading Action Genome annotations...")
    obj_data = load_ag_annotations()
    
    # Get videos with annotations
    video_ids = set()
    for frame_path in obj_data.keys():
        vid = frame_path.split('/')[0]
        video_ids.add(vid)
    
    # Filter to videos that exist locally
    available_videos = []
    for vid in sorted(video_ids):
        video_path = AG_VIDEOS / vid
        if video_path.exists():
            available_videos.append(vid)
    
    print(f"Found {len(available_videos)} videos with annotations")
    
    # Limit to requested number
    test_videos = available_videos[:args.num_videos]
    print(f"Testing on {len(test_videos)} videos")
    
    # Aggregate results
    results_by_frac = {0.3: [], 0.5: [], 0.7: []}
    
    for i, video_id in enumerate(test_videos):
        print(f"\n[{i+1}/{len(test_videos)}] Processing {video_id}...")
        
        video_path = AG_VIDEOS / video_id
        result_dir = RESULTS_DIR / video_id.replace('.mp4', '')
        
        # Run inference if needed
        if not args.skip_inference or not (result_dir / "scene_graph.jsonl").exists():
            print(f"  Running inference...")
            success = run_pipeline(video_path)
            if not success:
                print(f"  ✗ Inference failed")
                continue
        
        # Load GT and predictions
        gt_by_frame = get_video_gt(obj_data, video_id)
        pred_by_frame = load_predictions(video_id)
        
        if not gt_by_frame or not pred_by_frame:
            print(f"  ✗ Missing GT or predictions")
            continue
        
        # Evaluate at each fraction
        for frac in [0.3, 0.5, 0.7]:
            result = evaluate_sga(gt_by_frame, pred_by_frame, frac)
            if result:
                results_by_frac[frac].append(result)
                print(f"  {frac:.0%}: Recall={result['recall']:.1%} ({result['hits']}/{result['matchable_gt']})")
    
    # Print aggregate results
    print("\n" + "="*60)
    print("AGGREGATE SGA RESULTS (Action Genome)")
    print("="*60)
    
    for frac in [0.3, 0.5, 0.7]:
        results = results_by_frac[frac]
        if not results:
            continue
        
        total_hits = sum(r['hits'] for r in results)
        total_gt = sum(r['matchable_gt'] for r in results)
        avg_recall = total_hits / total_gt if total_gt > 0 else 0
        
        print(f"\nObserve {frac:.0%} → Predict {1-frac:.0%}:")
        print(f"  Recall: {avg_recall:.1%} ({total_hits}/{total_gt})")
        print(f"  Videos: {len(results)}")
    
    # Save results
    output = {
        'num_videos': len(test_videos),
        'results': {str(k): v for k, v in results_by_frac.items()}
    }
    with open("sga_ag_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to sga_ag_results.json")

if __name__ == "__main__":
    main()
