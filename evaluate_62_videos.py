#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from pathlib import Path
import json

VIDORS = [
    "0004_11566980553", "0010_8610561401", "0018_4748191834", "0027_4571353789",
    "0028_4021064662", "0039_6951351121", "0046_11919433184", "0051_3702633786",
    "0053_5599511471", "0054_2612939953", "0057_7001078933", "0062_6430774273",
    "0069_2740320945", "0075_11566764085", "0096_5296138427", "1000_6828150903",
    "1001_7007447516", "1002_5280626374", "1005_4760962392", "1005_7031128593",
    "1005_7401573420", "1006_4580824633", "1007_6631583821", "1011_4633647136",
    "1012_4024008346", "1015_4698622422", "1017_3056841458", "1019_3768851893",
    "1020_2471845614", "1021_3478653250", "1021_4278168115", "1025_4615486172",
    "1025_6244382586", "1052_8530515192", "1100_9117425466", "1122_3393449055",
    "1124_9861436503", "1161_5895320023", "1164_6895784766", "1203_8316378691"
]

EPIC_KITCHENS = [
    "P01_03", "P02_10", "P03_06", "P04_27", "P05_05", "P08_07", "P09_07", 
    "P11_11", "P14_06", "P19_06", "P28_19"
]

EGO4DS = [
    "0be30efe-9d71-4698-8304-f1d441aeea58_1", "1bfe5ac2-cbf8-4364-8a30-60d97dd395df_1",
    "22cc4d54-34be-4580-983a-9e710e831c16", "43b0205a-4e3c-46a7-9d1c-c04ead730180",
    "6e0a6558-c212-4cab-b374-007671edb59c_2", "8be918b2-c819-4a84-98dc-5fe24835a4ac",
    "c20407ac-83d6-4c84-88cb-63bced9d456b", "c2e6d807-d903-4b64-98e1-2c07ca700c78_2",
    "d1d4a1b3-a651-4eb8-bb7f-8d66982854fa", "d2222009-a717-4b16-91ce-6399c5bb798a",
    "eed8d8d7-6773-493b-af21-880f0acb063a"
]

ALL_VIDEOS = VIDORS + EPIC_KITCHENS + EGO4DS

def find_video_path(video_id):
    """Find video path in datasets/PVSG/"""
    for ext in ['.mp4', '.MP4']:
        cmd = ["find", "datasets/PVSG", "-name", f"{video_id}{ext}"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            path = res.stdout.strip().split('\n')[0]
            if path:
                return path
        except:
            continue
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Just find paths and check existence")
    parser.add_argument("--backend", default="groundingdino", choices=["yoloworld", "groundingdino"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process")
    args = parser.parse_args()

    print(f"Total videos to process: {len(ALL_VIDEOS)}", flush=True)
    
    video_map = {}
    missing = []
    for vid in ALL_VIDEOS:
        path = find_video_path(vid)
        if path:
            video_map[vid] = path
        else:
            missing.append(vid)

    if missing:
        print(f"Warning: Missing {len(missing)} videos:")
        for m in missing:
            print(f"  - {m}", flush=True)
    else:
        print("All video files found.", flush=True)

    if args.dry_run:
        return

    to_process = list(video_map.keys())
    if args.limit:
        to_process = to_process[:args.limit]

    print(f"\nProcessing {len(to_process)} videos...", flush=True)
    
    for i, vid in enumerate(to_process):
        print(f"\n[{i+1}/{len(to_process)}] Processing {vid}...", flush=True)
        path = video_map[vid]
        
        # Check if already processed
        out_dir = Path("results") / vid
        if (out_dir / "scene_graph.jsonl").exists():
            print(f"  Results already exist for {vid}, skipping pipeline.", flush=True)
        else:
            cmd = [
                sys.executable, "run_and_eval.py",
                "--video", path,
                "--backend", args.backend
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  Error processing {vid}: {e}")
                continue

    # Final Evaluation
    print("\n--- Running Final Evaluation ---")
    # We use a custom call to the evaluation script but only for these 62 videos
    # We can pass the batch dir if we have one, or just let it process all in results/
    # but the user wants results specifically for these 62.
    
    # We'll create a temporary list for the evaluator
    try:
        from scripts.eval_sgg_recall import evaluate_video, load_pvsg_ground_truth
        pvsg_json = 'datasets/PVSG/pvsg.json'
        gt_videos = load_pvsg_ground_truth(pvsg_json)
        
        all_metrics = []
        for vid in ALL_VIDEOS:
            if os.path.exists(os.path.join('results', vid, 'scene_graph.jsonl')):
                res = evaluate_video(vid, 'results', gt_videos)
                if 'error' not in res:
                    all_metrics.append(res)
        
        if all_metrics:
            # Aggregate Pooled Metrics (as per run_and_eval.py)
            import numpy as np
            from collections import defaultdict
            
            def get_cat_stats(k, metrics):
                cat_stats = defaultdict(lambda: [0, 0])
                for res in metrics:
                    for p, (m, t) in res['pred_stats'][k].items():
                        cat_stats[p][0] += m
                        cat_stats[p][1] += t
                return cat_stats

            def get_pooled_mr(k, metrics):
                stats = get_cat_stats(k, metrics)
                recalls = [m/t for m, t in stats.values() if t > 0]
                return np.mean(recalls) * 100.0 if recalls else 0.0

            mR20_avg = get_pooled_mr(20, all_metrics)
            mR50_avg = get_pooled_mr(50, all_metrics)
            mR100_avg = get_pooled_mr(100, all_metrics)
            
            R20_avg = np.mean([m['R@20'] for m in all_metrics])
            R50_avg = np.mean([m['R@50'] for m in all_metrics])
            R100_avg = np.mean([m['R@100'] for m in all_metrics])
            
            print("\n" + "="*80, flush=True)
            print(f"FINAL BATCH EVALUATION REPORT ({len(all_metrics)} VIDEOS PROCESSED)", flush=True)
            print("Note: All metrics are based on (Subject, Predicate, Object) Triplet Matching", flush=True)
            print("="*80, flush=True)
            print(f"{'Metric Type':<15} | {'Metric':<15} | {'Score (%)':>10}", flush=True)
            print("-" * 50, flush=True)
            print(f"{'Weighted Avg':<15} | {'Recall@20':<15} | {R20_avg:>10.2f}%", flush=True)
            print(f"{'Per-Predicate':<15} | {'mRecall@20':<15} | {mR20_avg:>10.2f}%", flush=True)
            print("-" * 50, flush=True)
            print(f"{'Weighted Avg':<15} | {'Recall@50':<15} | {R50_avg:>10.2f}%", flush=True)
            print(f"{'Per-Predicate':<15} | {'mRecall@50':<15} | {mR50_avg:>10.2f}%", flush=True)
            print("-" * 50, flush=True)
            print(f"{'Weighted Avg':<15} | {'Recall@100':<15} | {R100_avg:>10.2f}%", flush=True)
            print(f"{'Per-Predicate':<15} | {'mRecall@100':<15} | {mR100_avg:>10.2f}%", flush=True)
            print("="*80 + "\n", flush=True)

            # Per-predicate breakdown for mRecall (The "rm@k stuff")
            print("Per-Predicate Recall Breakdown (at @100):", flush=True)
            print(f"{'Predicate':<20} | {'Matched':>7} | {'Total GT':>8} | {'Recall':>9}", flush=True)
            print("-" * 52, flush=True)
            stats100 = get_cat_stats(100, all_metrics)
            for p in sorted(stats100.keys()):
                m, t = stats100[p]
                rec = (m/t)*100.0 if t > 0 else 0.0
                print(f"{p:<20} | {m:>7} | {t:>8} | {rec:>8.1f}%", flush=True)
            print("-" * 52 + "\n", flush=True)

            # Per-video breakdown
            print("Per-Video Summary:", flush=True)
            print(f"{'Video ID':<40} | {'R@20':>7} | {'mR@20':>7} | {'R@100':>7}", flush=True)
            print("-" * 70, flush=True)
            for m in all_metrics:
                print(f"{m['video_id']:<40} | {m['R@20']:>6.1f}% | {m['mR@20']:>6.1f}% | {m['R@100']:>6.1f}%", flush=True)
            print("-" * 70 + "\n", flush=True)
        else:
            print("No valid evaluation results found.")
            
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()
