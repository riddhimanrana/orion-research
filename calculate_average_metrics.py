import json
import os
import sys

# Ensure local imports work
sys.path.append(os.getcwd())
from scripts.eval_sgg_recall import evaluate_video, load_pvsg_ground_truth

VIDEOS = [
    "0001_4164158586",
    "0003_3396832512",
    "0004_11566980553",
    "0005_2505076295",
    "0006_2889117240",
    "0008_6225185844",
    "0010_8610561401",
    "0018_3057666738",
    "0024_5224805531",
    "0028_4021064662"
]

metrics = ["mR@20", "mR@50", "mR@100"]
totals = {m: 0.0 for m in metrics}
count = 0

print(f"{'Video ID':<20} | {'mR@20':<8} | {'mR@50':<8} | {'mR@100':<8}")
print("-" * 55)

# Load GT once
gt_path = os.path.abspath("datasets/PVSG/pvsg.json")
if not os.path.exists(gt_path):
    print(f"Error: GT file not found at {gt_path}")
    sys.exit(1)

gt_videos = load_pvsg_ground_truth(gt_path)

results_dir = os.path.abspath("results")

for vid in VIDEOS:
    res = evaluate_video(vid, results_dir, gt_videos)
    
    if "error" not in res:
        print(f"{vid:<20} | {res['mR@20']:>8.2f} | {res['mR@50']:>8.2f} | {res['mR@100']:>8.2f}")
        for m in metrics:
            totals[m] += res[m]
        count += 1
    else:
        print(f"{vid:<20} | ERROR: {res['error']}")

print("-" * 55)
if count > 0:
    print(f"{'AVERAGE':<20} | {totals['mR@20']/count:>8.2f} | {totals['mR@50']/count:>8.2f} | {totals['mR@100']/count:>8.2f}")
else:
    print("No valid results found.")
