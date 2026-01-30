import json
import os
import sys

# Import normalization logic
from scripts.eval_sgg_recall import normalize_class, normalize_predicate, load_orion_triplets, load_gt_triplets

def main():
    video_id = "1000_8770650748"
    gt_path = "datasets/PVSG/pvsg.json"
    results_dir = "results"
    
    # Load GT
    with open(gt_path, 'r') as f:
        data = json.load(f)
    
    gt_video = next((v for v in data['data'] if v['video_id'] == video_id), None)
    if not gt_video:
        print(f"Video {video_id} not found in GT")
        return

    print(f"=== GT Analysis for {video_id} ===")
    objects = {obj['object_id']: obj['category'] for obj in gt_video['objects']}
    print(f"GT Objects: {len(objects)}")
    # print(objects)
    
    gt_triplets = load_gt_triplets(gt_video)
    print(f"GT Triplets (Normalized): {len(gt_triplets)}")
    for t in gt_triplets[:10]:
        print(f"  {t}")
        
    print("\n=== Prediction Analysis ===")
    pred_triplets = load_orion_triplets(video_id, results_dir)
    print(f"Pred Triplets (Normalized): {len(pred_triplets)}")
    for t in pred_triplets[:10]:
        print(f"  {t}")
        
    print("\n=== Intersection ===")
    gt_set = set(gt_triplets)
    pred_set = set(pred_triplets)
    intersection = gt_set & pred_set
    print(f"Matches: {len(intersection)}")
    for t in intersection:
        print(f"  MATCH: {t}")
        
    # Check specific mismatches
    print("\n=== Potential Mismatches (Preds not in GT) ===")
    count = 0
    for t in pred_triplets:
        if t not in gt_set:
            if count < 10:
                print(f"  MISS: {t}")
            count += 1
            
if __name__ == "__main__":
    main()
