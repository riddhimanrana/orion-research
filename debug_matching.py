import os
import json
import sys
sys.path.append(os.getcwd())
from scripts.eval_sgg_recall import normalize_class, normalize_predicate, load_orion_triplets, load_gt_triplets

video_id = "1017_12450142184"
with open('datasets/PVSG/pvsg.json', 'r') as f:
    gt_videos = {v['video_id']: v for v in json.load(f)['data']}

gt_triplets = load_gt_triplets(gt_videos[video_id])
print("\n--- GROUND TRUTH TRIPLETS ---")
for t in gt_triplets:
    print(t)

pred_triplets = load_orion_triplets(video_id, "results")
print(f"\n--- PREDICTED TRIPLETS (TOP 50) out of {len(pred_triplets)} ---")
for t in pred_triplets[:50]:
    print(t)

# Check for partial matches
print("\n--- CHECKING MATCHES ---")
gt_norm = set([(normalize_class(s), normalize_predicate(p), normalize_class(o)) for s, p, o in gt_triplets])
pred_norm = set([(normalize_class(s), normalize_predicate(p), normalize_class(o)) for s, p, o in pred_triplets])

print(f"Normalized GT Set Size: {len(gt_norm)}")
print(f"Normalized Pred Set Size: {len(pred_norm)}")

common = gt_norm.intersection(pred_norm)
print(f"Common Triplets: {common}")

if not common:
    print("\n--- WHY NO MATCH? ---")
    gt_subjs = set([t[0] for t in gt_norm])
    pred_subjs = set([t[0] for t in pred_norm])
    print(f"GT Subjects: {gt_subjs}")
    print(f"Pred Subjects: {pred_subjs}")
    print(f"Subject Intersection: {gt_subjs.intersection(pred_subjs)}")
    
    gt_preds = set([t[1] for t in gt_norm])
    pred_preds = set([t[1] for t in pred_norm])
    print(f"GT Predicates: {gt_preds}")
    print(f"Pred Predicates: {pred_preds}")
    print(f"Predicate Intersection: {gt_preds.intersection(pred_preds)}")

    gt_objs = set([t[2] for t in gt_norm])
    pred_objs = set([t[2] for t in pred_norm])
    print(f"GT Objects: {gt_objs}")
    print(f"Pred Objects: {pred_objs}")
    print(f"Object Intersection: {gt_objs.intersection(pred_objs)}")
