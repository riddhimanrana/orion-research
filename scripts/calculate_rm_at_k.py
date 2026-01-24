import json
import argparse
from collections import defaultdict

# Usage: python calculate_rm_at_k.py --gt orion-core-fs/datasets/PVSG/pvsg.json --pred results/0020_5323209509/tracks.jsonl --k 20 50 100

def load_ground_truth(gt_path):
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    # If the file is a dict with 'data' key, use that, else assume it's a list
    if isinstance(gt, dict) and 'data' in gt:
        return gt['data']
    return gt

def load_predictions(pred_path):
    with open(pred_path, 'r') as f:
        data = json.load(f)
    # If the file is a dict with 'data' key, use that, else assume it's a list
    if isinstance(data, dict) and 'data' in data:
        return data['data']
    return data

def compute_rm_at_k(gt, preds, k):
    # Extract all relations from all videos in gt and preds
    def extract_relations_by_vid(data):
        vid2rels = {}
        for vid in data:
            if not isinstance(vid, dict):
                continue
            vid_id = vid.get('video_id', None)
            if not vid_id or 'relations' not in vid:
                continue
            rels = []
            for rel_list in vid['relations']:
                for rel in rel_list:
                    if isinstance(rel, dict):
                        rels.append(json.dumps(rel, sort_keys=True))
                    else:
                        rels.append(json.dumps(rel))
            vid2rels[vid_id] = rels
        return vid2rels

    gt_vid2rels = extract_relations_by_vid(gt)
    pred_vid2rels = extract_relations_by_vid(preds)

    recalls = {}
    for vid_id, pred_rels in pred_vid2rels.items():
        gt_rels = gt_vid2rels.get(vid_id, [])
        gt_set = set(gt_rels)
        pred_set = set(pred_rels[:k])
        recall = len(gt_set & pred_set) / max(1, len(gt_set))
        recalls[vid_id] = recall
    return recalls

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, help='Path to ground truth JSON')
    parser.add_argument('--pred', required=True, help='Path to predictions JSONL')
    parser.add_argument('--k', nargs='+', type=int, default=[20, 50, 100], help='Values of k for rm@k')
    args = parser.parse_args()

    gt = load_ground_truth(args.gt)
    preds = load_predictions(args.pred)

    # Print video_ids for debugging
    gt_vids = set()
    for vid in gt:
        if isinstance(vid, dict) and 'video_id' in vid:
            gt_vids.add(vid['video_id'])
    pred_vids = set()
    for vid in preds:
        if isinstance(vid, dict) and 'video_id' in vid:
            pred_vids.add(vid['video_id'])
    print(f"Ground truth video_ids: {sorted(gt_vids)}")
    print(f"Prediction video_ids: {sorted(pred_vids)}")
    missing = pred_vids - gt_vids
    if missing:
        print(f"Warning: These prediction video_ids are not found in ground truth: {sorted(missing)}")
    extra = gt_vids - pred_vids
    if extra:
        print(f"Note: These ground truth video_ids are not present in predictions: {sorted(extra)}")

    for k in args.k:
        recalls = compute_rm_at_k(gt, preds, k)
        if not recalls:
            print(f'rm@{k}: No matching video_ids found in predictions.')
            continue
        for vid_id, recall in recalls.items():
            print(f'video {vid_id} rm@{k}: {recall:.4f}')
        avg_recall = sum(recalls.values()) / len(recalls)
        print(f'Average rm@{k}: {avg_recall:.4f}')

if __name__ == '__main__':
    main()
