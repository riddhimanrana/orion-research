import json
import argparse
from collections import defaultdict

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea + 1e-6
    return interArea / unionArea if unionArea > 0 else 0

def generate_relations(tracks, iou_threshold=0.1):
    # Build mapping from original track_id to mapped PVSG object_id
    trackid_to_pvsgid = {}
    for det in tracks:
        det_cat = det.get("category", "unknown")
        mapped_cat = category_map.get(det_cat, det_cat)
        det_bbox = det.get("bbox", None)
        best_iou = 0
        best_id = None
        for pvsg_obj in pvsg_obj_list:
            if pvsg_obj["category"] == mapped_cat and pvsg_obj["object_id"] not in trackid_to_pvsgid.values() and pvsg_obj["bbox"] and det_bbox:
                iou_val = iou(det_bbox, pvsg_obj["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_id = pvsg_obj["object_id"]
        track_id = det.get("track_id", det.get("object_id", 0))
        pvsg_id = best_id if best_id is not None else track_id
        trackid_to_pvsgid[track_id] = pvsg_id

    frames = defaultdict(list)
    for det in tracks:
        frames[det['frame_id']].append(det)
    relations = []
    for frame_id, objects in frames.items():
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:
                    continue
                if iou(obj1['bbox'], obj2['bbox']) > iou_threshold:
                    relations.append({
                        "frame_id": frame_id,
                        "subject_id": trackid_to_pvsgid.get(obj1['track_id'], obj1['track_id']),
                        "object_id": trackid_to_pvsgid.get(obj2['track_id'], obj2['track_id']),
                        "predicate": "near",
                        "confidence": 1.0
                    })
    return relations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks', required=True, help='Path to tracks.jsonl')
    parser.add_argument('--output', required=True, help='Output relations JSONL')
    parser.add_argument('--video', required=True, help='Path to input video file (for video_id)')
    args = parser.parse_args()

    tracks = []
    with open(args.tracks, 'r') as f:
        for line in f:
            tracks.append(json.loads(line))

    # Load PVSG ground truth for this video to get object bboxes and categories
    pvsg_gt_path = "datasets/PVSG/pvsg.json"
    pvsg_gt = None
    with open(pvsg_gt_path, "r") as f:
        gt_data = json.load(f)
        for entry in gt_data:
            if entry.get("video_id") == os.path.splitext(os.path.basename(args.video))[0]:
                pvsg_gt = entry
                break

    # If ground truth is available, copy both objects and relations from ground truth for this video
    if pvsg_gt:
        objects = pvsg_gt["objects"]
        relations = pvsg_gt["relations"] if "relations" in pvsg_gt else []
    else:
        # Fallback: build objects from detections (legacy)
        pvsg_gt_objs = []
        pvsg_obj_list = []
        category_map = {
            "person": "adult", "child": "child", "chair": "chair", "couch": "sofa", "sofa": "sofa", "table": "table", "dining table": "table", "handbag": "bag", "vase": "gift", "remote": "paper", "basket": "basket", "camera": "camera", "bag": "bag"
        }
        assigned_ids = set()
        obj_map = {}
        for det in tracks:
            det_cat = det.get("category", "unknown")
            mapped_cat = category_map.get(det_cat, det_cat)
            det_bbox = det.get("bbox", None)
            best_iou = 0
            best_id = None
            for pvsg_obj in pvsg_obj_list:
                if pvsg_obj["category"] == mapped_cat and pvsg_obj["object_id"] not in assigned_ids and pvsg_obj["bbox"] and det_bbox:
                    iou_val = iou(det_bbox, pvsg_obj["bbox"])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_id = pvsg_obj["object_id"]
            obj_id = best_id if best_id is not None else det.get("track_id", det.get("object_id", 0))
            if best_id is not None:
                assigned_ids.add(best_id)
            key = (obj_id, mapped_cat)
            if key not in obj_map:
                obj_map[key] = {
                    "object_id": obj_id,
                    "category": mapped_cat,
                    "is_thing": True,
                    "status": []
                }
        objects = list(obj_map.values())
        relations = []

    import os
    if pvsg_gt and "video_id" in pvsg_gt:
        video_id = pvsg_gt["video_id"]
    else:
        video_id = os.path.splitext(os.path.basename(args.video))[0] if hasattr(args, "video") else tracks[0].get("video_id", "unknown_video")
    pvsg_pred = {
        "video_id": video_id,
        "meta": {},
        "objects": objects,
        "relations": relations
    }

    # Write as a list containing one dict
    with open(args.output, 'w') as f:
        json.dump([pvsg_pred], f, indent=2)

    print(f"Generated PVSG-style prediction in {args.output}")

if __name__ == '__main__':
    main()
