
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# PVSG Relation Vocabulary (index 0 to 56)
PVSG_PREDICATES = [
    "beside", "biting", "blowing", "brushing", "caressing", "carrying", "catching", "chasing",
    "cleaning", "closing", "cooking", "cutting", "drinking from", "eating", "entering", "feeding",
    "grabbing", "guiding", "hanging from", "hitting", "holding", "hugging", "in", "in front of",
    "jumping from", "jumping over", "kicking", "kissing", "licking", "lighting", "looking at",
    "lying on", "next to", "on", "opening", "over", "picking", "playing", "playing with",
    "pointing to", "pulling", "pushing", "riding", "running on", "shaking hand with", "sitting on",
    "standing on", "stepping on", "stirring", "swinging", "talking to", "throwing", "touching",
    "toward", "walking on", "watering", "wearing"
]

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_depth(track: Dict[str, Any]) -> float:
    # Try different fields
    d = track.get("depth_mm", 0.0)
    if d > 0: return d
    b3d = track.get("bbox_3d")
    if b3d and len(b3d) >= 3:
        return b3d[2] # Z
    return 0.0

def get_centroid_3d(track: Dict[str, Any]) -> Tuple[float, float, float]:
    # If explicit centroid_3d_mm exists
    c = track.get("centroid_3d_mm")
    if c: return tuple(c)
    # Or from bbox_3d
    b3d = track.get("bbox_3d")
    if b3d and len(b3d) >= 3:
        return (b3d[0], b3d[1], b3d[2])
    # Fallback to 2D
    b2d = track.get("bbox")
    if b2d:
        cx, cy = (b2d[0]+b2d[2])/2, (b2d[1]+b2d[3])/2
        return (cx, cy, get_depth(track))
    return (0,0,0)

def iou(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union

def relation_classifier(sub: Dict, obj: Dict, frame_w: int, frame_h: int) -> List[Tuple[str, float]]:
    """Heuristic classifier for PVSG relations."""
    rels = []
    
    s_cls = sub.get("category", sub.get("class_name", "object")).lower()
    o_cls = obj.get("category", obj.get("class_name", "object")).lower()
    
    s_box = sub.get("bbox")
    o_box = obj.get("bbox")
    
    if not s_box or not o_box:
        return []

    # 3D info
    s_z = get_depth(sub)
    o_z = get_depth(obj)
    s_pos = get_centroid_3d(sub)
    o_pos = get_centroid_3d(obj)
    
    # Simple geometry
    dist_2d = math.hypot((s_box[0]+s_box[2])/2 - (o_box[0]+o_box[2])/2, 
                         (s_box[1]+s_box[3])/2 - (o_box[1]+o_box[3])/2)
    diag = math.hypot(frame_w, frame_h)
    norm_dist = dist_2d / diag
    
    depth_diff = abs(s_z - o_z)
    has_depth = (s_z > 0 and o_z > 0)
    
    # 1. SPATIAL (on, next to, etc)
    
    # "on" / "sitting on" / "standing on" / "lying on" / "riding"
    # Subject center Y < Object center Y (subject is higher, assuming Y down) NO wait Y is down.
    # Subject center Y < Object center y means subject is ABOVE object visually?
    # Actually "on" means subject max Y is near object min Y?
    # Or subject bottom is near object top. 
    # Y axis usually points DOWN in images. So Top is 0.
    # Subject ON Object -> Subject Bottom (Max Y) approx Object Top (Min Y).
    # Or Subject Bottom <= Object Bottom and overlapping horizontally.
    
    s_bottom = s_box[3]
    o_top = o_box[1]
    o_bottom = o_box[3]
    
    h_overlap = min(s_box[2], o_box[2]) - max(s_box[0], o_box[0])
    h_union = max(s_box[2], o_box[2]) - min(s_box[0], o_box[0])
    h_iou = h_overlap / h_union if h_union > 0 else 0
    
    is_vertically_aligned = h_iou > 0.3
    is_above = (s_bottom < o_bottom) and abs(s_bottom - o_top) < (frame_h * 0.05)
    
    # Check simple "on"
    if is_vertically_aligned and s_bottom <= o_bottom:
        # Refine "on"
        pred = "on"
        if s_cls == "person":
            if o_cls in ["chair", "couch", "sofa", "bench", "stool", "bed"]:
                pred = "sitting on"
            elif o_cls in ["ground", "floor", "grass", "road", "pavement"]:
                # Could be standing or walking or sitting
                # Use aspect ratio?
                w = s_box[2]-s_box[0]
                h = s_box[3]-s_box[1]
                if h < w: pred = "lying on"
                else: pred = "standing on"
            elif o_cls in ["horse", "bike", "bicycle", "motorcycle", "car", "bus"]:
                pred = "riding"
            elif o_cls in ["table", "desk"]:
                 # sitting on table? or just on
                 pred = "on"
            else:
                 pred = "standing on" # Default person on object
        
        rels.append((pred, 0.8))
        
        if pred == "sitting on":
             if o_cls == "bed": rels.append(("lying on", 0.6))

    # "holding" / "carrying"
    # If person and small object, and high overlap
    interaction_iou = iou(s_box, o_box)
    if s_cls == "person" and o_cls != "person" and interaction_iou > 0.1:
        # Check geometric containment or proximity
        # Simple heuristic: object center is within person box
        o_cx, o_cy = (o_box[0]+o_box[2])/2, (o_box[1]+o_box[3])/2
        if s_box[0] < o_cx < s_box[2] and s_box[1] < o_cy < s_box[3]:
             rels.append(("holding", 0.7))
             rels.append(("carrying", 0.6))
             rels.append(("touching", 0.9))

    # "next to" / "beside"
    if norm_dist < 0.15 and not is_vertically_aligned:
        # Check depth consistency
        if has_depth and depth_diff < 500: # 500mm tolerance
             rels.append(("next to", 0.8))
             rels.append(("beside", 0.8))
        elif not has_depth:
             rels.append(("next to", 0.6))

    # "in front of" / "behind" (PVSG only has "in front of")
    # 3D check: large overlap 2D, but Z differs
    if interaction_iou > 0.5 and has_depth:
        if s_z < o_z - 300: # Subject is closer (smaller Z)
             rels.append(("in front of", 0.8))
    
    # "looking at"
    # Cone of view check? 
    # Heuristic: person head (top 20%) -> object vector
    if s_cls == "person":
        # Estimate head position
        head_x = (s_box[0]+s_box[2])/2
        head_y = s_box[1] + (s_box[3]-s_box[1])*0.15
        
        vec_x = o_pos[0] - head_x
        vec_y = o_pos[1] - head_y
        # If object is somewhat in front (assuming person faces 'forward'?)
        # Without pose, assume objects in front/center are looked at
        rels.append(("looking at", 0.3)) # Low confidence default

    return rels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--out", default="pvsg_relations.jsonl")
    args = parser.parse_args()
    
    tracks = load_jsonl(Path(args.tracks))
    
    # Group by frame
    by_frame = {}
    for t in tracks:
        by_frame.setdefault(t["frame_id"], []).append(t)
    
    results = []
    
    # Assume fixed resolution for simplicity or read from track
    W, H = 1920, 1080 # default
    if tracks:
         # Try to guess from tracks? No field 'frame_width' in standard tracks.jsonl usually
         # but build_scene_graphs heuristic usually has it.
         pass

    print(f"Processing {len(by_frame)} frames...")
    
    count = 0
    for fid, frame_tracks in sorted(by_frame.items()):
        n = len(frame_tracks)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                
                sub = frame_tracks[i]
                obj = frame_tracks[j]
                
                # Predict
                preds = relation_classifier(sub, obj, W, H)
                
                for pred, conf in preds:
                    if pred not in PVSG_PREDICATES:
                        continue # Should not happen if heuristic matches list
                    
                    res = {
                        "video_id": "test_vidor", # Placeholder
                        "frame_id": fid,
                        "subject_id": sub["track_id"],
                        "object_id": obj["track_id"],
                        "subject_class": sub.get("class_name"),
                        "object_class": obj.get("class_name"),
                        "predicate": pred,
                        "confidence": conf,
                        "pvsg_id": PVSG_PREDICATES.index(pred)
                    }
                    results.append(res)
                    count += 1
    
    print(f"Generated {count} relations.")
    
    # Summary
    counts = {}
    for r in results:
        p = r["predicate"]
        counts[p] = counts.get(p, 0) + 1
    
    print("\nPredicate Counts:")
    for p, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {p}: {c}")
    
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
