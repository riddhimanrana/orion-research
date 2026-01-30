import json
import numpy as np
from pathlib import Path

def dist(b1, b2):
    c1 = [(b1[0]+b1[2])/2, (b1[1]+b1[3])/2]
    c2 = [(b2[0]+b2[2])/2, (b2[1]+b2[3])/2]
    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter)

tracks_path = 'results/pvsg_final/tracks.jsonl'
lines = [json.loads(l) for l in open(tracks_path) if l.strip()]

# Track 0 (person), Track 8 (ball)
p_tracks = {t['frame_id']: t['bbox'] for t in lines if t['track_id'] == 0}
b_tracks = {t['frame_id']: t['bbox'] for t in lines if t['track_id'] == 8}

print(f"{'Frame':<10} {'Distance':<10} {'IoU':<10}")
for f in range(30):
    if f in p_tracks and f in b_tracks:
        d = dist(p_tracks[f], b_tracks[f])
        i = iou(p_tracks[f], b_tracks[f])
        print(f"{f:<10} {d:<10.2f} {i:<10.3f}")
