#!/usr/bin/env python3
"""
Debug Re-ID failures - why are we creating duplicate tracks?
"""

import cv2
import numpy as np
from ultralytics import YOLO
from orion.perception.enhanced_tracker import EnhancedTracker
from orion.managers.model_manager import ModelManager
from sklearn.metrics.pairwise import cosine_similarity


def compute_iou(box1, box2):
    """Helper to compute IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


video_path = "data/examples/video.mp4"
yolo = YOLO("models/yoloe-11s-seg-pf.pt")

# Initialize tracker with CLIP
manager = ModelManager.get_instance()
tracker = EnhancedTracker(
    max_age=30,
    min_hits=2,
    appearance_threshold=0.5,  # Lower threshold for testing
    iou_threshold=0.3,
    clip_model=manager.clip
)

print("\n" + "="*80)
print("RE-ID DEBUGGING: Why are we creating duplicate tracks?")
print("="*80)

# Process just frames 100-102 to see what's happening
cap = cv2.VideoCapture(video_path)
frames_to_process = [100, 101, 102]

all_embeddings = []  # Store embeddings to analyze similarity

for frame_idx in frames_to_process:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO detection
    results = yolo(frame, conf=0.35, verbose=False)
    boxes = results[0].boxes
    
    # Prepare detections
    detections = []
    embeddings = []
    
    for box, conf, cls_id in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.conf.cpu().numpy(),
        boxes.cls.cpu().numpy()
    ):
        x1, y1, x2, y2 = map(int, box)
        class_name = yolo.names[int(cls_id)]
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        # Get CLIP embedding
        embedding = manager.clip.encode_image(crop)
        
        det = {
            'bbox_2d': np.array([x1, y1, x2, y2]),
            'bbox_3d': np.array([x1, y1, 0, x2-x1, y2-y1, 1]),
            'class_name': class_name,
            'confidence': float(conf),
            'depth_mm': 500.0,
        }
        
        detections.append(det)
        embeddings.append(embedding)
    
    print(f"\n{'='*80}")
    print(f"FRAME {frame_idx}: {len(detections)} detections")
    print("─"*80)
    
    # Before tracking update, analyze what we have
    if frame_idx == 100:
        print("\nFirst frame - will create all tracks")
        all_embeddings = embeddings.copy()
    else:
        print(f"\nExisting tracks: {len([t for t in tracker.tracks if t.time_since_update < 5])}")
        
        # Check similarity between current detections and existing tracks
        print("\nEmbedding similarity analysis:")
        print("─"*60)
        
        for i, (det, emb) in enumerate(zip(detections, embeddings)):
            x1, y1, x2, y2 = det['bbox_2d']
            print(f"\nDetection {i}: {det['class_name']:20s} @ ({x1:4.0f}, {y1:4.0f})")
            
            # Compare with existing tracks
            if len(tracker.tracks) > 0:
                print("  Similarity to existing tracks:")
                for track in tracker.tracks[:10]:  # Top 10 tracks
                    if track.avg_appearance is not None:
                        sim = cosine_similarity(
                            emb.reshape(1, -1),
                            track.avg_appearance.reshape(1, -1)
                        )[0][0]
                        
                        # Check IoU too
                        tx1, ty1, tx2, ty2 = track.bbox_2d
                        iou = compute_iou([x1,y1,x2,y2], [tx1,ty1,tx2,ty2])
                        
                        if sim > 0.4 or iou > 0.1:  # Show if reasonable match
                            print(f"    Track {track.id:2d} ({track.class_name:15s}): sim={sim:.3f}, iou={iou:.3f}")
    
    # Update tracker
    tracks = tracker.update(detections, embeddings, frame_idx=frame_idx)
    
    print(f"\nAfter update: {len(tracker.tracks)} total tracks, {len(tracks)} confirmed")
    for track in tracks[:10]:
        print(f"  Track {track.id:2d}: {track.class_name:20s} (hits={track.hits}, age={track.age})")

cap.release()

# Analyze final embeddings
print("\n" + "="*80)
print("EMBEDDING SIMILARITY MATRIX (confirmed tracks)")
print("─"*80)

confirmed = [t for t in tracker.tracks if t.hits >= 2 and t.avg_appearance is not None]
print(f"\n{len(confirmed)} confirmed tracks with embeddings")

if len(confirmed) > 1:
    # Build similarity matrix
    embeddings_matrix = np.array([t.avg_appearance for t in confirmed])
    sim_matrix = cosine_similarity(embeddings_matrix)
    
    print("\nTracks that are suspiciously similar (>0.7 similarity):")
    print("(These should probably be the same track!)")
    print("─"*60)
    
    for i in range(len(confirmed)):
        for j in range(i+1, len(confirmed)):
            sim = sim_matrix[i][j]
            if sim > 0.7:
                t1 = confirmed[i]
                t2 = confirmed[j]
                print(f"Track {t1.id:2d} ({t1.class_name:15s}) <-> Track {t2.id:2d} ({t2.class_name:15s}): {sim:.3f}")
                print(f"  Bbox1: {t1.bbox_2d}, Bbox2: {t2.bbox_2d}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("─"*80)
print("""
If we see high-similarity tracks (>0.7), Re-ID is failing because:
  1. IoU threshold too strict (objects moved between frames)
  2. Appearance threshold too strict (not recognizing same object)
  3. Cost matrix computation broken
  4. Hungarian matching not finding correct assignments

Next: Check the matching algorithm in EnhancedTracker
""")
print("="*80)
