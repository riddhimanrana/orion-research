#!/usr/bin/env python3
"""
Test: Are CLIP embeddings actually different for different objects?
"""

import cv2
import numpy as np
from ultralytics import YOLO
from orion.managers.model_manager import ModelManager
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

video_path = "data/examples/video.mp4"
yolo = YOLO("models/yoloe-11s-seg-pf.pt")
manager = ModelManager.get_instance()
clip = manager.clip

# Get frame 100
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

# Detect
results = yolo(frame, conf=0.35, verbose=False)
boxes = results[0].boxes

print("\n" + "="*80)
print("CLIP EMBEDDING QUALITY TEST")
print("="*80)

# Get crops and embeddings
objects = []
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
    
    # Get embedding
    embedding = clip.encode_image(crop, normalize=True)
    
    objects.append({
        'class': class_name,
        'bbox': [x1, y1, x2, y2],
        'crop': crop,
        'embedding': embedding
    })

print(f"\nExtracted {len(objects)} objects")

# Build similarity matrix
embeddings_matrix = np.array([obj['embedding'] for obj in objects])
sim_matrix = cosine_similarity(embeddings_matrix)

print("\nSimilarity matrix:")
print("─"*80)
print("     ", end="")
for i in range(min(10, len(objects))):
    print(f"{i:5d}", end="")
print()

for i in range(min(10, len(objects))):
    print(f"{i:3d}: ", end="")
    for j in range(min(10, len(objects))):
        print(f"{sim_matrix[i][j]:5.2f}", end="")
    obj = objects[i]
    print(f"  | {obj['class']:20s} {obj['bbox']}")

# Check if embeddings are actually different
print("\n" + "="*80)
print("ANALYSIS:")
print("─"*80)

# Same class comparisons
print("\nObjects with SAME class:")
same_class_sims = []
for i in range(len(objects)):
    for j in range(i+1, len(objects)):
        if objects[i]['class'] == objects[j]['class']:
            sim = sim_matrix[i][j]
            same_class_sims.append(sim)
            print(f"  {objects[i]['class']:15s} ({i}) vs ({j}): {sim:.3f}")

# Different class comparisons
print("\nObjects with DIFFERENT class:")
diff_class_sims = []
for i in range(len(objects)):
    for j in range(i+1, len(objects)):
        if objects[i]['class'] != objects[j]['class']:
            sim = sim_matrix[i][j]
            diff_class_sims.append(sim)

if diff_class_sims:
    print(f"  Average similarity: {np.mean(diff_class_sims):.3f}")
    print(f"  Min: {np.min(diff_class_sims):.3f}, Max: {np.max(diff_class_sims):.3f}")

print("\n" + "="*80)
print("PROBLEM DIAGNOSIS:")
print("─"*80)

if same_class_sims and diff_class_sims:
    avg_same = np.mean(same_class_sims)
    avg_diff = np.mean(diff_class_sims)
    
    print(f"Same class avg:      {avg_same:.3f}")
    print(f"Different class avg: {avg_diff:.3f}")
    print(f"Separation:          {avg_same - avg_diff:.3f}")
    
    if avg_diff > 0.7:
        print("\n❌ CRITICAL PROBLEM: Different objects have >0.7 similarity!")
        print("   This means CLIP embeddings are NOT discriminative.")
        print("   Possible causes:")
        print("     1. Crops too small/low quality")
        print("     2. All crops look similar (same background)")
        print("     3. CLIP model not loaded correctly")
        print("     4. Need to use CLIP with text prompts for better discrimination")
    elif avg_same - avg_diff < 0.1:
        print("\n⚠️  WARNING: Same-class and different-class similarities too close!")
        print("   CLIP cannot reliably distinguish objects.")
    else:
        print("\n✓ CLIP embeddings are discriminative enough for Re-ID")

# Save some crops for visual inspection
print("\n" + "="*80)
print("Saving sample crops for visual inspection...")
for i, obj in enumerate(objects[:5]):
    cv2.imwrite(f"results/crop_{i}_{obj['class']}.jpg", obj['crop'])
print(f"✓ Saved to results/crop_*.jpg")
print("="*80)
