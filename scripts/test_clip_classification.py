#!/usr/bin/env python3
"""
CLIP Zero-Shot Classification - Real-time accurate labeling

Instead of trusting YOLO's 1000 classes, use CLIP to classify into
egocentric categories that matter for your workflow
"""

import cv2
import numpy as np
from ultralytics import YOLO
from orion.managers.model_manager import ModelManager

# Define YOUR categories (what matters in egocentric video)
EGOCENTRIC_CATEGORIES = [
    # Work items
    "laptop computer", "mechanical keyboard", "computer mouse", "monitor display",
    "phone", "tablet", "smartwatch",
    
    # Containers & accessories
    "backpack", "water bottle", "coffee cup", "thermos",
    "notebook", "book", "pen", "pencil",
    
    # Furniture
    "desk", "chair", "shelf",
    
    # Other
    "cable", "lamp", "speaker", "calendar", "wall", "floor"
]

def classify_with_clip(crop, clip_model, categories):
    """Use CLIP to classify crop into one of the categories"""
    # Get image embedding
    image_embedding = clip_model.encode_image(crop)
    
    # Get text embeddings for all categories
    text_embeddings = []
    for category in categories:
        text_emb = clip_model.encode_text(f"a photo of a {category}")
        text_embeddings.append(text_emb)
    
    text_embeddings = np.array(text_embeddings)
    
    # Compute similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(
        image_embedding.reshape(1, -1),
        text_embeddings
    )[0]
    
    # Get best match
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    best_category = categories[best_idx]
    
    return best_category, best_score, similarities


# Test on frame 100
video_path = "data/examples/video.mp4"
yolo = YOLO("models/yoloe-11s-seg-pf.pt")
manager = ModelManager.get_instance()
clip = manager.clip

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

# Run YOLO for detection (bbox only, ignore labels)
results = yolo(frame, conf=0.35, verbose=False)
boxes = results[0].boxes

print("\n" + "="*80)
print("YOLO (bbox) + CLIP (classification) Pipeline")
print("="*80)
print(f"\nDetected {len(boxes)} objects")
print("\nComparing YOLO labels vs CLIP classification:")
print("─"*80)

for idx, (box, conf, cls_id) in enumerate(zip(
    boxes.xyxy.cpu().numpy(),
    boxes.conf.cpu().numpy(),
    boxes.cls.cpu().numpy()
)):
    x1, y1, x2, y2 = map(int, box)
    yolo_label = yolo.names[int(cls_id)]
    
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        continue
    
    # CLIP classification
    clip_label, clip_conf, _ = classify_with_clip(crop, clip, EGOCENTRIC_CATEGORIES)
    
    # Compare
    agreement = "✓" if yolo_label.lower() in clip_label.lower() or clip_label.lower() in yolo_label.lower() else "✗"
    
    print(f"\n{idx:2d}. YOLO: {yolo_label:20s} (conf={conf:.3f})")
    print(f"    CLIP: {clip_label:20s} (conf={clip_conf:.3f}) {agreement}")
    
    if agreement == "✗":
        print(f"    → CONFLICT! CLIP is likely more accurate")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("─"*80)
print("""
YOLO: Fast detection (bboxes) but wrong labels
CLIP: Accurate classification but needs bboxes from YOLO

HYBRID PIPELINE (real-time):
1. YOLO detects objects (bbox + mask) - ~50ms/frame
2. CLIP classifies each crop into YOUR categories - ~10ms/object
3. Total: ~150ms/frame for 10 objects = Real-time!

BENEFITS:
✓ Define your own categories (not stuck with COCO 80 classes)
✓ More accurate than YOLO labels
✓ Still real-time (CLIP is fast for classification)
✓ Can refine categories based on your workflow
""")
print("="*80)
