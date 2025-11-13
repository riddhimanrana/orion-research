#!/usr/bin/env python3
"""
Simple detection demo with class corrections and FastVLM descriptions
"""

import cv2
import numpy as np
from pathlib import Path

# Class corrections for common YOLO misclassifications
CLASS_CORRECTIONS = {
    'musical keyboard': 'keyboard',  # Mechanical keyboard misclassified
    'laptop keyboard': 'keyboard',   # Part of mechanical keyboard  
    'desktop computer': 'laptop',    # Laptop misclassified
    'tv': 'monitor',                 # Monitor misclassified as TV
    'diaper bag': 'bag',             # Generic bag
}

IMPORTANCE_MAP = {
    'laptop': 5, 'keyboard': 5, 'mouse': 5, 'monitor': 5, 'phone': 5,
    'thermos': 4, 'bottle': 4, 'cup': 4, 'book': 4, 'bag': 4,
    'coaster': 3, 'desk mat': 3, 'lamp': 3, 'speaker': 3, 'cable': 3,
    'desk': 2, 'chair': 2, 'calendar': 2, 'bulletin board': 2,
    'wall': 1, 'floor': 1, 'carpet': 1, 'ceiling': 1,
}


def run_detection_with_fastvlm(video_path: str, frame_idx: int = 100):
    """Run detection with class corrections + FastVLM descriptions for top items"""
    from ultralytics import YOLO
    from orion.managers.model_manager import ModelManager
    
    print("\n" + "="*80)
    print("ENHANCED DETECTION: yoloe-11s-seg-pf + Class Corrections + FastVLM")
    print("="*80)
    
    # Load model
    model_path = "models/yoloe-11s-seg-pf.pt"
    if not Path(model_path).exists():
        model_path = "yolo11s-seg.pt"
    
    yolo = YOLO(model_path)
    print(f"\n✓ YOLO: {model_path}")
    
    # Load models
    manager = ModelManager.get_instance()
    clip = manager.clip
    fastvlm = manager.fastvlm
    print(f"✓ CLIP: ready for embeddings")
    print(f"✓ FastVLM: ready for descriptions")
    
    # Get frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Could not read frame {frame_idx}")
        return
    
    print(f"\n📹 Processing frame {frame_idx}...")
    
    # Run detection
    results = yolo(frame, conf=0.35, verbose=False)
    
    if not results or not results[0].boxes:
        print("  No detections")
        return
    
    boxes = results[0].boxes
    detections = []
    
    print(f"\n🔍 Detections ({len(boxes)}):")
    print("─"*80)
    
    for idx, (box, conf, cls_id) in enumerate(zip(
        boxes.xyxy.cpu().numpy(),
        boxes.conf.cpu().numpy(),
        boxes.cls.cpu().numpy()
    )):
        x1, y1, x2, y2 = map(int, box)
        original_class = yolo.names[int(cls_id)]
        
        # Apply corrections
        corrected_class = CLASS_CORRECTIONS.get(original_class, original_class)
        
        # Get CLIP embedding
        crop = frame[y1:y2, x1:x2]
        embedding = clip.encode_image(crop) if crop.size > 0 else None
        
        # Importance
        importance = IMPORTANCE_MAP.get(corrected_class, 3)
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'class': corrected_class,
            'original_class': original_class,
            'confidence': float(conf),
            'importance': importance,
            'embedding': embedding,
            'crop': crop
        })
        
        # Print
        correction_note = f" (was: {original_class})" if corrected_class != original_class else ""
        emoji = {5: "🔴", 4: "🟠", 3: "🟡", 2: "⚪", 1: "⚫"}.get(importance, "🟡")
        print(f"  {emoji} {idx:2d}. {corrected_class:20s} | conf={conf:.3f} | imp={importance}{correction_note}")
    
    # Sort by importance
    detections.sort(key=lambda x: (-x['importance'], -x['confidence']))
    
    # Use FastVLM for top 3 items to get better descriptions
    print(f"\n🤖 FastVLM Descriptions (top 3 high-importance items):")
    print("─"*80)
    
    described_count = 0
    for det in detections:
        if described_count >= 3:
            break
        if det['importance'] < 4:  # Only describe important items
            continue
            
        crop = det['crop']
        if crop.size == 0:
            continue
        
        # Get FastVLM description
        description = fastvlm.describe(crop, prompt="Describe this object in detail:")
        
        print(f"  • {det['class']}:")
        print(f"    \"{description.strip()}\"")
        print()
        
        det['fastvlm_description'] = description
        described_count += 1
    
    # Save annotated
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        imp = det['importance']
        
        # Color by importance
        color = (
            int(255 * (1 - imp/5)),  # Blue
            int(128 * imp/5),        # Green  
            int(255 * imp/5)          # Red
        )
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} ({imp})"
        cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    output_path = "results/detection_with_corrections.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"✓ Saved: {output_path}\n")
    
    return detections


if __name__ == "__main__":
    video_path = "data/examples/video.mp4"
    detections = run_detection_with_fastvlm(video_path, frame_idx=100)
    
    print("="*80)
    print("KEY INSIGHTS:")
    print("─"*80)
    print("1. Class corrections fix YOLO misclassifications (musical→keyboard, tv→monitor)")
    print("2. FastVLM provides rich descriptions for important items")
    print("3. Importance scoring (1-5) guides tracking priority")
    print("4. Next: Add depth for spatial zones, SLAM for persistent memory")
    print("="*80)
