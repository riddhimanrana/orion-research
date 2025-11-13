#!/usr/bin/env python3
"""
FINAL RECOMMENDATION: Switch to yoloe-11s-seg-pf and implement multi-layer classification

This script demonstrates the improved pipeline architecture:
1. yoloe-11s-seg-pf for better workspace object detection
2. Depth-based spatial zones (when integrated)
3. FastVLM fallback for unknown objects
4. Importance scoring for tracking priority
"""

import cv2
import numpy as np
from pathlib import Path


def run_enhanced_detection(video_path: str, frame_idx: int = 100):
    """Demonstrate enhanced detection pipeline"""
    from ultralytics import YOLO
    from orion.managers.model_manager import ModelManager
    
    print("="*80)
    print("ENHANCED DETECTION PIPELINE")
    print("="*80)
    
    # Load improved model
    yoloe_path = "models/yoloe-11s-seg-pf.pt"
    if not Path(yoloe_path).exists():
        print(f"\n❌ yoloe-11s-seg-pf not found at {yoloe_path}")
        print("   Using yolo11s-seg fallback...")
        yoloe_path = "yolo11s-seg.pt"
    
    yolo = YOLO(yoloe_path)
    print(f"\n✓ Model loaded: {yoloe_path}")
    
    # Load CLIP for embeddings
    manager = ModelManager.get_instance()
    clip = manager.clip
    print(f"✓ CLIP loaded for embeddings")
    
    # Get frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Could not read frame {frame_idx}")
        return
    
    print(f"\n📹 Processing frame {frame_idx}...")
    
    # Run detection with tuned threshold
    conf_threshold = 0.35  # Balanced threshold
    results = yolo(frame, conf=conf_threshold, verbose=False)
    
    if len(results) == 0 or results[0].boxes is None:
        print("  No detections")
        return
    
    boxes = results[0].boxes
    masks = results[0].masks
    
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    class_names = [yolo.names[int(c)] for c in classes]
    
    print(f"\n🔍 Detections ({len(class_names)}):")
    print("─"*80)
    
    # Importance scoring (egocentric priority)
IMPORTANCE_MAP = {
    # Level 5: Critical personal workspace items
    'laptop': 5, 'notebook': 5, 'laptop keyboard': 5, 'keyboard': 5, 'mechanical keyboard': 5,
    'mouse': 5, 'trackpad': 5, 'monitor': 5, 'cell phone': 5, 'tablet': 5, 'smartwatch': 5,
    
    # Level 4: Frequently used items
    'thermos': 4, 'water bottle': 4, 'bottle': 4, 'cup': 4, 'coffee mug': 4,
    'notebook': 4, 'book': 4, 'pen': 4, 'pencil': 4, 'pencil case': 4,
    'backpack': 4, 'bag': 4, 'headphones': 4, 'earbuds': 4,
    
    # Level 3: Workspace accessories  
    'desk mat': 3, 'coaster': 3, 'wrist rest': 3, 'foot rest': 3,
    'lamp': 3, 'speaker': 3, 'picture frame': 3, 'plant': 3,
    'microphone': 3, 'webcam': 3, 'cable': 3, 'charger': 3,
    
    # Level 2: Furniture/static items
    'desk': 2, 'chair': 2, 'table': 2, 'shelf': 2,
    'calendar': 2, 'bulletin board': 2, 'whiteboard': 2, 'clock': 2,
    
    # Level 1: Background
    'wall': 1, 'floor': 1, 'carpet': 1, 'ceiling': 1, 'door': 1, 'window': 1,
}

# Class corrections for common YOLO misclassifications
CLASS_CORRECTIONS = {
    'musical keyboard': 'keyboard',  # Mechanical keyboard misclassified
    'laptop keyboard': 'keyboard',   # Part of mechanical keyboard
    'desktop computer': 'laptop',    # Laptop misclassified
    'tv': 'monitor',                 # Monitor misclassified as TV
    'diaper bag': 'bag',             # Generic bag
}    detections_with_priority = []
    
    for idx, (box, conf, cls_name) in enumerate(zip(xyxy, confs, class_names)):
        x1, y1, x2, y2 = map(int, box)
        
        # Get CLIP embedding
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        embedding = clip.encode_image(crop)
        
        # Calculate importance score
        base_importance = IMPORTANCE_MAP.get(cls_name.lower(), 3)  # Default medium
        confidence_bonus = min(conf - 0.5, 0.3) * 5  # Up to +1.5 for high confidence
        importance = base_importance + confidence_bonus
        
        # Get mask coverage (how much of bbox is actual object vs background)
        mask_coverage = 1.0
        if masks is not None and idx < len(masks):
            mask = masks[idx].data.cpu().numpy()[0]
            mask_pixels = np.sum(mask > 0.5)
            bbox_area = (x2 - x1) * (y2 - y1)
            mask_coverage = mask_pixels / bbox_area if bbox_area > 0 else 0
        
        detections_with_priority.append({
            'bbox': [x1, y1, x2, y2],
            'class_name': corrected_name,  # Use corrected name
            'original_class': class_name,   # Keep original for debugging
            'confidence': conf,
            'importance': importance,
            'base_importance': base_importance,
            'area': (x2 - x1) * (y2 - y1),
            'embedding': embeddings[len(detections_with_priority)] if embeddings else None
        })
        
        # Visual indicator
        priority_emoji = {5: "🔴", 4: "🟠", 3: "🟡", 2: "⚪", 1: "⚫"}
        emoji = priority_emoji.get(base_importance, "🟡")
        
        print(f"  {emoji} {idx:2d}. {cls_name:20s} | conf={conf:.3f} | importance={importance:.1f} | coverage={mask_coverage:.2f}")
    
    # Sort by importance (track high-priority items more carefully)
    detections_with_priority.sort(key=lambda x: x['importance'], reverse=True)
    
    print(f"\n📊 Priority Ranking (for tracking):")
    print("─"*80)
    for i, det in enumerate(detections_with_priority[:10], 1):
        print(f"  {i:2d}. {det['class']:20s} (importance: {det['importance']:.1f})")
    
    print(f"\n💡 Next Steps:")
    print("   1. Use yoloe-11s-seg-pf model ✓")
    print("   2. Prioritize tracking high-importance objects (laptop, keyboard, phone)")
    print("   3. Add depth for spatial zones (desk vs shelf vs floor)")
    print("   4. Use FastVLM for unknown objects (importance=3 by default)")
    print("   5. Implement SLAM to remember object locations across scenes")
    
    # Save annotated frame
    output_path = "results/enhanced_detection_demo.jpg"
    annotated = frame.copy()
    
    for det in detections_with_priority:
        x1, y1, x2, y2 = map(int, det['bbox'])
        
        # Color by importance (red=high, blue=low)
        importance_ratio = (det['importance'] - 1) / 4  # 0-1 scale
        color = (
            int(255 * (1 - importance_ratio)),  # Blue
            int(128 * importance_ratio),        # Green
            int(255 * importance_ratio),         # Red
        )
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        label = f"{det['class']} ({det['importance']:.1f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
        cv2.putText(
            annotated, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    
    cv2.imwrite(output_path, annotated)
    print(f"\n✓ Saved annotated frame: {output_path}")
    
    return detections_with_priority


if __name__ == "__main__":
    video_path = "data/examples/video.mp4"
    detections = run_enhanced_detection(video_path, frame_idx=100)
    
    print("\n" + "="*80)
    print("SUMMARY: Why yoloe-11s-seg-pf + Importance Scoring Works")
    print("="*80)
    print("""
yoloe-11s-seg-pf advantages:
  ✓ More specific classes (monitor vs tv, thermos vs bottle)
  ✓ Workspace-optimized (laptop keyboard, desktop computer, etc.)
  ✓ Better for egocentric scenes (desk items, personal objects)

Importance scoring enables:
  ✓ Priority-based tracking (focus on laptop, phone, not walls)
  ✓ Efficient Re-ID (high-importance tracks get more appearance history)
  ✓ Better memory allocation (don't waste embeddings on background)

Missing piece: Non-COCO objects
  → Use FastVLM for undetected regions
  → Build custom embedding classifier
  → Spatial context from SLAM (if near laptop → likely accessory)
    
The pipeline becomes:
  1. yoloe-11s-seg-pf: Detect known objects with importance
  2. Depth + SLAM: Understand spatial layout (zones)
  3. FastVLM: Describe unknown but important regions
  4. EnhancedTracker: Track high-priority objects persistently
  5. Re-ID Gallery: Recognize objects across scenes
""")
