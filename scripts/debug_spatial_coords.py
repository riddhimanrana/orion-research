#!/usr/bin/env python3
"""
Debug: Why is spatial Re-ID creating more tracks?
Check if the problem is coordinate transformation.
"""

import cv2
import numpy as np
from ultralytics import YOLO

video_path = "data/examples/video.mp4"
yolo = YOLO("models/yoloe-11s-seg-pf.pt")

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

# Track monitor positions across frames
monitor_positions = []

for frame_idx in range(100, 105):  # Just 5 frames
    ret, frame = cap.read()
    if not ret:
        break
    
    results = yolo(frame, conf=0.35, verbose=False)
    boxes = results[0].boxes
    
    frame_monitors = []
    for box, cls_id in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        class_name = yolo.names[int(cls_id)]
        
        if 'monitor' in class_name.lower():
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            frame_monitors.append({
                'bbox': [x1, y1, x2, y2],
                'center': [center_x, center_y],
                'class': class_name
            })
    
    monitor_positions.append(frame_monitors)
    print(f"\nFrame {frame_idx}: {len(frame_monitors)} monitors detected")
    for i, m in enumerate(frame_monitors):
        print(f"  Monitor {i}: center=({m['center'][0]:.0f}, {m['center'][1]:.0f})  "
              f"bbox={m['bbox']}")

cap.release()

print("\n" + "="*80)
print("CROSS-FRAME ANALYSIS")
print("="*80)

# Check if monitors at similar positions across frames
if len(monitor_positions) >= 2:
    frame0_monitors = monitor_positions[0]
    frame1_monitors = monitor_positions[1]
    
    print(f"\nComparing Frame 100 ({len(frame0_monitors)} monitors) vs Frame 101 ({len(frame1_monitors)} monitors):")
    
    for i, m0 in enumerate(frame0_monitors):
        print(f"\n  Frame 100 Monitor {i} at ({m0['center'][0]:.0f}, {m0['center'][1]:.0f}):")
        
        for j, m1 in enumerate(frame1_monitors):
            dist = np.linalg.norm(
                np.array(m0['center']) - np.array(m1['center'])
            )
            print(f"    vs Frame 101 Monitor {j}: distance = {dist:.1f}px")
            
            if dist < 50:  # Should match if <50px movement
                print(f"      → SHOULD MATCH! (distance < 50px)")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("─"*80)
print("If monitors that barely moved (< 50px) are NOT matching,")
print("then the spatial Re-ID threshold (200mm) is too strict OR")
print("the coordinate transformation is wrong (using pixels as mm).")
print("="*80)
