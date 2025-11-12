"""Scan entire room.mp4 to find frames with objects"""
import cv2
import numpy as np
from ultralytics import YOLO

print("Scanning room.mp4 for frames with detectable content...")
print("="*60)

model = YOLO('yolo11m.pt')
cap = cv2.VideoCapture('data/examples/room.mp4')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frames_with_detections = []
sharpness_scores = []

# Sample every 30 frames
sample_interval = 30
frame_idx = 0

while frame_idx < min(total_frames, 600):  # Check first 600 frames (~20 seconds)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break
    
    # Check sharpness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_scores.append((frame_idx, laplacian_var))
    
    # Run YOLO
    results = model(frame, conf=0.2, verbose=False)
    detections = results[0].boxes
    
    if len(detections) > 0 or laplacian_var > 50:  # High sharpness or detections
        frames_with_detections.append({
            'frame': frame_idx,
            'detections': len(detections),
            'sharpness': laplacian_var,
            'objects': [model.names[int(box.cls[0])] for box in detections]
        })
        
        if len(detections) > 0:
            print(f"Frame {frame_idx:4d}: {len(detections)} objects, sharpness={laplacian_var:.1f}")
            for box in detections[:3]:
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                print(f"  - {cls_name}: {conf:.3f}")
    
    frame_idx += sample_interval

cap.release()

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"Total frames scanned: {len(sharpness_scores)}")
print(f"Frames with detections: {len([f for f in frames_with_detections if f['detections'] > 0])}")
print(f"Frames with high sharpness (>50): {len([f for f in frames_with_detections if f['sharpness'] > 50])}")

# Find best frame
if frames_with_detections:
    best_frame = max(frames_with_detections, key=lambda x: x['detections'] * x['sharpness'])
    print(f"\nBest frame: {best_frame['frame']}")
    print(f"  Detections: {best_frame['detections']}")
    print(f"  Sharpness: {best_frame['sharpness']:.1f}")
    print(f"  Objects: {', '.join(best_frame['objects'])}")
else:
    print("\n❌ NO FRAMES WITH OBJECTS FOUND!")
    print("This video appears to be entirely walls/ceiling/floor.")
    
# Show sharpness distribution
sharpness_values = [s[1] for s in sharpness_scores]
print(f"\nSharpness stats:")
print(f"  Mean: {np.mean(sharpness_values):.1f}")
print(f"  Max: {np.max(sharpness_values):.1f} at frame {sharpness_scores[np.argmax(sharpness_values)][0]}")
print(f"  Min: {np.min(sharpness_values):.1f}")
