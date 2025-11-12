#!/usr/bin/env python3
"""
Generate rerun visualization montage showing progression across entire video
Displays representative frames from beginning, middle, and end of sequence
"""
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("/Users/riddhiman.rana/Desktop/Coding/Orion/orion-research/spatial_mapping_output")
MONTAGE_DIR = Path("/Users/riddhiman.rana/Desktop/Coding/Orion/orion-research/montage_visualization")
MONTAGE_DIR.mkdir(exist_ok=True)

print("\n" + "="*80)
print("🎬 GENERATING RERUN VISUALIZATION MONTAGE")
print("="*80)

# Load all images grouped by frame
frames_data = defaultdict(dict)
for img_path in sorted(OUTPUT_DIR.glob("*.png")):
    parts = img_path.stem.split("_")
    category = f"{parts[0]}_{parts[1]}"  # e.g., "00_intrinsics"
    frame_num = int(parts[-1])
    frames_data[frame_num][category] = img_path

total_frames = max(frames_data.keys()) + 1
print(f"\n📊 Total frames: {total_frames}")
print(f"Frame range: 0 to {total_frames-1}")

# Select key frames: start, middle, end
key_frame_indices = [1, total_frames // 3, total_frames // 2, 2 * total_frames // 3, total_frames - 1]
print(f"\n🎯 Creating montages for key frames:")
for idx in key_frame_indices:
    print(f"   Frame {idx}")

# Create montage for each key frame
for frame_idx in key_frame_indices:
    if frame_idx not in frames_data:
        continue
    
    frame_data = frames_data[frame_idx]
    
    # Load all 6 visualizations for this frame
    images = {}
    titles = [
        "Camera Intrinsics",
        "Depth Heatmap (REAL)",
        "YOLO Detections",
        "Depth Distribution",
        "3D Point Cloud",
        "2D Reprojection"
    ]
    
    categories = [
        "00_intrinsics",
        "01_depth",
        "02_yolo",
        "03_depth",
        "04_point",
        "05_reprojection"
    ]
    
    for cat, title in zip(categories, titles):
        if cat in frame_data:
            try:
                img = cv2.imread(str(frame_data[cat]))
                if img is not None:
                    # Resize to consistent height (300px)
                    h, w = img.shape[:2]
                    new_h = 300
                    new_w = int(w * new_h / h)
                    img = cv2.resize(img, (new_w, new_h))
                    images[cat] = (img, title)
            except Exception as e:
                print(f"   Error loading {cat}: {e}")
    
    if len(images) < 3:
        print(f"   ⚠️  Frame {frame_idx}: Not enough images, skipping")
        continue
    
    # Arrange in 2x3 grid
    row1 = []
    row2 = []
    
    for i, cat in enumerate(categories):
        if cat in images:
            img, title = images[cat]
            
            # Add title text
            img_with_title = np.ones((img.shape[0] + 40, img.shape[1], 3), dtype=np.uint8) * 255
            img_with_title[40:, :] = img
            cv2.putText(img_with_title, title, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            if i < 3:
                row1.append(img_with_title)
            else:
                row2.append(img_with_title)
    
    # Pad rows to same length if needed
    if len(row1) > 0 and len(row2) > 0:
        # Concatenate rows
        max_w_row1 = sum(img.shape[1] for img in row1)
        max_w_row2 = sum(img.shape[1] for img in row2) if row2 else 0
        max_w = max(max_w_row1, max_w_row2)
        
        row1_concat = np.hstack(row1)
        if len(row2) > 0:
            row2_concat = np.hstack(row2)
            # Pad shorter row
            if row2_concat.shape[1] < row1_concat.shape[1]:
                pad_width = row1_concat.shape[1] - row2_concat.shape[1]
                row2_concat = np.pad(row2_concat, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=255)
            elif row1_concat.shape[1] < row2_concat.shape[1]:
                pad_width = row2_concat.shape[1] - row1_concat.shape[1]
                row1_concat = np.pad(row1_concat, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=255)
            
            montage = np.vstack([row1_concat, row2_concat])
        else:
            montage = row1_concat
        
        # Add header
        header = np.ones((50, montage.shape[1], 3), dtype=np.uint8) * 240
        cv2.putText(header, f"Frame #{frame_idx} - Spatial Mapping Visualization", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        final_montage = np.vstack([header, montage])
        
        # Save
        output_path = MONTAGE_DIR / f"montage_frame_{frame_idx:03d}.png"
        cv2.imwrite(str(output_path), final_montage)
        print(f"   ✓ Created: {output_path.name}")

print("\n" + "="*80)
print(f"🎉 Montage generation complete!")
print(f"📁 Output: {MONTAGE_DIR}/")
print("="*80 + "\n")
