#!/usr/bin/env python3
"""
Visualize the crops to understand why CLIP fails
"""

import cv2
import matplotlib.pyplot as plt
from glob import glob

crops = sorted(glob("results/crop_*.jpg"))

fig, axes = plt.subplots(1, min(5, len(crops)), figsize=(15, 3))
if len(crops) == 1:
    axes = [axes]

for ax, crop_path in zip(axes, crops[:5]):
    img = cv2.imread(crop_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get object name from filename
    name = crop_path.split("_")[-1].replace(".jpg", "").replace("_", " ")
    
    ax.imshow(img)
    ax.set_title(name, fontsize=10)
    ax.axis('off')
    
    # Show dimensions
    h, w = img.shape[:2]
    ax.text(0.5, -0.1, f"{w}x{h}", transform=ax.transAxes, 
            ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("results/crop_visualization.png", dpi=150, bbox_inches='tight')
print("✓ Saved to results/crop_visualization.png")

# Analyze crop sizes
print("\nCrop sizes:")
for crop_path in crops:
    img = cv2.imread(crop_path)
    h, w = img.shape[:2]
    name = crop_path.split("/")[-1]
    print(f"  {name:30s}: {w:4d}x{h:4d} = {w*h:6d} pixels")
