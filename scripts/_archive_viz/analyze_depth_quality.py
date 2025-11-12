#!/usr/bin/env python3
"""
Analyze if depth maps are actually good or just gradients
"""
import cv2
import numpy as np
from pathlib import Path

depth_dir = Path("/Users/riddhiman.rana/Desktop/Coding/Orion/orion-research/depth_debug_output")

print("\n" + "="*80)
print("📊 DEPTH MAP QUALITY ANALYSIS")
print("="*80)

for depth_img_path in sorted(depth_dir.glob("frame_*_depth.png")):
    # Read the depth image (already colorized)
    img = cv2.imread(str(depth_img_path))
    
    # We can't get original depth from colorized image, so let's extract RGB variation
    # High entropy = good scene structure, Low entropy = gradient
    
    b, g, r = cv2.split(img)
    
    # Compute gradients in each channel
    grad_x_r = np.abs(cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)).mean()
    grad_y_r = np.abs(cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)).mean()
    grad_x_g = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)).mean()
    grad_y_g = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)).mean()
    grad_x_b = np.abs(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)).mean()
    grad_y_b = np.abs(cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)).mean()
    
    # Total gradient
    total_grad = grad_x_r + grad_y_r + grad_x_g + grad_y_g + grad_x_b + grad_y_b
    
    # Color distribution (entropy)
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()
    
    # Shannon entropy
    hist_r = hist_r / hist_r.sum()
    hist_g = hist_g / hist_g.sum()
    hist_b = hist_b / hist_b.sum()
    
    entropy_r = -np.sum(hist_r[hist_r > 0] * np.log2(hist_r[hist_r > 0]))
    entropy_g = -np.sum(hist_g[hist_g > 0] * np.log2(hist_g[hist_g > 0]))
    entropy_b = -np.sum(hist_b[hist_b > 0] * np.log2(hist_b[hist_b > 0]))
    total_entropy = entropy_r + entropy_g + entropy_b
    
    print(f"\n{depth_img_path.name}:")
    print(f"  Total gradient: {total_grad:.2f}")
    print(f"  Total entropy: {total_entropy:.2f}")
    print(f"  Quality: {'✓ GOOD (varied depth)' if total_entropy > 6.0 else '⚠ SUSPECT (too smooth)'}")

print("\n" + "="*80)
print("INTERPRETATION:")
print("  Entropy > 7.0 = Varied depth structure (excellent)")
print("  Entropy 6-7 = Moderate structure (good)")
print("  Entropy < 6 = Very smooth/gradient-like (concerning)")
print("="*80 + "\n")
