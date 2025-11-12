"""Analyze frame content in detail"""
import cv2
import numpy as np

print("Analyzing room_original.jpg content...")
img = cv2.imread('debug_crops/room_original.jpg')

print(f"\nImage stats:")
print(f"  Shape: {img.shape}")
print(f"  Mean: {img.mean():.1f}")
print(f"  Std: {img.std():.1f}")
print(f"  Min: {img.min()}, Max: {img.max()}")

# Check edge density (sharp features)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
edge_density = (edges > 0).sum() / edges.size * 100
print(f"  Edge density: {edge_density:.2f}%")

# Check contrast
contrast = img.std()
print(f"  Contrast: {contrast:.1f}")

# Check if mostly uniform color (like a wall)
# Sample 100 random patches and check variance
np.random.seed(42)
patch_vars = []
for _ in range(100):
    y = np.random.randint(0, img.shape[0] - 64)
    x = np.random.randint(0, img.shape[1] - 64)
    patch = img[y:y+64, x:x+64]
    patch_vars.append(patch.std())

avg_patch_var = np.mean(patch_vars)
print(f"  Avg patch variance: {avg_patch_var:.1f}")

if edge_density < 1.0:
    print("\n⚠️  WARNING: Very low edge density - frame may be mostly empty/wall")
if avg_patch_var < 20:
    print("⚠️  WARNING: Low patch variance - content may be too uniform")
if contrast < 30:
    print("⚠️  WARNING: Low contrast - image may be washed out")

# Check histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
peak_bin = np.argmax(hist)
print(f"\n  Histogram peak at bin {peak_bin}/255 (brightness)")
print(f"  % pixels in peak ±10: {hist[max(0,peak_bin-10):min(256,peak_bin+10)].sum() / hist.sum() * 100:.1f}%")

# Check for motion blur
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
print(f"  Laplacian variance (sharpness): {laplacian_var:.1f}")
if laplacian_var < 100:
    print("⚠️  WARNING: Low sharpness - may be blurry")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

if edge_density < 1.0 and avg_patch_var < 20:
    print("❌ Frame appears to be mostly empty (wall/ceiling/floor)")
    print("   This is likely a slow pan of a blank surface")
    print("   YOLO cannot detect anything because there ARE no objects!")
elif laplacian_var < 100:
    print("❌ Frame is too blurry for object detection")
elif contrast < 30:
    print("❌ Frame has too low contrast")
else:
    print("❓ Frame appears normal but YOLO still fails - unexpected!")

# Compare with video_short
print("\n" + "="*60)
print("Comparing with video_short_original.jpg...")
img2 = cv2.imread('debug_crops/video_short_original.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
edges2 = cv2.Canny(gray2, 50, 150)
edge_density2 = (edges2 > 0).sum() / edges2.size * 100
laplacian_var2 = cv2.Laplacian(gray2, cv2.CV_64F).var()

print(f"  Edge density: {edge_density2:.2f}% (room: {edge_density:.2f}%)")
print(f"  Sharpness: {laplacian_var2:.1f} (room: {laplacian_var:.1f})")
print(f"  Contrast: {img2.std():.1f} (room: {img.std():.1f})")
