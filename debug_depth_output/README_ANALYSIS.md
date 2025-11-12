# Depth Estimation Debug Analysis Guide

## 📍 Location
`debug_depth_output/frame_XXXX_depth_debug.jpg`

## 🖼️ Visualization Layout

Each debug image is a 2x2 grid showing:

### Top-Left: Original RGB + Detections
- Green boxes = Objects with known size priors (keyboard, tv, mouse)
- Gray boxes = Objects without size priors
- Labels show class name and confidence score

### Top-Right: Color-Coded Depth Map
- **Blue/Purple** = Close objects (~0.5-1m)
- **Green/Yellow** = Mid-range (~1-2m)  
- **Orange/Red** = Far objects (~2-5m)
- Color scale legend on the right shows min/max depth in meters

### Bottom-Left: Depth Map + Object Measurements
- Each detected object shows:
  - **Med**: Median depth in bounding box
  - **Avg**: Average depth
  - **Range**: Min-max depth range
- This shows what the model "sees" for each object

### Bottom-Right: Statistics Panel
Shows:
- **Depth Statistics**: Overall scene depth (range, mean, median, std dev)
- **Scale Correction**: 
  - Factor applied (e.g., 0.327x means objects are 3x closer than initial estimate)
  - Number of anchor objects used (objects with size priors that passed validation)
- **Detected Objects**: List with:
  - Green = Has size prior (used for scale)
  - Gray = No size prior
  - Shows expected real-world size if available

## 🔍 Current Results Summary

From the 5 frames analyzed:

### Frame 0:
- **Raw depth**: 0.50-5.00m (mean: 2.72m)
- **Scale correction**: 0.327x using 2 objects (TV + mouse)
- **Corrected depth**: 0.16-1.64m (mean: 0.89m)

### Frames 10-100:
- **Raw depth**: 0.50-5.00m (mean: 2.60-2.92m)
- **Scale correction**: 0.313x using 1-2 objects
- **Corrected depth**: 0.16-1.57m (mean: 0.82-0.91m)

## ❓ Questions to Investigate

### 1. **Are the corrected depths realistic?**
   - Desktop keyboard/mouse: Should be ~0.5-1.0m away
   - TV on desk: Should be ~0.8-1.5m away
   - Wall/background: Should be ~2-4m away
   
   👉 **Check**: Do the color-coded regions match what you expect?

### 2. **Why is keyboard rejected for scale recovery?**
   - Terminal shows: `keyboard: scale 0.10 out of range [0.2, 5.0]`
   - This means keyboard appears 10x farther than expected
   
   👉 **Check**: Look at the keyboard's depth values in the visualization
   - Is it the same depth as nearby objects (mouse)?
   - Could it be at an angle/perspective issue?
   - Is the bounding box accurate?

### 3. **Is Depth Anything V2 accurate enough?**
   - The model outputs relative depth (0-1), we scale it to 0.5-5m
   - Then semantic scale recovery adjusts by ~0.3x (objects 3x closer)
   
   👉 **Check**: 
   - Does the depth map look smooth/continuous?
   - Are edges preserved (keyboard edges, TV edges)?
   - Do similar-distance objects have similar colors?

### 4. **Ground plane at 0.00m - why?**
   - Ground plane detector looks at bottom 40% of image
   - Returns 0.00m which seems wrong
   
   👉 **Check**: 
   - What's in the bottom 40% of the image?
   - Is there actually a visible floor/ground?
   - Or is it mostly desk/wall?

## 🎯 What to Look For in the Images

### Good Signs ✅:
- [ ] Smooth depth gradients (no weird jumps)
- [ ] Objects at similar real-world distances have similar depths
- [ ] Closer objects are blue/purple, farther are orange/red
- [ ] Object boundaries are clear in depth map
- [ ] Scale correction brings depths to realistic ranges

### Problem Signs ❌:
- [ ] Depth map is very noisy/patchy
- [ ] Objects at same distance have very different colors
- [ ] Edges are blurry/inaccurate
- [ ] Scale correction is extreme (>2x or <0.2x)
- [ ] Corrected depths still seem unrealistic

## 🚀 Next Steps Based on Findings

### If depth looks generally correct but scale is off:
→ Adjust OBJECT_SIZE_PRIORS or initial depth range

### If depth map is noisy/inaccurate:
→ Consider switching to ZoeDepth or Metric3D (true metric depth)

### If specific objects are wrong:
→ Improve semantic scale recovery logic or add more object priors

### If everything looks good:
→ Enable SLAM for multi-frame refinement and absolute scale

## 📊 Expected Real-World Depths (Desktop Scene)

Based on typical desktop/office setup:
- **Keyboard**: 0.5-0.8m (arm's reach)
- **Mouse**: 0.5-0.7m (next to keyboard)
- **Monitor/TV**: 0.6-1.0m (on desk)
- **Desk surface**: 0.5-0.8m
- **Wall behind**: 1.5-3.0m
- **Ceiling**: 2.5-3.5m

Compare these to the depth values shown in the visualizations!
