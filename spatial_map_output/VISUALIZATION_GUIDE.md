# 📊 Spatial Map Visualization Guide

## What You're Seeing in `spatial_map_XXXX.jpg`

### Frame 0 (spatial_map_0000.jpg)
- **First 1/30th of a second of depth**
- Green circle in middle = camera at origin (0, 0)
- Red/yellow dots = far points (3-5m)
- Shows initial scene layout

### Frame 4 (spatial_map_0004.jpg)
- Points accumulating
- Pattern emerges showing the room geometry
- Furniture positions become visible
- ~46k points in map

### Frame 9 (spatial_map_0009.jpg)
- More accumulation
- Spatial map becoming richer
- 46,054 total points

### Frame 14 (spatial_map_0014.jpg)
- Well-established spatial representation
- Multiple passes creating denser map
- Clear object boundaries

### Frame 19 (spatial_map_0019.jpg)
- Final state of 20-frame sequence
- Most complete spatial understanding
- 9,734 voxels occupied

## Color Meaning

- **Bright colors** = High confidence depth (close/clear)
- **Dim colors** = Lower confidence (far/noisy)
- **Distribution pattern** = Scene geometry
- **Clustering** = Objects/walls

## Spatial Query Interpretation

When we ask:

### "What's within 1.5m?"
- Points close to camera
- Usually furniture, walls, clutter
- **Result: 7,977 points** = ~17% of map

### "What's to my left?"
- Azimuth from -90° (left) to -30° (left-front)
- Left side of view
- **Result: 1,611 points** = ~3% of map

### "What's to my right?"
- Azimuth from 30° (right-front) to 90° (right)
- Right side of view
- **Result: 587 points** = ~1% of map

### "What's forward?"
- Azimuth from -30° to 30° (forward cone)
- Things in front of camera
- **Result: 19,059 points** = ~41% of map

## Key Takeaway

The top-down view is like a **bird's eye view** of what the camera sees:
- Green center = "you are here"
- Distribution = where things are relative to you
- Density = confidence in those locations
- Gaps = occlusions or far away

---

## Integration Flow

```
Video Frame → YOLO Detection → Depth Map → Spatial Map → 3D Understanding
              (2D objects)      (metric 3D)   (accumulate)  (queries)
                                                ↓
                                        Can now answer:
                                        • Where is object X?
                                        • What's nearby?
                                        • What's clear to move to?
```

---

## Testing the Output

To regenerate or modify:

```bash
python scripts/test_spatial_map.py
```

This:
1. Loads `video_short.mp4` (or configure input)
2. Processes first 20 frames
3. Saves visualizations every 5 frames
4. Prints spatial queries results
5. Saves statistics JSON

---

## Next Iteration Ideas

1. **Add object labels to points** - Know which points are "keyboard", "TV", etc.
2. **Include camera motion** - Currently assumes stationary camera, can add SLAM poses
3. **Temporal persistence** - Let map grow across multiple video sequences
4. **Semantic layers** - Different occupancy grids for different object types
5. **Interactive query** - LLM: "Is it safe to move forward?" → check voxel grid ahead
