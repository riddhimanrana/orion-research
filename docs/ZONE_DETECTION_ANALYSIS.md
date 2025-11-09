# Zone Detection Analysis

## Current Status

**Test Result**: 8 zones detected in 66-second video  
**Expected**: 3-4 zones (bedroom 1 → bedroom 2 → hallway → bedroom 3)  
**Problem**: Over-detection due to coordinate frame inconsistency

## Root Cause: Monocular Depth Limitation

### The Issue
Without SLAM/visual odometry, monocular depth estimation creates **camera-relative coordinates**:
- Frame 0: Object at (500mm, 0mm, 2000mm) - bedroom 1
- Frame 500: Same physical location might be (−500mm, 0mm, 2000mm) - different camera angle
- System sees these as **two different locations** → creates 2 zones

### Why Re-ID Fails
Zone re-identification tries to match using:
1. **Spatial distance** (80% weight): Fails because coordinates shift with camera
2. **Semantic embedding** (20% weight): Not distinctive enough between similar rooms

## Solutions

### Option 1: Visual SLAM Integration (Future)
- Integrate ORB-SLAM3 or similar
- Build consistent world map
- Track camera poses
- Transform all observations to world coordinates
- **Effort**: High (2-3 weeks)
- **Accuracy**: Excellent

### Option 2: Optical Flow-Based Registration (Medium)
- Track feature points across frames
- Estimate relative camera motion
- Accumulate transformations
- Register observations to approximate world frame
- **Effort**: Medium (3-5 days)
- **Accuracy**: Good for short sequences

### Option 3: Pragmatic Active Zone Tracking (IMPLEMENTED)
- Accept that coordinates are camera-relative
- Mark zones as "active" (seen in last 10s) or "inactive"
- Only show current active zone(s) in UI
- Keep all zones in memory for analytics
- Use scene classification to label zones (bedroom, hallway, etc.)
- **Effort**: Low (< 1 day)
- **Accuracy**: Moderate (good enough for demo)

## Recommendation

For Phase 3 demo: **Use Option 3**

### Implementation
1. Add `is_active` flag to zones based on `last_seen` timestamp
2. Show only active zones in visualization
3. Total zones = historical count (8), Active zones = current view (1-2)
4. Add scene-based labeling (bedroom, hallway, office)

### Expected Output
```
Phase 3 Statistics:
  Total zones discovered: 8 (historical)
  Active zones: 2 (current view)
  Zone types: bedroom (3), hallway (2), office (2), bathroom (1)
```

### For Production: Plan Option 1
- Research: ORB-SLAM3, Kimera, or similar
- Integration timeline: Phase 4
- Benefits: True multi-room mapping, loop closure, accurate world coordinates

---

**Next Steps**:
1. ✅ Implement active zone filtering
2. ✅ Add scene-based zone labeling
3. ⏸️ Plan SLAM integration for Phase 4
