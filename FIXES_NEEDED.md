# CRITICAL FIXES FOR ORION SLAM + RE-ID ISSUES
## November 11, 2025

## Issues Identified:

### 1. **Depth Anything V2 Not Visualizing** (Purple/Invalid Depth in Rerun)
**Root Cause**: Depth map normalization producing invalid ranges or NaN values
**Location**: `orion/visualization/rerun_super_accurate.py` line 150-153

### 2. **Poor Re-ID / Duplicate Objects**
**Root Cause**: No temporal tracking - each frame creates new object IDs
**Missing**: ByteTrack/BoT-SORT integration, CLIP embeddings for re-identification

### 3. **Fast Motion / Motion Blur Issues**
**Root Cause**: No motion compensation or temporal smoothing
**Missing**: Optical flow assist, confidence-based filtering

### 4. **3D Positions Not Showing in Rerun**
**Root Cause**: Objects not being logged in world space with 3D boxes
**Location**: `orion/visualization/rerun_super_accurate.py` _log_3d_objects method

## Fixes To Implement:

### Fix 1: Depth Visualization (CRITICAL)
- Add NaN/Inf checking before normalization
- Use proper depth range (0.5m - 10m realistic indoor range)
- Add depth statistics logging
- Fix colormap application

### Fix 2: Temporal Object Tracking with Re-ID
- Integrate ByteTrack-style tracker
- Add CLIP embedding extraction per detection  
- Cosine similarity matching across frames
- Keep embedding buffer (last 10 frames)
- Optical flow prediction for occluded objects

### Fix 3: Motion Blur Handling
- Add Laplacian variance check (sharpness < 100 = skip)
- Temporal depth fusion (average last 5 frames)
- Confidence-based weighting
- Motion-aware detection threshold

### Fix 4: 3D Bounding Boxes in World Space
- Transform detection boxes to world coordinates using camera pose
- Log as Boxes3D with proper orientation
- Add velocity arrows
- Show object trails

## Implementation Priority:

1. **CRITICAL**: Fix depth visualization (5 min)
2. **HIGH**: Add basic temporal tracking with IDs (30 min)
3. **HIGH**: Add CLIP embeddings for re-ID (20 min)  
4. **MEDIUM**: Fix 3D box visualization (15 min)
5. **MEDIUM**: Motion blur detection (10 min)
6. **LOW**: Temporal depth fusion (future optimization)

## Files to Modify:

1. `orion/visualization/rerun_super_accurate.py` - Fix depth viz + 3D boxes
2. `scripts/test_super_accurate.py` - Add temporal tracker
3. `orion/perception/temporal_tracker.py` - NEW FILE for re-ID
4. `orion/perception/slam_fusion.py` - Add motion blur detection

Let's start with Fix 1 (CRITICAL)...
