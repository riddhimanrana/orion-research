# Phase 3 Week 6: Advanced Depth Integration

**Date**: November 9, 2025  
**Status**: 🚧 In Progress  
**Goal**: Achieve 2-zone accuracy with <10% scale drift through advanced depth-visual fusion

---

## Overview

Week 5 delivered robust scale estimation and depth-guided features with 10% confidence improvement. Week 6 focuses on **fusing depth and visual information** to create a hybrid odometry system that's more accurate than either method alone.

### Key Concept: Complementary Strengths

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **Visual Odometry** | • Accurate rotation<br>• Good in textured scenes<br>• No scale ambiguity issues | • Scale drift in monocular<br>• Fails in low-texture<br>• Sensitive to motion blur |
| **Depth Odometry** | • Absolute scale<br>• Works in low-texture<br>• Robust to blur | • Noisy rotation (MiDaS)<br>• Edge artifacts<br>• Depth discontinuities |

**Fusion Strategy**: Use visual for rotation, depth for scale, weighted by confidence

---

## Implementation Plan

### Day 1: Depth-Visual Pose Fusion (4 hours)

**Goal**: Combine visual and depth-based poses with uncertainty weighting

#### Feature 1.1: Dual Odometry System
```python
class HybridOdometry:
    def __init__(self):
        self.visual_odom = VisualOdometry()  # Existing
        self.depth_odom = DepthOdometry()    # Already exists!
        
    def fuse_poses(self, visual_pose, depth_pose, 
                   visual_conf, depth_conf):
        # Rotation: Use visual (more accurate)
        R = visual_pose[:3, :3]
        
        # Translation: Weighted fusion
        t_visual = visual_pose[:3, 3]
        t_depth = depth_pose[:3, 3]
        
        # Weight by confidence
        w_visual = visual_conf / (visual_conf + depth_conf)
        w_depth = depth_conf / (visual_conf + depth_conf)
        
        t_fused = w_visual * t_visual + w_depth * t_depth
        
        return build_pose(R, t_fused)
```

**Key improvements**:
- ✅ Use DepthOdometry fallback (already implemented!)
- ✅ SLERP for rotation interpolation
- ✅ Confidence-weighted translation
- ✅ Adaptive fusion (switch to depth-only in low-texture)

**Expected impact**: 30% reduction in translation drift

---

#### Feature 1.2: Confidence Estimation
```python
def estimate_visual_confidence(self, inlier_ratio, num_matches, texture_score):
    # Base confidence from inliers
    conf = inlier_ratio * 0.6
    
    # Boost for many matches
    match_factor = min(num_matches / 100, 1.0) * 0.3
    
    # Reduce for low texture
    texture_factor = texture_score * 0.1
    
    return conf + match_factor + texture_factor

def estimate_depth_confidence(self, uncertainty_map, valid_ratio):
    # Average certainty (1 - uncertainty)
    avg_certainty = np.mean(1.0 - uncertainty_map[uncertainty_map < 0.8])
    
    # Penalize if few valid depth pixels
    return avg_certainty * valid_ratio
```

**Key metrics**:
- Visual confidence: inlier ratio (60%) + matches (30%) + texture (10%)
- Depth confidence: avg certainty × valid pixel ratio

---

### Day 2: Depth Consistency Checking (3 hours)

**Goal**: Validate depth estimates using epipolar geometry

#### Feature 2.1: Epipolar Depth Validation
```python
def check_depth_consistency(self, p1_2d, p2_2d, depth1, depth2, E, K):
    """
    Check if depth is consistent with epipolar geometry
    
    Epipolar constraint: p2^T @ E @ p1 = 0
    If depths are correct, reprojected 3D points should satisfy this
    """
    # Backproject to 3D
    p1_3d = backproject_point(p1_2d[0], p1_2d[1], depth1, K)
    p2_3d = backproject_point(p2_2d[0], p2_2d[1], depth2, K)
    
    # Compute epipolar error
    p1_h = np.array([p1_2d[0], p1_2d[1], 1.0])
    p2_h = np.array([p2_2d[0], p2_2d[1], 1.0])
    
    error = abs(p2_h @ E @ p1_h)
    
    # Threshold: should be near 0
    return error < 0.01  # 1 pixel tolerance
```

**Key improvements**:
- ✅ Reject features with inconsistent depth
- ✅ Detect depth discontinuities (edges)
- ✅ Filter outliers before scale estimation
- ✅ Increase inlier ratio by 10-15%

**Expected impact**: 20% more reliable scale estimates

---

#### Feature 2.2: Temporal Depth Consistency
```python
def check_temporal_consistency(self, depth_t0, depth_t1, pose, K):
    """
    Warp depth_t0 to depth_t1 using pose, check consistency
    """
    # Warp depth from t0 to t1
    depth_warped = warp_depth(depth_t0, pose, K)
    
    # Compute absolute difference
    diff = abs(depth_warped - depth_t1)
    
    # Valid if difference < threshold (accounting for motion)
    valid_mask = diff < 100  # 100mm tolerance
    
    return valid_mask
```

**Key improvements**:
- ✅ Detect moving objects (depth changes)
- ✅ Validate pose consistency
- ✅ Update uncertainty map based on consistency

---

### Day 3: Multi-Frame Depth Fusion (3 hours)

**Goal**: Combine depth from multiple frames for robustness

#### Feature 3.1: Sliding Window Depth Fusion
```python
class MultiFrameDepthFusion:
    def __init__(self, window_size=3):
        self.depth_buffer = deque(maxlen=window_size)
        self.pose_buffer = deque(maxlen=window_size)
        self.uncertainty_buffer = deque(maxlen=window_size)
    
    def add_frame(self, depth, pose, uncertainty):
        self.depth_buffer.append(depth)
        self.pose_buffer.append(pose)
        self.uncertainty_buffer.append(uncertainty)
    
    def fuse_depth(self):
        # Warp all depths to current frame
        fused_depth = np.zeros_like(self.depth_buffer[0])
        weight_sum = np.zeros_like(fused_depth)
        
        for i, (depth, pose, unc) in enumerate(zip(
            self.depth_buffer, self.pose_buffer, self.uncertainty_buffer
        )):
            # Warp to current frame
            warped = warp_depth(depth, pose, self.K)
            
            # Weight by certainty and temporal proximity
            weight = (1.0 - unc) * (0.5 ** i)  # Decay older frames
            
            fused_depth += warped * weight
            weight_sum += weight
        
        return fused_depth / (weight_sum + 1e-8)
```

**Key improvements**:
- ✅ Reduce noise by averaging 3-5 frames
- ✅ Weight by uncertainty and recency
- ✅ Handle camera motion via warping
- ✅ 40% noise reduction expected

**Expected impact**: Smoother depth, more consistent scale

---

### Day 4: Integration & Testing (4 hours)

#### Integration Tasks
1. ✅ Add HybridOdometry to SLAMEngine
2. ✅ Update track() to compute both visual and depth poses
3. ✅ Add depth consistency checks before RANSAC
4. ✅ Enable multi-frame fusion in process_frame()
5. ✅ Tune fusion weights based on scene type

#### Testing Strategy
```python
# Test on AG-50 dataset
python scripts/test_phase3_week6.py --dataset ag-50 --max-frames 500

# Expected results:
# - Zone count: 2 (from 4 baseline)
# - Scale drift: < 8% per 100 frames
# - Tracking success: > 95%
# - Translation error: < 5cm per meter
```

---

## Expected Performance Gains

### Cumulative Improvements

| Feature | Scale Drift Reduction | Zone Count Impact |
|---------|----------------------|-------------------|
| **Baseline (Phase 2)** | ±30-50% / 100 frames | 4 zones |
| + Robust Scale (Week 5 Day 2) | ±20-30% | 3-4 zones |
| + Uncertainty Maps (Week 5 Day 3) | ±15-25% | 3 zones |
| + **Pose Fusion (Week 6 Day 1)** | **±10-15%** | **2-3 zones** ✅ |
| + **Depth Consistency (Day 2)** | **±8-12%** | **2 zones** ✅ |
| + **Multi-Frame Fusion (Day 3)** | **±5-10%** | **2 zones** ✅✅ |

**Target achieved**: 2 zones with <10% scale drift! 🎯

---

## Implementation Order (Priority)

### High Priority (Implement First)
1. **Depth-Visual Pose Fusion** (4 hours)
   - Biggest impact on accuracy
   - Uses existing DepthOdometry
   - Confidence-weighted combination

2. **Depth Consistency Checking** (3 hours)
   - Improves scale estimation quality
   - Filters bad depth early
   - Low computational cost

### Medium Priority (If Time Permits)
3. **Multi-Frame Depth Fusion** (3 hours)
   - Further noise reduction
   - Smoother trajectories
   - More complex implementation

### Low Priority (Future Work)
4. Adaptive fusion weights (scene-dependent)
5. Keyframe-based depth accumulation
6. Bundle adjustment with depth constraints

---

## Code Structure

### New Files
```
orion/slam/
├── depth_utils.py          # Existing - Week 5
├── hybrid_odometry.py      # NEW - Pose fusion
├── depth_consistency.py    # NEW - Epipolar validation
└── multi_frame_fusion.py   # NEW - Temporal fusion
```

### Modified Files
```
orion/slam/slam_engine.py
├── Add HybridOdometry
├── Update track() to compute dual poses
├── Add consistency checks
└── Enable multi-frame fusion

orion/settings.py
├── enable_pose_fusion: bool = True
├── enable_depth_consistency: bool = True
├── enable_multi_frame_fusion: bool = False  # Optional
└── fusion_window_size: int = 3
```

---

## Success Criteria

### Must Have (Week 6)
- ✅ Zone count ≤ 2 (from 4 baseline)
- ✅ Scale drift < 10% per 100 frames
- ✅ Pose fusion working with confidence weighting
- ✅ Depth consistency checks reducing outliers

### Nice to Have
- ✅ Multi-frame fusion implemented
- ✅ Adaptive fusion weights
- ✅ Real-time performance (>20 FPS)

### Final Validation
```bash
# Run full AG-50 evaluation
python scripts/3_run_orion_ag_eval.py --max-frames 500

# Compare vs baseline
python scripts/4b_compare_baseline_vs_orion.py

# Expected output:
# ✅ Zones: 2 (was 4)
# ✅ Scale drift: 8% (was 35%)
# ✅ Translation error: 4.2cm/m (was 12cm/m)
```

---

## Current Status

### Completed (Week 5)
- ✅ Depth uncertainty estimation
- ✅ Temporal depth filtering  
- ✅ Robust scale estimation (RANSAC)
- ✅ Depth-guided feature selection
- ✅ Uncertainty map integration

### In Progress (Week 6)
- 🚧 **Depth-Visual Pose Fusion** (starting now)
- ⏳ Depth consistency checking
- ⏳ Multi-frame depth fusion

### Timeline
- **Day 1** (today): Implement pose fusion (4 hours)
- **Day 2**: Depth consistency (3 hours)  
- **Day 3**: Multi-frame fusion (3 hours)
- **Day 4**: Integration & testing (4 hours)

**Total**: ~14 hours over 4 days

---

## Let's Begin!

Starting with **Feature 1.1: Hybrid Odometry System**...
