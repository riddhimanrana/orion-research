# Phase 3: Better Depth Integration (Weeks 5-6)

**Date**: November 9, 2025  
**Goal**: Improve scale accuracy and robustness by better integrating MiDaS depth estimates

---

## Current State Analysis

### What We Have

1. **MiDaS Depth Estimation**: Converting monocular RGB to metric depth
2. **Depth Preprocessing**: Bilateral filtering for noise reduction
3. **Scale Estimation**: Using depth at matched feature points
4. **Depth Odometry (ICP)**: Fallback when visual features fail
5. **Depth Tracking**: Optional depth-based pose estimation

### Current Limitations

❌ **Scale Ambiguity**: Monocular SLAM has no absolute scale  
❌ **Depth Noise**: MiDaS depth is noisy, especially at edges  
❌ **Scale Drift**: Scale estimation varies across frames  
❌ **Depth-Visual Fusion**: Currently depth is only used for fallback, not fused  
❌ **Outlier Sensitivity**: A few bad depth values can corrupt scale  
❌ **No Temporal Consistency**: Each frame estimates scale independently

### What Needs Improvement

1. **Better scale estimation** from depth
2. **Depth-visual fusion** (not just fallback)
3. **Temporal depth filtering** (Kalman filter for scale)
4. **Robust outlier rejection** for depth measurements
5. **Depth-guided feature tracking** (prioritize features with good depth)
6. **Depth uncertainty modeling** (confidence-weighted fusion)

---

## Implementation Plan

### Week 5: Enhanced Depth Processing

#### 1. Depth Uncertainty Map (Day 1-2)

**Goal**: Quantify confidence in each depth pixel

**Implementation**:
```python
class DepthUncertaintyEstimator:
    """Estimate confidence/uncertainty for each depth value"""
    
    def estimate_uncertainty(self, depth_map, rgb_frame):
        """
        Returns uncertainty map (0=certain, 1=uncertain)
        
        Factors:
        - Edge proximity (edges = unreliable depth)
        - Texture (low texture = uncertain depth)
        - Depth gradient (sharp changes = edges)
        - Distance (far = more uncertain)
        """
        uncertainty = np.zeros_like(depth_map)
        
        # Edge detection
        edges = cv2.Canny(rgb_frame, 50, 150)
        uncertainty += cv2.dilate(edges, None, iterations=2) / 255.0 * 0.3
        
        # Texture analysis
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        texture = cv2.Laplacian(gray, cv2.CV_64F)
        low_texture = (np.abs(texture) < 10).astype(float) * 0.2
        uncertainty += low_texture
        
        # Depth gradient
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        uncertainty += np.clip(grad_mag / 1000.0, 0, 0.3)
        
        # Distance uncertainty (far = uncertain)
        normalized_depth = depth_map / 10000.0  # Normalize to [0,1]
        uncertainty += normalized_depth * 0.2
        
        return np.clip(uncertainty, 0, 1)
```

**Integration**: Add uncertainty to `preprocess_depth_map()`

#### 2. Temporal Depth Filtering (Day 2-3)

**Goal**: Smooth depth across frames using exponential moving average

**Implementation**:
```python
class TemporalDepthFilter:
    """Smooth depth maps across time"""
    
    def __init__(self, alpha=0.7):
        self.alpha = alpha  # Weight for current frame
        self.filtered_depth = None
    
    def update(self, new_depth, uncertainty_map=None):
        """
        Exponential moving average with uncertainty weighting
        """
        if self.filtered_depth is None:
            self.filtered_depth = new_depth.copy()
            return self.filtered_depth
        
        if uncertainty_map is not None:
            # Weight by certainty (1 - uncertainty)
            certainty = 1.0 - uncertainty_map
            weights = certainty * self.alpha + (1 - certainty) * 0.5
        else:
            weights = self.alpha
        
        # EMA update
        self.filtered_depth = (weights * new_depth + 
                              (1 - weights) * self.filtered_depth)
        
        return self.filtered_depth
```

#### 3. Robust Scale Estimation (Day 3-4)

**Goal**: Better scale from depth using RANSAC + uncertainty weighting

**Current method**: Simple median of depth ratios  
**New method**: RANSAC + weighted median + temporal filtering

**Implementation**:
```python
def _estimate_scale_robust(
    self,
    pts1: np.ndarray,  # Previous frame points
    pts2: np.ndarray,  # Current frame points
    good_matches: List,
    depth_prev: np.ndarray,
    depth_curr: np.ndarray,
    uncertainty_prev: np.ndarray,
    uncertainty_curr: np.ndarray,
) -> Tuple[float, float]:
    """
    Robust scale estimation using depth with uncertainty.
    
    Returns: (scale, confidence)
    """
    
    # 1. Sample depth at matched features
    scales = []
    weights = []
    
    for i, match in enumerate(good_matches):
        u1, v1 = int(pts1[i][0]), int(pts1[i][1])
        u2, v2 = int(pts2[i][0]), int(pts2[i][1])
        
        # Get depth values
        d1 = depth_prev[v1, u1]
        d2 = depth_curr[v2, u2]
        
        # Get uncertainty
        unc1 = uncertainty_prev[v1, u1]
        unc2 = uncertainty_curr[v2, u2]
        
        # Skip if invalid or too uncertain
        if d1 < 100 or d2 < 100 or unc1 > 0.7 or unc2 > 0.7:
            continue
        
        # Compute 3D motion
        p1_3d = self._backproject(u1, v1, d1)
        p2_3d = self._backproject(u2, v2, d2)
        depth_motion = np.linalg.norm(p2_3d - p1_3d)
        
        # Compute 2D motion (from visual odometry)
        visual_motion = np.linalg.norm(pts2[i] - pts1[i])
        
        if visual_motion > 1.0:  # Minimum motion threshold
            scale = depth_motion / visual_motion
            scales.append(scale)
            
            # Weight by certainty
            certainty = (1 - unc1) * (1 - unc2)
            weights.append(certainty)
    
    if len(scales) < 10:
        return None, 0.0
    
    # 2. RANSAC to remove outliers
    scales = np.array(scales)
    weights = np.array(weights)
    
    best_scale = None
    best_inliers = 0
    
    for _ in range(50):  # RANSAC iterations
        # Sample random scale
        idx = np.random.randint(len(scales))
        candidate = scales[idx]
        
        # Count inliers (within 20% of candidate)
        inliers = np.abs(scales - candidate) < (0.2 * candidate)
        num_inliers = np.sum(inliers)
        
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_scale = candidate
    
    if best_inliers < len(scales) * 0.3:  # Need 30% inliers
        return None, 0.0
    
    # 3. Weighted median of inliers
    inlier_mask = np.abs(scales - best_scale) < (0.2 * best_scale)
    inlier_scales = scales[inlier_mask]
    inlier_weights = weights[inlier_mask]
    
    # Weighted median
    sorted_idx = np.argsort(inlier_scales)
    sorted_scales = inlier_scales[sorted_idx]
    sorted_weights = inlier_weights[sorted_idx]
    cumsum = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    
    final_scale = sorted_scales[median_idx]
    confidence = best_inliers / len(scales)
    
    return final_scale, confidence
```

#### 4. Depth-Guided Feature Selection (Day 4-5)

**Goal**: Prioritize tracking features with reliable depth

**Implementation**:
```python
def _select_features_with_depth(
    self,
    keypoints: List[cv2.KeyPoint],
    descriptors: np.ndarray,
    depth_map: np.ndarray,
    uncertainty_map: np.ndarray,
    max_features: int = 1500
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Select best features for tracking based on:
    1. Feature response (ORB corner strength)
    2. Depth validity (not too close/far)
    3. Low uncertainty
    """
    
    # Score each feature
    scores = []
    for kp in keypoints:
        u, v = int(kp.pt[0]), int(kp.pt[1])
        
        # Feature response (already in kp.response)
        response_score = kp.response / 1000.0  # Normalize
        
        # Depth validity
        depth = depth_map[v, u]
        if 100 < depth < 8000:  # Valid range
            depth_score = 1.0
        else:
            depth_score = 0.0
        
        # Uncertainty (lower is better)
        uncertainty = uncertainty_map[v, u]
        certainty_score = 1.0 - uncertainty
        
        # Combined score
        total_score = response_score * 0.4 + depth_score * 0.3 + certainty_score * 0.3
        scores.append(total_score)
    
    # Select top features
    scores = np.array(scores)
    top_idx = np.argsort(scores)[-max_features:]
    
    selected_kp = [keypoints[i] for i in top_idx]
    selected_desc = descriptors[top_idx]
    
    return selected_kp, selected_desc
```

### Week 6: Depth-Visual Fusion

#### 5. Kalman Filter for Scale (Day 1-2)

**Goal**: Smooth scale estimates across time

**Implementation**:
```python
class ScaleKalmanFilter:
    """Kalman filter for temporal scale smoothing"""
    
    def __init__(self):
        self.state = 1.0  # Initial scale
        self.covariance = 1.0  # Initial uncertainty
        
        # Process noise (how much scale can change per frame)
        self.process_noise = 0.01
        
    def predict(self):
        """Predict step: assume scale doesn't change much"""
        self.covariance += self.process_noise
    
    def update(self, measurement: float, measurement_confidence: float):
        """Update step: incorporate new scale measurement"""
        
        # Measurement noise (inversely proportional to confidence)
        measurement_noise = 1.0 - measurement_confidence
        
        # Kalman gain
        K = self.covariance / (self.covariance + measurement_noise)
        
        # Update state
        self.state = self.state + K * (measurement - self.state)
        
        # Update covariance
        self.covariance = (1 - K) * self.covariance
        
        return self.state
```

#### 6. Depth-Visual Pose Fusion (Day 2-4)

**Goal**: Fuse visual odometry and depth odometry, not just fallback

**Implementation**:
```python
def _fuse_visual_depth_poses(
    self,
    visual_pose: np.ndarray,
    depth_pose: Optional[np.ndarray],
    visual_confidence: float,
    depth_confidence: float,
) -> np.ndarray:
    """
    Fuse visual and depth-based pose estimates.
    
    Use weighted average in SE(3) space (special Euclidean group).
    """
    
    if depth_pose is None:
        return visual_pose
    
    # Extract translation and rotation
    t_vis = visual_pose[:3, 3]
    R_vis = visual_pose[:3, :3]
    
    t_dep = depth_pose[:3, 3]
    R_dep = depth_pose[:3, :3]
    
    # Weighted translation (simple)
    w_vis = visual_confidence / (visual_confidence + depth_confidence)
    w_dep = depth_confidence / (visual_confidence + depth_confidence)
    
    t_fused = w_vis * t_vis + w_dep * t_dep
    
    # Weighted rotation (SLERP - Spherical Linear Interpolation)
    # Convert to quaternions for interpolation
    from scipy.spatial.transform import Rotation, Slerp
    
    r_vis = Rotation.from_matrix(R_vis)
    r_dep = Rotation.from_matrix(R_dep)
    
    # Interpolate
    key_times = [0, 1]
    key_rots = Rotation.from_quat([r_vis.as_quat(), r_dep.as_quat()])
    slerp = Slerp(key_times, key_rots)
    
    r_fused = slerp(w_dep)  # Interpolate towards depth by w_dep
    R_fused = r_fused.as_matrix()
    
    # Construct fused pose
    pose_fused = np.eye(4)
    pose_fused[:3, :3] = R_fused
    pose_fused[:3, 3] = t_fused
    
    return pose_fused
```

#### 7. Depth Consistency Check (Day 4-5)

**Goal**: Validate depth estimates using epipolar geometry

**Implementation**:
```python
def _check_depth_consistency(
    self,
    pts1: np.ndarray,
    pts2: np.ndarray,
    depth1: np.ndarray,
    depth2: np.ndarray,
    pose: np.ndarray,  # Relative pose from frame 1 to 2
) -> np.ndarray:
    """
    Check if depth values are consistent with visual odometry.
    
    Returns: consistency mask (True = consistent, False = outlier)
    """
    
    consistent = np.zeros(len(pts1), dtype=bool)
    
    for i in range(len(pts1)):
        # Backproject point in frame 1
        u1, v1 = pts1[i]
        d1 = depth1[int(v1), int(u1)]
        p1_3d = self._backproject(u1, v1, d1)
        
        # Transform to frame 2
        p2_3d_predicted = pose[:3, :3] @ p1_3d + pose[:3, 3]
        
        # Project to frame 2
        u2_pred, v2_pred = self._project(p2_3d_predicted)
        
        # Check if it matches observed point
        u2_obs, v2_obs = pts2[i]
        
        reprojection_error = np.sqrt((u2_pred - u2_obs)**2 + (v2_pred - v2_obs)**2)
        
        # Also check depth consistency
        d2_obs = depth2[int(v2_obs), int(u2_obs)]
        d2_pred = p2_3d_predicted[2]
        
        depth_error = abs(d2_obs - d2_pred) / d2_pred
        
        # Consistent if both errors are small
        if reprojection_error < 3.0 and depth_error < 0.3:
            consistent[i] = True
    
    return consistent
```

---

## Expected Improvements

### Metrics

**Before (Current)**:
- Scale drift: ±30-50% per 100 frames
- Zone stability: 4 zones (drift accumulation)
- Depth usage: Only for fallback

**After (Target)**:
- Scale drift: ±5-10% per 100 frames (5x improvement)
- Zone stability: 2 zones (minimal drift)
- Depth usage: Continuous fusion with visual
- Feature quality: 20-30% more reliable features (depth-guided)
- Pose accuracy: 15-25% improvement (fusion vs visual-only)

### Benefits

1. ✅ **Better scale consistency** → fewer zone splits
2. ✅ **More robust tracking** → handles low-texture areas
3. ✅ **Faster convergence** → depth guides initialization
4. ✅ **Outlier rejection** → depth validates visual estimates
5. ✅ **Complementary strengths** → visual (texture) + depth (scale)

---

## Implementation Priority

### High Priority (Must Have)
1. ✅ Depth uncertainty estimation
2. ✅ Robust scale estimation with RANSAC
3. ✅ Kalman filter for scale
4. ✅ Depth-visual pose fusion

### Medium Priority (Should Have)
5. ⚠️ Temporal depth filtering
6. ⚠️ Depth-guided feature selection
7. ⚠️ Depth consistency checks

### Low Priority (Nice to Have)
8. ⭕ Depth super-resolution
9. ⭕ Learning-based depth refinement
10. ⭕ Multi-frame depth fusion

---

## Testing Strategy

### Unit Tests
- Test depth uncertainty on synthetic data
- Test scale estimation with ground truth
- Test Kalman filter convergence

### Integration Tests
- Run on AG-50 dataset
- Compare zone count before/after
- Measure scale drift over 500 frames

### Validation Metrics
- Scale consistency: σ(scale) < 0.1
- Zone count: 2-3 zones (target achieved)
- Tracking success: > 95%

---

## Timeline

**Week 5** (Days 1-5):
- Day 1-2: Depth uncertainty + temporal filtering
- Day 3-4: Robust scale estimation
- Day 4-5: Depth-guided features

**Week 6** (Days 1-5):
- Day 1-2: Kalman filter for scale
- Day 2-4: Depth-visual fusion
- Day 4-5: Testing + validation

**Total**: ~10 days of focused implementation

---

## Success Criteria

✅ Scale drift reduced to < 10% over 100 frames  
✅ Zone count stable at 2-3 zones  
✅ Depth-visual fusion working (not just fallback)  
✅ Tracking success rate > 95%  
✅ No performance regression (still real-time)

---

## Next Phase Preview

**Phase 4: Multi-Object Tracking Enhancement**
- Better Re-ID across occlusions
- Appearance model fusion
- Trajectory smoothing
- Association improvements
