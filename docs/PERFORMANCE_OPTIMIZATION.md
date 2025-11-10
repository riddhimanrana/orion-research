# Performance Optimization Guide

**Date**: January 2025  
**Goal**: Process 60-second videos in ≤60 seconds (1x real-time or faster)  
**Current**: ~0.73 FPS (1379ms/frame) → 60s video takes ~82s

---

## 📊 Current Performance Profile

From profiling `video_short.mp4` (30 frames, 1080x1920, 30 FPS):

| Component | Mean (ms) | % Total | Priority |
|-----------|-----------|---------|----------|
| **YOLO Detection** | 528.9 | 38.4% | 🔴 **HIGH** |
| **SLAM Tracking** | 666.3 | 48.3% | 🟡 **MEDIUM** |
| **Depth Estimation** | 102.4 | 7.4% | 🟢 **LOW** |
| **CLIP Embedding** | 81.4 | 5.9% | 🟢 **LOW** |
| **Total per Frame** | 1379.0 | 100% | - |

**Current FPS**: 0.73 (1379ms/frame)  
**Target FPS**: ≥1.0 (≤1000ms/frame) for 60s video in 60s  
**Gap**: Need **379ms** reduction per frame

---

## 🔥 Optimization Strategies

### 1. **YOLO Detection Optimization** (🔴 HIGH PRIORITY - 38.4% of time)

#### Current: YOLO11x (~529ms/frame)

**Options**:

| Model | Latency | Accuracy | Speedup |
|-------|---------|----------|---------|
| YOLO11x (current) | 529ms | Highest | 1.0x |
| YOLO11m | ~300ms | High | **1.8x** ⭐ |
| YOLO11s | ~150ms | Medium | **3.5x** |
| YOLO11n | ~80ms | Lower | **6.6x** |

**Recommendation**: Try **YOLO11m** first (good accuracy/speed tradeoff)

**Implementation**:
```python
# In scripts/run_slam_complete.py or config
yolo = YOLO("yolo11m.pt")  # Change from yolo11x.pt
```

**Expected Impact**: -229ms → **1150ms/frame total** (0.87 FPS)

---

### 2. **Adaptive Frame Skip** (Current: fixed 15 frames)

#### Current Implementation:
- Motion-adaptive: 2-30 frame skip
- High motion: skip=4 (~7.5 FPS)
- Low motion: skip=30 (~1 FPS)

**Enhancement**: Add scene complexity detection
```python
# Skip more frames when:
# - Low motion (<5 px/frame)
# - High SLAM confidence (low reprojection error)
# - Stable entity tracking (high IoU matches)

if avg_motion < 5 and slam_confidence > 0.9 and tracking_stable:
    self.skip_frames = min(60, self.skip_frames + 5)  # Up to 60 (0.5 FPS)
```

**Expected Impact**: Process 30-50% fewer frames → **effective 2-3x speedup**

---

### 3. **SLAM Optimization** (🟡 MEDIUM PRIORITY - 48.3% of time)

#### Current: ~666ms/frame (high variance: σ=1325ms)

**High Variance** indicates:
- Some frames: 100-200ms (fast tracking)
- Other frames: 1000-2000ms (feature extraction, matching)

**Optimizations**:

#### A) **Adaptive Feature Count**
```python
# In SLAMConfig
if slam_tracking_good:
    max_features = 500  # Reduce from 1000
else:
    max_features = 2000  # Increase for recovery
```

#### B) **GPU-Accelerated Feature Matching**
```python
# Use CUDA/MPS for feature matching
matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
# OR
matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
matcher.setGpuDevice(0)
```

#### C) **Skip SLAM for Low-Motion Frames**
```python
# Only run SLAM when motion > threshold
if motion_score > 3.0 or frame_count % 10 == 0:
    slam_result = slam.track(frame, ...)
else:
    # Interpolate from previous pose
    slam_result = interpolate_pose(prev_pose, motion_vector)
```

**Expected Impact**: -200-300ms → **866-966ms/frame** (1.03-1.15 FPS) ⭐

---

### 4. **Depth Estimation Optimization** (🟢 LOW PRIORITY - 7.4% of time)

#### Current: MiDaS (~102ms/frame) - already good!

**Options**:
- **MiDaS-Small**: ~50ms (faster, less accurate)
- **Current MiDaS**: ~100ms (good balance) ⭐
- **MiDaS-Large**: ~200ms (more accurate, slower)

**Keep current** unless need more speed. Already optimized with:
- 4x downsample for 3D logging
- 5K point limit per frame
- Batch logging

---

### 5. **Input Resolution Reduction**

#### Current: 1080x1920 (2.07 MP)

**Options**:
```python
# Resize before processing
target_width = 960  # 50% reduction → 4x faster processing
scale = target_width / frame.shape[1]
frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)

# Process at lower resolution
detections = yolo(frame_resized)
depth = depth_estimator.estimate(frame_resized)

# Scale back results
detections[:, :4] /= scale
depth = cv2.resize(depth, (orig_width, orig_height))
```

**Expected Impact**: ~30-40% speedup on YOLO + depth → **-200ms** → **1179ms/frame**

---

### 6. **Model Quantization** (Advanced)

Convert YOLO to INT8 or FP16:
```bash
# Export YOLO to CoreML (Apple Silicon optimized)
yolo export model=yolo11m.pt format=coreml int8=True

# Or TensorRT (NVIDIA)
yolo export model=yolo11m.pt format=engine half=True
```

**Expected Impact**: 2-3x faster inference on YOLO

---

## 🎯 Recommended Optimization Path

### **Phase 1: Quick Wins** (Target: 1.0 FPS, ≤1000ms/frame)

1. ✅ **Switch to YOLO11m** (-229ms)
2. ✅ **Increase adaptive skip threshold** (skip more aggressively)
3. ✅ **Skip SLAM for low-motion frames** (-200ms)

**Result**: ~950ms/frame → **1.05 FPS** → **60s video in 57s** ✅

---

### **Phase 2: Further Optimization** (Target: 1.5 FPS, ≤667ms/frame)

4. ⚡ **Reduce input resolution** to 960px width (-150ms)
5. ⚡ **Adaptive SLAM features** (-100ms)
6. ⚡ **GPU-accelerated matching** (-50ms)

**Result**: ~650ms/frame → **1.54 FPS** → **60s video in 39s** ⭐

---

### **Phase 3: Production-Ready** (Target: 2.0+ FPS, ≤500ms/frame)

7. 🚀 **YOLO11n or quantized model** (-200ms)
8. 🚀 **CoreML/TensorRT export** (-100ms)
9. 🚀 **Multi-threaded pipeline** (parallel YOLO + depth)

**Result**: ~350ms/frame → **2.86 FPS** → **60s video in 21s** 🔥

---

## 📈 Performance Targets

| Level | FPS | ms/frame | 60s Video Time | Speedup |
|-------|-----|----------|----------------|---------|
| **Current** | 0.73 | 1379 | 82s | 1.0x |
| **Phase 1** ⭐ | 1.05 | 950 | 57s | 1.4x |
| **Phase 2** 🔥 | 1.54 | 650 | 39s | 2.1x |
| **Phase 3** 🚀 | 2.86 | 350 | 21s | 3.9x |

---

## 🛠️ Implementation Checklist

### Phase 1 (Quick Wins)
- [ ] Download YOLO11m weights
- [ ] Update `run_slam_complete.py` to use YOLO11m
- [ ] Tune adaptive skip thresholds:
  - [ ] Increase `max_skip` from 30 to 60
  - [ ] Lower motion threshold from 5 to 3
- [ ] Add SLAM skip logic for low-motion frames
- [ ] Test on 60s video
- [ ] Measure actual speedup

### Phase 2 (Optimization)
- [ ] Add resolution scaling option to CLI
- [ ] Implement adaptive SLAM feature count
- [ ] Profile GPU utilization
- [ ] Add GPU-accelerated feature matching
- [ ] Benchmark on 60s video

### Phase 3 (Production)
- [ ] Export models to optimized formats
- [ ] Implement multi-threaded pipeline
- [ ] Add queue-based frame processing
- [ ] Stress test with multiple videos
- [ ] Deploy optimized config

---

## 🧪 Testing & Validation

### Performance Regression Tests
```bash
# Baseline (current)
python scripts/profile_performance.py --video test.mp4 --frames 100 > baseline.txt

# After optimization
python scripts/profile_performance.py --video test.mp4 --frames 100 > optimized.txt

# Compare
diff baseline.txt optimized.txt
```

### Accuracy Validation
- Run on same test set before/after optimization
- Compare:
  - Object detection mAP
  - SLAM trajectory error (ATE)
  - Entity tracking accuracy
  - Scene classification F1

### Real-World Testing
```bash
# 60-second video test
time python -m orion research slam \
  --video test_60s.mp4 \
  --viz rerun \
  --max-frames 1800

# Should complete in ≤60s for Phase 1
```

---

## 📚 References

- **YOLO Performance**: https://docs.ultralytics.com/models/yolo11/#performance-metrics
- **MiDaS Models**: https://github.com/isl-org/MiDaS
- **CoreML Optimization**: https://developer.apple.com/machine-learning/core-ml/
- **Rerun Performance**: https://www.rerun.io/docs/reference/performance

---

## 🔗 Related Files

- **Profiler**: `scripts/profile_performance.py`
- **SLAM Pipeline**: `scripts/run_slam_complete.py`
- **SLAM Engine**: `orion/slam/slam_engine.py`
- **Model Manager**: `orion/managers/model_manager.py`
- **CLI**: `orion/cli/commands/research.py`
