# Production Optimization Roadmap

**Date**: January 2025  
**Status**: ✅ Motion-Adaptive SLAM + Rerun Working  
**Next**: Upgrade Rerun → ARKit Integration → Performance Optimization

---

## ✅ Completed (Current State)

### 1. **Research SLAM CLI** ✅
- Command: `python -m orion research slam --video <path> --viz rerun`
- Location: `orion/cli/commands/research.py`
- Features:
  - YOLO11x object detection
  - MiDaS depth estimation
  - Visual SLAM with loop closure
  - Entity tracking with Re-ID
  - Spatial zone construction
  - Scene classification

### 2. **Motion-Adaptive Frame Skip** ✅
- Dynamic skip rate: 2-30 frames based on:
  - Motion score (frame differencing)
  - SLAM tracking quality
  - Entity tracking stability
- High motion: skip=4 (~7.5 FPS)
- Low motion: skip=30 (~1 FPS)
- SLAM failure: skip=2 for recovery

### 3. **Rerun 3D Visualization** ✅
- Version: **0.26.2** (latest!)
- Features logged:
  - RGB frames
  - Depth maps (colorized)
  - 3D depth point clouds (backprojected)
  - Entity trajectories with velocity vectors
  - SLAM camera frustum
  - Spatial zone 3D meshes
  - Time-series metrics (FPS, entities, zones)
- Performance: 4x downsample, 5K points/frame, batch logging

### 4. **Performance Profiling** ✅
- Script: `scripts/profile_performance.py`
- Current: **0.73 FPS** (1379ms/frame)
- Bottlenecks identified:
  1. YOLO11x: 529ms (38.4%)
  2. SLAM: 666ms (48.3%)
  3. Depth: 102ms (7.4%)
  4. CLIP: 81ms (5.9%)

---

## 🎯 Phase 1: Quick Wins (Target: 1.0 FPS, ≤60s for 60s video)

### **Goal**: Process 60-second videos in ≤60 seconds (1x real-time)

### Actions:

#### 1. ✅ **Update Rerun Version Constraint**
**Status**: DONE ✅  
**File**: `pyproject.toml`  
**Change**: `rerun-sdk>=0.26.0` (was `>=0.19.0,<0.20.0`)

#### 2. 🔄 **Switch to YOLO11m** (Target: -229ms → 1150ms/frame)
**Status**: IN PROGRESS  
**Files to Edit**:
- `scripts/run_slam_complete.py` (line ~340)
- Download weights: `yolo11m.pt`

**Code Change**:
```python
# Before
yolo = YOLO("yolo11x.pt")

# After
yolo = YOLO("yolo11m.pt")  # 1.8x faster, good accuracy
```

**Expected Result**: 0.87 FPS (1150ms/frame)

#### 3. 🔄 **Tune Adaptive Skip** (Target: +30% speedup)
**Status**: IN PROGRESS  
**File**: `scripts/run_slam_complete.py` (lines 663-710)

**Code Change**:
```python
# Increase max_skip from 30 to 60
self.max_skip = 60  # 0.5 FPS for very static scenes

# Lower motion threshold
LOW_MOTION = 3.0  # Was 5.0 - skip more aggressively
```

**Expected Result**: Process 30-50% fewer frames → effective 2-3x speedup

#### 4. 🔄 **Skip SLAM for Low-Motion Frames** (Target: -200ms)
**Status**: IN PROGRESS  
**File**: `scripts/run_slam_complete.py`

**Code Addition** (after line 710):
```python
# Only run SLAM when needed
if avg_motion > 3.0 or self.frame_count % 10 == 0 or slam_pose is None:
    # Full SLAM tracking
    slam_result = self.slam.track(frame, timestamp, frame_idx, yolo_detections)
else:
    # Skip SLAM - use previous pose
    slam_result = {
        'success': True,
        'pose': self.prev_slam_pose,
        'features': 0,
        'matches': 0
    }
```

**Expected Result**: 0.95-1.05 FPS (950-1050ms/frame) ✅

---

## 🔥 Phase 2: Optimization (Target: 1.5 FPS, ≤40s for 60s video)

### **Goal**: 2.5x faster than real-time

### Actions:

#### 5. ⚡ **Reduce Input Resolution** (Target: -150ms)
**Status**: PLANNED  
**Implementation**: Add CLI flag `--resolution` to downscale frames

```python
# Add to CLI parser
parser.add_argument("--resolution", type=int, default=1920,
                    help="Target width (height scaled proportionally)")

# In processing loop
if args.resolution < frame.shape[1]:
    scale = args.resolution / frame.shape[1]
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
```

**Expected Result**: 1.2 FPS (830ms/frame)

#### 6. ⚡ **Adaptive SLAM Feature Count** (Target: -100ms)
**Status**: PLANNED  
**File**: `orion/slam/slam_engine.py`

```python
# In OpenCVSLAM.__init__
self.adaptive_features = True

# In track()
if self.tracking_good and reprojection_error < 2.0:
    max_features = 500  # Reduce from 1000
else:
    max_features = 2000  # Increase for recovery
    
self.detector.setMaxFeatures(max_features)
```

**Expected Result**: 1.35 FPS (740ms/frame)

#### 7. ⚡ **GPU-Accelerated Feature Matching** (Target: -50ms)
**Status**: PLANNED  
**File**: `orion/slam/slam_engine.py`

```python
# Use CUDA/MPS matcher
if torch.cuda.is_available():
    self.matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
elif torch.backends.mps.is_available():
    # MPS-accelerated matching (if available)
    self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
```

**Expected Result**: 1.45 FPS (690ms/frame)

---

## 🚀 Phase 3: Production-Ready (Target: 2.0+ FPS, ≤30s for 60s video)

### **Goal**: 3-4x faster than real-time

### Actions:

#### 8. 🚀 **YOLO Model Optimization**
**Options**:
- **YOLO11n**: 6.6x faster than 11x (80ms vs 529ms)
- **CoreML Export** (Apple Silicon): `yolo export format=coreml int8=True`
- **TensorRT** (NVIDIA): `yolo export format=engine half=True`

#### 9. 🚀 **Multi-Threaded Pipeline**
**Architecture**:
```
Frame Reader → Queue → [YOLO Thread] → Queue → [SLAM Thread] → Rerun Logger
                   ↘ [Depth Thread] ↗
```

#### 10. 🚀 **Depth Model Optimization**
**Options**:
- MiDaS-Small: 2x faster (50ms vs 100ms)
- Skip depth for low-motion frames
- Cache depth maps for static regions

---

## 🍎 ARKit Integration (Parallel Track)

### **Goal**: iOS scene reconstruction + Orion semantics

### Deliverables:

#### 1. 📱 **iOS ARKit Streaming App**
**Status**: PLANNED  
**Features**:
- Stream RGB + LiDAR depth @ 60 FPS
- Camera poses (6DOF from ARKit VIO)
- Scene mesh reconstruction
- Network streaming to Mac server

#### 2. 🐍 **Python ARKit Receiver**
**Status**: PLANNED  
**Location**: `orion/arkit/receiver.py`  
**Features**:
- TCP server for ARKit data
- Decode RGB/depth/pose
- Feed into Orion pipeline

#### 3. 🔗 **Sensor Fusion**
**Status**: PLANNED  
**Location**: `orion/arkit/fusion.py`  
**Features**:
- Kalman filter for ARKit + SLAM pose fusion
- LiDAR + MiDaS depth blending
- ARKit mesh → Orion spatial zones

#### 4. 📊 **Rerun Visualization**
**Status**: PLANNED  
**Updates**: `orion/visualization/rerun_logger.py`  
**Features**:
- Log ARKit camera poses
- Log LiDAR depth point clouds
- Log scene mesh
- Compare ARKit vs SLAM trajectories

---

## 📈 Performance Targets

| Metric | Current | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| **FPS** | 0.73 | 1.05 ⭐ | 1.54 🔥 | 2.86 🚀 |
| **ms/frame** | 1379 | 950 | 650 | 350 |
| **60s video** | 82s | 57s ⭐ | 39s 🔥 | 21s 🚀 |
| **Speedup** | 1.0x | 1.4x | 2.1x | 3.9x |

---

## 🛠️ Implementation Timeline

### **Week 1: Quick Wins**
- [ ] Day 1: Switch to YOLO11m, test performance
- [ ] Day 2: Tune adaptive skip thresholds
- [ ] Day 3: Implement SLAM skip for low motion
- [ ] Day 4: Test on 60s video, measure actual speedup
- [ ] Day 5: Documentation and benchmarks

**Target**: 1.0 FPS ✅

### **Week 2: Optimization**
- [ ] Day 1: Add resolution scaling CLI flag
- [ ] Day 2: Implement adaptive SLAM features
- [ ] Day 3: GPU-accelerated feature matching
- [ ] Day 4: Profile and tune
- [ ] Day 5: Stress testing

**Target**: 1.5 FPS 🔥

### **Week 3-4: ARKit Integration**
- [ ] Week 3: iOS app development + streaming
- [ ] Week 4: Python receiver + sensor fusion
- [ ] Bonus: Rerun visualization for ARKit

**Target**: Working ARKit → Orion pipeline 🍎

### **Future: Production-Ready**
- [ ] Model quantization (CoreML/TensorRT)
- [ ] Multi-threaded pipeline
- [ ] Depth caching and optimization
- [ ] End-to-end stress testing

**Target**: 2.0+ FPS 🚀

---

## 📚 Documentation

### Created:
1. ✅ **Performance Optimization Guide**: `docs/PERFORMANCE_OPTIMIZATION.md`
   - Detailed bottleneck analysis
   - Optimization strategies with code examples
   - Performance targets and benchmarks

2. ✅ **ARKit Integration Guide**: `docs/ARKIT_INTEGRATION.md`
   - Architecture overview
   - iOS Swift code examples
   - Python receiver implementation
   - Sensor fusion algorithms
   - Rerun logging for ARKit

3. ✅ **This Roadmap**: `docs/PRODUCTION_ROADMAP.md`
   - Current status
   - Phased implementation plan
   - Timeline and deliverables

### To Update:
- `README.md`: Add performance benchmarks
- `docs/PHASE_4_README.md`: Link to optimization guide
- `docs/SYSTEM_ARCHITECTURE.md`: Add ARKit integration

---

## 🧪 Testing & Validation

### Performance Tests:
```bash
# Baseline (current)
python scripts/profile_performance.py \
  --video test.mp4 \
  --frames 100 \
  --use-slam \
  --use-depth

# After each phase
python scripts/profile_performance.py \
  --video test.mp4 \
  --frames 100 \
  --use-slam \
  --use-depth
```

### Accuracy Tests:
```bash
# Run on benchmark dataset
python scripts/3_run_orion_ag_eval.py \
  --config configs/phase1.yaml \
  --output results/optimized.json

# Compare metrics
python scripts/4_evaluate_ag_predictions.py \
  --baseline results/baseline.json \
  --optimized results/optimized.json
```

### Real-World Test:
```bash
# 60-second video
time python -m orion research slam \
  --video test_60s.mp4 \
  --viz rerun \
  --max-frames 1800

# Should complete in ≤60s for Phase 1 ✅
```

---

## 🎯 Success Criteria

### Phase 1 (Quick Wins):
- ✅ Process 60-second video in ≤60 seconds
- ✅ No accuracy degradation (mAP, ATE within 5%)
- ✅ Rerun visualization working smoothly
- ✅ Documentation complete

### Phase 2 (Optimization):
- ✅ Process 60-second video in ≤40 seconds
- ✅ Maintain accuracy (mAP, ATE within 10%)
- ✅ Scalable to longer videos (5+ minutes)

### Phase 3 (Production):
- ✅ Process 60-second video in ≤30 seconds
- ✅ Real-time processing for live video (1x speed)
- ✅ ARKit integration working end-to-end
- ✅ Deployed and battle-tested

---

## 🔗 Quick Links

- **Performance Guide**: `docs/PERFORMANCE_OPTIMIZATION.md`
- **ARKit Guide**: `docs/ARKIT_INTEGRATION.md`
- **Profiler**: `scripts/profile_performance.py`
- **SLAM Script**: `scripts/run_slam_complete.py`
- **CLI**: `orion/cli/commands/research.py`
- **Rerun Logger**: `orion/visualization/rerun_logger.py`

---

## 📞 Next Steps

1. **Review** this roadmap
2. **Prioritize** Phase 1 tasks
3. **Implement** YOLO11m switch (fastest win)
4. **Test** on 60s video
5. **Iterate** based on results

Let's ship it! 🚀
