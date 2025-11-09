# Performance Optimization Plan: Targeting 1-1.5 FPS Real-Time Processing

## 🎯 Target Performance
**Goal**: Consistent **1.0-1.5 FPS** for real-time spatial mapping

**Current Performance** (Phase 4 with SLAM):
- **0.73 FPS** (~1.37s per frame) with default SLAM
- **0.7 FPS** (~1.43s per frame) with relaxed SLAM (60.3% tracking success)

**Gap**: Need **+30-50% speedup** to reach 1.0-1.5 FPS

---

## 📊 Bottleneck Analysis

### **Current Pipeline Breakdown** (estimated per-frame timing):

| Component | Time (ms) | % Total | Priority |
|-----------|-----------|---------|----------|
| **Depth Estimation** (ZoeDepth/MiDaS) | 600-800ms | **~55%** | 🔴 CRITICAL |
| **YOLO Detection** (YOLO11x) | 200-300ms | **~20%** | 🟡 HIGH |
| **SLAM** (ORB features + matching) | 150-200ms | **~13%** | 🟡 HIGH |
| **CLIP Embedding** | 50-80ms | **~5%** | 🟢 MEDIUM |
| **Visualization** | 30-50ms | **~3%** | 🟢 LOW |
| **Zone Detection** | 20-40ms | **~2%** | 🟢 LOW |
| **Other** (tracking, I/O) | 20-30ms | **~2%** | 🟢 LOW |
| **TOTAL** | **1370ms** | **100%** | - |

### **Key Insights**:
1. **Depth estimation is the #1 bottleneck** (55% of compute time)
2. YOLO + SLAM combined = 33% of time
3. Semantic components (CLIP, zones) are already efficient

---

## 🚀 Optimization Strategy

### **Phase 1: Depth Estimation Optimization** 🔴 (Target: 600ms → 300ms)

#### **Option 1A: Switch to FastDepth** ⭐ **RECOMMENDED**
- **Current**: ZoeDepth (600-800ms on MPS)
- **Replace with**: FastDepth (ResNet18 backbone)
- **Expected**: 150-200ms (3-4x faster)
- **Trade-off**: Slightly lower accuracy, but sufficient for zone detection
- **Implementation**:
  ```python
  # orion/perception/depth.py
  def _load_model(self):
      if self.model_name == "fastdepth":
          # Lightweight ResNet18-based depth
          self.model = torch.hub.load('intel-isl/FastDepth', 'fastdepth', pretrained=True)
      elif self.model_name == "midas_small":
          # MiDaS small variant (faster than v3.1)
          self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
  ```

#### **Option 1B: Reduce Depth Resolution**
- **Current**: Full 1080x1920 depth estimation
- **Optimize**: Downsample to 540x960 (0.5x) or 360x640 (0.33x)
- **Expected**: 2-3x speedup
- **Trade-off**: Coarser depth, but SLAM only needs sparse features
- **Implementation**:
  ```python
  def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
      h, w = frame.shape[:2]
      # Downsample for depth estimation
      frame_small = cv2.resize(frame, (w // 2, h // 2))
      depth_small = self._inference(frame_small)
      # Upsample back (fast bilinear interpolation)
      depth = cv2.resize(depth_small, (w, h), interpolation=cv2.INTER_LINEAR)
      return depth
  ```

#### **Option 1C: Conditional Depth** ⚡ **AGGRESSIVE**
- **Strategy**: Only run depth on keyframes or when SLAM fails
- **Logic**:
  - If SLAM tracking succeeds → skip depth, use previous depth map
  - If SLAM fails or new zone detected → run depth
- **Expected**: 50-70% reduction in depth calls
- **Implementation**:
  ```python
  def process_frame_conditional(self, frame, slam_success):
      if slam_success and self.frames_since_depth < 10:
          # Reuse previous depth
          depth_map = self.last_depth_map
          self.frames_since_depth += 1
      else:
          # Run full depth estimation
          depth_map = self.depth_estimator.estimate(frame)
          self.last_depth_map = depth_map
          self.frames_since_depth = 0
      return depth_map
  ```

---

### **Phase 2: SLAM Optimization** 🟡 (Target: 150ms → 80ms)

#### **Option 2A: Reduce ORB Feature Count**
- **Current**: 3000 features (relaxed params)
- **Optimize**: Use 1500 features (was default)
- **Conditional**: Use 3000 only when tracking quality drops
- **Expected**: 40% speedup in feature detection
- **Implementation**:
  ```python
  def adaptive_feature_count(self, tracking_quality):
      if tracking_quality > 0.8:
          return 1500  # High confidence, fewer features
      elif tracking_quality > 0.5:
          return 2000  # Medium confidence
      else:
          return 3000  # Low confidence, max features
  ```

#### **Option 2B: GPU-Accelerated Feature Matching**
- **Current**: CPU-based BFMatcher
- **Optimize**: Use OpenCV CUDA/GPU matching (if available)
- **Expected**: 2x speedup in matching phase
- **Implementation**:
  ```python
  if cv2.cuda.getCudaEnabledDeviceCount() > 0:
      self.matcher = cv2.cuda_BFMatcher.create(cv2.NORM_HAMMING)
  else:
      self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
  ```

#### **Option 2C: Frame Skipping for SLAM**
- **Strategy**: Run SLAM every N frames, interpolate between
- **Params**: Process every 2nd frame (30fps → 15fps effective)
- **Expected**: 50% reduction in SLAM calls
- **Trade-off**: Slightly less accurate trajectory
- **Implementation**:
  ```python
  def process_frame(self, frame, frame_idx):
      if frame_idx % self.slam_interval == 0:
          # Run full SLAM
          pose = self.slam_engine.track(frame)
      else:
          # Interpolate from previous pose
          pose = self.interpolate_pose(self.last_pose, frame_idx)
      return pose
  ```

---

### **Phase 3: YOLO Optimization** 🟡 (Target: 250ms → 150ms)

#### **Option 3A: Use YOLO11n (Nano) or YOLO11s (Small)**
- **Current**: YOLO11x (largest, 56.9M params)
- **Optimize**: YOLO11n (2.6M params) or YOLO11s (9.4M params)
- **Expected**: 3-4x speedup
- **Trade-off**: -5% mAP (still >40% mAP for detection)
- **Benchmark**:
  ```
  YOLO11x: 250ms/frame, 54.7 mAP
  YOLO11m: 120ms/frame, 51.5 mAP
  YOLO11s:  60ms/frame, 46.6 mAP
  YOLO11n:  40ms/frame, 39.5 mAP ⚡
  ```

#### **Option 3B: TensorRT Optimization** (NVIDIA GPUs only)
- **Strategy**: Export YOLO to TensorRT engine
- **Expected**: 2-3x speedup on CUDA
- **Implementation**:
  ```bash
  yolo export model=yolo11x.pt format=engine device=0
  ```

#### **Option 3C: Reduce Input Resolution**
- **Current**: 1080x1920 (full HD)
- **Optimize**: 640x640 (YOLOv8 standard) or 1080x1080 (crop)
- **Expected**: 2x speedup
- **Trade-off**: May miss small objects

---

### **Phase 4: End-to-End Optimizations** 🟢

#### **Option 4A: Model Quantization**
- **Apply to**: Depth model, YOLO, CLIP
- **Method**: INT8 quantization (PyTorch/CoreML)
- **Expected**: 20-30% speedup, 50% memory reduction
- **Implementation**:
  ```python
  # For MPS (Apple Silicon)
  model_int8 = torch.quantization.quantize_dynamic(
      model, {torch.nn.Linear}, dtype=torch.qint8
  )
  ```

#### **Option 4B: Async Processing Pipeline**
- **Strategy**: Decouple model inference with async queues
- **Expected**: 10-15% throughput improvement
- **Implementation**:
  ```python
  import asyncio
  from concurrent.futures import ThreadPoolExecutor
  
  async def async_pipeline(self, frame):
      with ThreadPoolExecutor() as executor:
          # Run depth and YOLO in parallel
          depth_future = executor.submit(self.depth_estimator, frame)
          yolo_future = executor.submit(self.yolo, frame)
          
          depth = await asyncio.wrap_future(depth_future)
          detections = await asyncio.wrap_future(yolo_future)
      
      # SLAM depends on both, runs after
      slam_pose = self.slam_engine.track(frame)
      return depth, detections, slam_pose
  ```

#### **Option 4C: Batch Processing** (for offline analysis)
- **Strategy**: Process frames in batches for better GPU utilization
- **Expected**: 20-30% speedup for offline processing
- **Not applicable** for real-time streaming

---

## 📈 Optimization Roadmap

### **Quick Wins** (1-2 days, +40% speedup)
1. ✅ Switch depth model: ZoeDepth → MiDaS_small or FastDepth
2. ✅ Reduce depth resolution: 1080x1920 → 540x960
3. ✅ YOLO model swap: YOLO11x → YOLO11s or YOLO11n
4. ✅ Adaptive SLAM features: 3000 → 1500 when tracking is good

**Expected result**: 1370ms → 850ms (**1.18 FPS**) ✅

### **Medium-Term** (1 week, +60% speedup)
5. Conditional depth estimation (skip on good SLAM frames)
6. SLAM frame skipping (every 2nd frame)
7. GPU-accelerated feature matching
8. Async pipeline (depth + YOLO in parallel)

**Expected result**: 1370ms → 700ms (**1.43 FPS**) ✅✅

### **Advanced** (2+ weeks, +80% speedup)
9. Model quantization (INT8)
10. TensorRT optimization (NVIDIA)
11. Custom lightweight depth network training
12. Semantic landmark SLAM (reduce feature computation)

**Expected result**: 1370ms → 550ms (**1.82 FPS**) 🚀

---

## 🔬 Profiling Tools

### **Create Performance Profiling Script**:

```python
# scripts/profile_performance.py
import cProfile
import pstats
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    yield
    elapsed = (time.time() - start) * 1000
    print(f"[TIMER] {name}: {elapsed:.2f}ms")

def profile_frame_processing(frame):
    with timer("Total Frame"):
        with timer("  Depth Estimation"):
            depth = depth_estimator.estimate(frame)
        
        with timer("  YOLO Detection"):
            detections = yolo(frame)
        
        with timer("  SLAM Tracking"):
            pose = slam_engine.track(frame)
        
        with timer("  CLIP Embedding"):
            embeddings = clip.encode_image(crops)
        
        with timer("  Visualization"):
            vis_frame = visualizer.draw(frame, tracks)
    
    return depth, detections, pose

# Run profiling
cProfile.run('profile_frame_processing(frame)', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumtime')
stats.print_stats(20)
```

---

## ✅ Recommended Action Plan

### **Immediate (This Week)**:
1. **Profile current pipeline** to confirm bottlenecks
2. **Switch to MiDaS_small** or FastDepth for depth estimation
3. **Reduce depth resolution** to 540x960 (0.5x)
4. **Test YOLO11s** as alternative to YOLO11x

### **Expected Outcome**:
- **Current**: 0.7 FPS (1370ms/frame)
- **After optimizations**: **1.2-1.4 FPS** (700-850ms/frame) ✅
- **Meets target**: 1.0-1.5 FPS for real-time spatial mapping

### **If Still Not Fast Enough**:
- Implement conditional depth (skip on good SLAM frames)
- Use YOLO11n (nano) for maximum speed
- Async pipeline (depth + YOLO parallel)

---

## 📝 Implementation Checklist

- [ ] Create profiling script (`scripts/profile_performance.py`)
- [ ] Benchmark current pipeline (record baseline metrics)
- [ ] Add FastDepth model to `orion/perception/depth.py`
- [ ] Add depth resolution parameter to config
- [ ] Benchmark YOLO11s vs YOLO11x
- [ ] Implement adaptive SLAM feature count
- [ ] Add conditional depth estimation logic
- [ ] Create performance comparison report
- [ ] Update documentation with optimized config

---

## 🎯 Success Metrics

| Metric | Current | Target | Optimized |
|--------|---------|--------|-----------|
| **FPS** | 0.7 | 1.0-1.5 | TBD |
| **Depth Estimation** | 600-800ms | <300ms | TBD |
| **YOLO Detection** | 200-300ms | <150ms | TBD |
| **SLAM Tracking** | 150-200ms | <100ms | TBD |
| **SLAM Success Rate** | 60.3% | >50% | TBD |
| **Zone Accuracy** | 5 zones | 3-4 zones | TBD |

---

**Next Steps**: Run profiling script to confirm bottlenecks, then implement Quick Wins for immediate +40% speedup.
