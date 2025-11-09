# PHASE 1: Depth + Hand + 3D Perception Foundation

**Objective**: Transform 2D perception into 3D-grounded perception with hand tracking. Establish the foundation for all downstream semantic understanding.

**Timeline**: Week 0–1 (CLI batch processing, desktop GPU)

**Success Criteria**:
- ✅ Depth maps generated for all frames (latency: <50ms/frame)
- ✅ Hand detections in 3D (3D palm center, 21 landmarks, pose classification)
- ✅ Camera intrinsics properly configured and documented
- ✅ 3D backprojection accurate (validate against known depth GT if available)
- ✅ Entity 3D centroids computed and stored
- ✅ Visibility states tracked (FULLY_VISIBLE, PARTIALLY_OCCLUDED, HAND_OCCLUDED, OFF_SCREEN)
- ✅ End-to-end latency: <150ms/frame (perception only)

---

## Architecture Overview

```
RAW VIDEO FRAME (RGB, t_i)
    ↓
┌─────────────────────────────────────┐
│  PARALLEL PERCEPTION MODULES         │
├─────────────────────────────────────┤
│  [Depth Estimation]                 │
│  ├─ Input: RGB frame (H×W×3)        │
│  ├─ Model: ZoeDepth small or MiDaS  │
│  ├─ Output: depth_map (H×W float)   │
│  └─ Latency: 30-50ms                │
│                                     │
│  [YOLO Detection]                   │
│  ├─ Input: RGB frame                │
│  ├─ Model: YOLO11x (existing)       │
│  ├─ Output: boxes, confidences      │
│  └─ Latency: 40ms (existing)        │
│                                     │
│  [Hand Detection]                   │
│  ├─ Input: RGB frame                │
│  ├─ Model: MediaPipe Hands          │
│  ├─ Output: landmarks_2d (21×2)     │
│  └─ Latency: 10-20ms                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3D BACKPROJECTION                  │
├─────────────────────────────────────┤
│  [Camera Intrinsics]                │
│  ├─ Input: fx, fy, cx, cy           │
│  ├─ Source: config or auto-estimate │
│  └─ Uses: depth_map + 2D coords     │
│                                     │
│  [Backproject YOLO Boxes]           │
│  ├─ For each box: compute 3D bbox   │
│  ├─ Centroid: mean(depth in box)    │
│  ├─ Volume: min/max depth in box    │
│  └─ Output: EntityState w/ 3D pos   │
│                                     │
│  [Backproject Hand Landmarks]       │
│  ├─ For each landmark: get depth    │
│  ├─ Convert to 3D (X, Y, Z)         │
│  ├─ Compute palm center             │
│  ├─ Classify pose (OPEN/CLOSED/...) │
│  └─ Output: Hand w/ 3D joints       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  OCCLUSION DETECTION                │
├─────────────────────────────────────┤
│  For each entity bbox_2d:           │
│  ├─ Sample depth in region          │
│  ├─ Compare to entity.depth_mean    │
│  ├─ Count pixels in front (z<mean)  │
│  ├─ If >30% occluded: mark OCCLUDED │
│  └─ Check if hand is occluding      │
└─────────────────────────────────────┘
    ↓
PerceptionResult (with 3D data)
├─ entities: List[EntityState]
├─ hand_detections: List[Hand]
├─ depth_map: np.ndarray
├─ timestamp: float
└─ visibility_map: Dict[entity_id, state]
```

---

## Key Components

### 1. Depth Estimator (`orion/perception/depth.py`)

```python
class DepthEstimator:
    """
    Monocular depth estimation for egocentric video
    """
    
    def __init__(self, model_name: str = "zoe"):
        """
        Args:
            model_name: "zoe", "midas_small", "dpt_small", etc.
        """
        self.model_name = model_name
        self.model = self._load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def estimate(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Estimate depth map for a single frame
        
        Args:
            frame: RGB image (H×W×3), uint8
        
        Returns:
            (depth_map, confidence_map)
            - depth_map: (H×W) float, normalized [0, 1] or metric scale
            - confidence_map: (H×W) float, per-pixel confidence [optional]
        """
        # Normalize and resize
        frame_tensor = self._preprocess(frame)
        
        # Run inference
        with torch.no_grad():
            depth_map = self.model(frame_tensor)
        
        # Postprocess
        depth_map = self._postprocess(depth_map)
        
        return depth_map, None  # confidence optional for MVP
    
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Normalize, resize, send to device"""
        ...
    
    def _postprocess(self, depth_map: torch.Tensor) -> np.ndarray:
        """Smooth, rescale, convert to uint8 or float32"""
        # Temporal smoothing (optional): apply exponential filter across frames
        # Spatial smoothing: bilateral or median
        ...
```

**Model Choice**:
- **ZoeDepth** (MVP): ~20-40ms/frame, good egocentric performance, <500MB
- **MiDaS small**: ~30-50ms/frame, proven, <500MB
- Trade-off: MiDaS older but more stable; ZoeDepth newer, better on near-field

**Latency Budget**: Target <50ms/frame; should be parallelizable with YOLO

---

### 2. Camera Intrinsics & 3D Backprojection

```python
@dataclass
class CameraIntrinsics:
    """Camera calibration parameters"""
    fx: float  # focal length x (pixels)
    fy: float  # focal length y (pixels)
    cx: float  # principal point x (pixels)
    cy: float  # principal point y (pixels)
    width: int  # image width
    height: int  # image height
    
    @classmethod
    def auto_estimate(cls, video_resolution: Tuple[int, int]) -> "CameraIntrinsics":
        """
        Auto-estimate intrinsics from video resolution
        Assumes 35mm equivalent ~28mm lens on phone camera
        """
        h, w = video_resolution
        # Rough heuristic: focal length ≈ sensor width / 2 * aspect ratio
        fx = fy = w  # simplified for MVP
        cx, cy = w / 2, h / 2
        return cls(fx, fy, cx, cy, w, h)

def backproject_point(u: float, v: float, depth_z: float, intrinsics: CameraIntrinsics) -> Tuple[float, float, float]:
    """
    Convert 2D pixel + depth to 3D world coordinates
    
    Args:
        u, v: pixel coordinates
        depth_z: depth value (mm or meters, depending on model output)
        intrinsics: camera intrinsics
    
    Returns:
        (X_3d, Y_3d, Z_3d) in millimeters
    """
    X_3d = (u - intrinsics.cx) * depth_z / intrinsics.fx
    Y_3d = (v - intrinsics.cy) * depth_z / intrinsics.fy
    Z_3d = depth_z
    return (X_3d, Y_3d, Z_3d)

def backproject_bbox(bbox_2d: Tuple[int,int,int,int], depth_map: np.ndarray, intrinsics: CameraIntrinsics) -> Dict:
    """
    Backproject 2D bounding box using depth map
    
    Returns:
        {
            'centroid_3d': (X, Y, Z),
            'bbox_3d': (x_min, y_min, z_min, x_max, y_max, z_max),
            'depth_mean': float,
            'depth_variance': float,
            'volume_3d': float,  # approximate volume in mm^3
        }
    """
    x1, y1, x2, y2 = bbox_2d
    depth_region = depth_map[y1:y2, x1:x2]
    
    # Compute statistics
    depth_mean = np.mean(depth_region)
    depth_std = np.std(depth_region)
    
    # Backproject corners and center
    corners_2d = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    corners_3d = [backproject_point(u, v, depth_mean, intrinsics) for u, v in corners_2d]
    centroid_3d = backproject_point((x1+x2)/2, (y1+y2)/2, depth_mean, intrinsics)
    
    # 3D bounding box (crude)
    xs = [c[0] for c in corners_3d]
    ys = [c[1] for c in corners_3d]
    zs = [c[2] for c in corners_3d]
    
    return {
        'centroid_3d': centroid_3d,
        'bbox_3d': (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)),
        'depth_mean': float(depth_mean),
        'depth_variance': float(depth_std**2),
    }
```

**Configuration**:
- CLI/config: allow override of camera intrinsics; default to auto-estimate
- Store in `config.yaml` or `--camera-intrinsics` CLI arg

---

### 3. Hand Tracker (`orion/perception/hand_tracker.py`)

```python
class HandTracker:
    """
    Detect hands and project to 3D using depth map
    """
    
    def __init__(self):
        self.mediapipe_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
    
    def detect(self, frame: np.ndarray, depth_map: np.ndarray, intrinsics: CameraIntrinsics) -> List[Hand]:
        """
        Detect hands in frame and project to 3D
        
        Returns:
            List[Hand] with 3D joint positions, pose, confidence
        """
        results = self.mediapipe_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_hand_landmarks:
            return []
        
        hands = []
        for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
            # Convert landmarks to pixel coordinates
            h, w = frame.shape[:2]
            landmarks_2d = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]
            
            # Backproject to 3D using depth map
            landmarks_3d = []
            for x_px, y_px in landmarks_2d:
                # Bilinear interpolation for depth
                z = self._sample_depth(depth_map, x_px, y_px)
                x_3d, y_3d, z_3d = backproject_point(x_px, y_px, z, intrinsics)
                landmarks_3d.append((x_3d, y_3d, z_3d))
            
            # Classify pose
            pose = self._classify_pose(landmarks_3d)
            
            # Compute palm center (average of wrist + base joints)
            palm_center = tuple(np.mean(landmarks_3d[:5], axis=0))
            
            hand = Hand(
                id=f"hand_{hand_idx}",
                landmarks_2d=landmarks_2d,
                landmarks_3d=landmarks_3d,
                palm_center_3d=palm_center,
                pose=pose,  # "OPEN", "CLOSED", "PINCH", etc.
                confidence=results.multi_handedness[hand_idx].classification[0].score,
                handedness=results.multi_handedness[hand_idx].classification[0].label,
            )
            
            hands.append(hand)
        
        return hands
    
    def _classify_pose(self, landmarks_3d: List[Tuple[float, float, float]]) -> str:
        """
        Classify hand pose from 3D landmarks
        
        Heuristic: measure finger opening, thumb-finger distances
        """
        thumb_tip = np.array(landmarks_3d[4])
        finger_tips = [np.array(landmarks_3d[i]) for i in [8, 12, 16, 20]]  # index, middle, ring, pinky
        
        avg_dist = np.mean([np.linalg.norm(f - thumb_tip) for f in finger_tips])
        
        if avg_dist > 100:  # mm; fingers spread
            return "OPEN"
        elif avg_dist < 30:  # mm; fingers closed
            return "CLOSED"
        else:
            return "PINCH"  # intermediate
    
    def _sample_depth(self, depth_map: np.ndarray, x_px: float, y_px: float) -> float:
        """
        Bilinear interpolation on depth map
        """
        h, w = depth_map.shape
        x_px = np.clip(x_px, 0, w - 1.01)
        y_px = np.clip(y_px, 0, h - 1.01)
        
        x0, y0 = int(x_px), int(y_px)
        x1, y1 = x0 + 1, y0 + 1
        
        wx, wy = x_px - x0, y_px - y0
        
        z00 = depth_map[y0, x0]
        z10 = depth_map[y0, x1]
        z01 = depth_map[y1, x0]
        z11 = depth_map[y1, x1]
        
        z = (1-wx)*(1-wy)*z00 + wx*(1-wy)*z10 + (1-wx)*wy*z01 + wx*wy*z11
        
        return z

@dataclass
class Hand:
    id: str
    landmarks_2d: List[Tuple[float, float]]  # 21 joints
    landmarks_3d: List[Tuple[float, float, float]]  # 3D projected
    palm_center_3d: Tuple[float, float, float]
    pose: str  # "OPEN", "CLOSED", "PINCH"
    confidence: float
    handedness: str  # "Left" or "Right"
```

---

### 4. Updated EntityState & Observation Types

```python
@dataclass
class EntityState:
    """Enhanced entity state with 3D information"""
    entity_id: str
    frame_number: int
    timestamp: float
    
    # === 2D (existing) ===
    class_label: str
    class_confidence: float
    bbox_2d_px: Tuple[int, int, int, int]  # x1, y1, x2, y2
    centroid_2d_px: Tuple[float, float]
    
    # === 3D (NEW) ===
    centroid_3d_mm: Tuple[float, float, float]  # (X, Y, Z) in millimeters
    bbox_3d_mm: Tuple[float, float, float, float, float, float]  # (x_min, y_min, z_min, x_max, y_max, z_max)
    depth_mean_mm: float
    depth_variance_mm: float
    
    # === Visual ===
    embedding: np.ndarray  # CLIP embedding
    
    # === Visibility (NEW) ===
    visibility_state: Literal["FULLY_VISIBLE", "PARTIALLY_OCCLUDED", "HAND_OCCLUDED", "OFF_SCREEN"]
    occlusion_percentage: float  # 0-1
    occlusion_by: Optional[str]  # "hand" or entity_id of occluding object
```

---

### 5. Occlusion Detection (`orion/perception/occlusion_detector.py`)

```python
class OcclusionDetector:
    """
    Detect occlusions using depth map
    """
    
    def detect_occlusions(
        self,
        entities: List[EntityState],
        hands: List[Hand],
        depth_map: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> Dict[str, OcclusionInfo]:
        """
        For each entity, determine occlusion state
        
        Returns:
            {entity_id: OcclusionInfo}
        """
        occlusions = {}
        
        for entity in entities:
            x1, y1, x2, y2 = entity.bbox_2d_px
            
            # Sample depth in entity region
            depth_region = depth_map[y1:y2, x1:x2]
            entity_depth = entity.depth_mean_mm
            
            # Find pixels in front of entity (z smaller by margin)
            margin_mm = 10  # 1cm
            occluding_pixels = depth_region < (entity_depth - margin_mm)
            
            occlusion_percentage = np.sum(occluding_pixels) / max(1, depth_region.size)
            
            # Determine occluding agent
            occluding_by = None
            if occlusion_percentage > 0.1:  # > 10% occluded
                
                # Check if hand is occluding
                for hand in hands:
                    hand_mask = self._project_hand_to_2d(hand, depth_map.shape)
                    hand_occluding = np.logical_and(
                        occluding_pixels,
                        hand_mask[y1:y2, x1:x2]
                    )
                    
                    if np.sum(hand_occluding) > 5:  # significant overlap
                        occluding_by = "hand"
                        break
            
            # Determine visibility state
            if occlusion_percentage > 0.5:
                visibility = "PARTIALLY_OCCLUDED"
            elif occluding_by == "hand":
                visibility = "HAND_OCCLUDED"
            elif entity.centroid_2d_px[0] < 0 or entity.centroid_2d_px[0] > depth_map.shape[1]:
                visibility = "OFF_SCREEN"
            else:
                visibility = "FULLY_VISIBLE"
            
            occlusions[entity.entity_id] = OcclusionInfo(
                entity_id=entity.entity_id,
                visibility_state=visibility,
                occlusion_percentage=float(occlusion_percentage),
                occluding_by=occluding_by,
            )
        
        return occlusions
    
    def _project_hand_to_2d(self, hand: Hand, image_shape: Tuple[int, int]) -> np.ndarray:
        """Project hand landmarks to binary mask"""
        mask = np.zeros(image_shape, dtype=bool)
        h, w = image_shape
        
        for x_px, y_px in hand.landmarks_2d:
            x_px, y_px = int(np.clip(x_px, 0, w-1)), int(np.clip(y_px, 0, h-1))
            # Draw circle around each landmark
            cv2.circle(mask, (x_px, y_px), radius=15, color=1, thickness=-1)
        
        return mask

@dataclass
class OcclusionInfo:
    entity_id: str
    visibility_state: str
    occlusion_percentage: float
    occluding_by: Optional[str]
```

---

## Integration Point: Updated PerceptionEngine

```python
class PerceptionEngine:
    def process_video(self, video_path: str) -> PerceptionResult:
        """
        Enhanced to include depth and hand tracking
        """
        
        # Initialize components
        self.depth_estimator = DepthEstimator(model_name="zoe")
        self.hand_tracker = HandTracker()
        self.occlusion_detector = OcclusionDetector()
        
        # Process video frame by frame
        for frame_idx, frame in enumerate(self._iterate_frames(video_path)):
            
            # === Step 1: Depth estimation (parallel) ===
            depth_map, _ = self.depth_estimator.estimate(frame)
            
            # === Step 2: YOLO detection (existing) ===
            detections = self.observer.detect_frame(frame)
            
            # === Step 3: Hand detection (parallel) ===
            hands = self.hand_tracker.detect(frame, depth_map, self.intrinsics)
            
            # === Step 4: 3D backprojection ===
            entities_3d = []
            for det in detections:
                bbox_2d = det['bbox']
                entity_data = backproject_bbox(bbox_2d, depth_map, self.intrinsics)
                
                entity_state = EntityState(
                    entity_id=det['entity_id'],
                    frame_number=frame_idx,
                    timestamp=frame_idx / self.fps,
                    class_label=det['class'],
                    class_confidence=det['confidence'],
                    bbox_2d_px=bbox_2d,
                    centroid_2d_px=((bbox_2d[0]+bbox_2d[2])/2, (bbox_2d[1]+bbox_2d[3])/2),
                    centroid_3d_mm=entity_data['centroid_3d'],
                    bbox_3d_mm=entity_data['bbox_3d'],
                    depth_mean_mm=entity_data['depth_mean'],
                    depth_variance_mm=entity_data['depth_variance'],
                    embedding=det['embedding'],  # from CLIP
                    visibility_state="UNKNOWN",  # will be set below
                )
                
                entities_3d.append(entity_state)
            
            # === Step 5: Occlusion detection ===
            occlusions = self.occlusion_detector.detect_occlusions(
                entities_3d, hands, depth_map, self.intrinsics
            )
            
            for entity in entities_3d:
                occ_info = occlusions[entity.entity_id]
                entity.visibility_state = occ_info.visibility_state
                entity.occlusion_percentage = occ_info.occlusion_percentage
            
            # === Step 6: Embed and store ===
            self.observations.extend(entities_3d)
            self.hand_detections.append(hands)
            self.depth_maps.append(depth_map)
        
        # Build result
        return PerceptionResult(
            entities=self._cluster_observations(),
            raw_observations=self.observations,
            hand_detections=self.hand_detections,
            depth_maps=self.depth_maps,  # optional, for debugging
            # ... existing fields
        )
```

---

## Testing & Validation Strategy

### Unit Tests
- **Depth estimator**: Compare output shapes, ensure no NaNs
- **Backprojection**: Known camera intrinsics + synthetic points → verify round-trip accuracy
- **Hand detection**: MediaPipe confidence thresholds, landmark plausibility (joints within body bounds)
- **Occlusion**: Synthetic depth maps with known occlusions

### Integration Tests
- **End-to-end on sample video**: Depth + YOLO + hand → verify EntityState fields populated
- **Latency**: Measure per-stage time; target <150ms total/frame
- **Output consistency**: Same video → same depth (if deterministic model)

### Dataset Validation (if depth GT available)
- **Ego4D with depth**: Compare estimated depth vs. GT depth using RMSE/MAE
- **Relative depth consistency**: Objects closer than others should have smaller depth

---

## Performance Targets

| Component | Latency | Memory | Note |
|-----------|---------|--------|------|
| Depth estimation | 30–50ms | 500MB model | ZoeDepth small |
| YOLO detection | 40ms | existing | existing |
| Hand tracking | 10–20ms | 300MB | MediaPipe |
| 3D backprojection | 5ms | <10MB | per-frame compute |
| Occlusion detection | 10ms | <10MB | depth map ops |
| **Total** | ~100–150ms | ~1GB | All parallel where possible |

---

## Failure Modes & Mitigations

| Failure | Symptom | Mitigation |
|---------|---------|-----------|
| Depth model fails near-field | Noisy depth close to camera | Temporal smoothing, bilateral filter, choose model tuned to near-field |
| Hand occlusion not detected | Hand overlap not marked | Test occlusion heuristic, increase margin or hand mask size |
| Camera intrinsics wrong | 3D coordinates nonsensical | Auto-estimate or allow CLI override; validate round-trip on known points |
| Depth rescaling issue | 3D coords in wrong scale (cm vs mm) | Normalize depth output; store scale factor metadata |

---

## Output Artifacts

- `perception_result.pkl`: Serialized PerceptionResult with all 3D data
- `depth_maps/`: Optional folder with depth map visualizations (colorized for debugging)
- `intrinsics.json`: Camera intrinsics used for backprojection
- `phase1_metrics.json`: Latency per frame, memory usage, entity count, etc.

---

## Next Phase (Phase 2)

- Use EntityState 3D positions to upgrade HDBSCAN clustering
- Implement Bayesian re-identification (object permanence)
- Track object trajectories in 3D
- Prepare for semantic analysis (state changes, causal reasoning)

---

## Configuration Example

```yaml
# config.yaml - Phase 1 settings

perception:
  depth:
    model: "zoe"  # or "midas_small", "dpt_small"
    device: "cuda"  # or "cpu"
    temporal_smoothing: true
    smooth_window_size: 5
  
  hand_tracking:
    enable: true
    model: "mediapipe"
    confidence_threshold: 0.7
    tracking_confidence: 0.7
  
  camera:
    intrinsics:
      fx: null  # auto-estimate if null
      fy: null
      cx: null
      cy: null
    width: 1920
    height: 1080

video:
  fps: 30
  target_fps: 2  # process every 15 frames (2 FPS) for MVP speed
  skip_rate: 15
```

---

## CLI Example

```bash
python -m orion.cli analyze \
  --video path/to/video.mp4 \
  --mode perception_3d \
  --depth-model zoe \
  --output-dir results/phase1 \
  --config config.yaml \
  --verbose
```

Expected output:
```
[Perception Engine] Loading models...
  ✓ YOLO11x loaded
  ✓ ZoeDepth loaded
  ✓ MediaPipe Hands loaded
[Perception Engine] Processing video...
  [Frame 0] YOLO: 12 detections | Depth: OK | Hands: 2 | Time: 124ms
  [Frame 1] YOLO: 11 detections | Depth: OK | Hands: 2 | Time: 118ms
  ...
[Perception Engine] Complete
  Total frames: 450
  Unique entities: 87
  Hand interactions: 234
  Total time: 58.5s (130ms/frame)
  Saved to: results/phase1/
```

