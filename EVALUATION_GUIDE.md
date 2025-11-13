## DINO Integration & Visual Evaluation - Quick Guide

### What's New

**Embedding Backends:**
- ✅ **DINOv3 ViT-L/16** for instance-level Re-ID (1024-dim, viewpoint-robust)
- ✅ **CLIP ViT-B/32** for semantic checks and label verification (512-dim)
- ✅ Automatic backend selection: accurate mode uses DINO, quick/balanced use CLIP

**3D & Spatial:**
- ✅ **DepthAnythingV2** integrated (replaces MiDaS/ZoeDepth)
- ✅ Spatial zones: ceiling/floor/wall/center classification
- ✅ SLAM ready (camera pose estimation, loop closure)
- ✅ 3D backprojection with camera intrinsics

**Tracking:**
- ✅ EnhancedTracker with Hungarian matching
- ✅ Spatial-semantic cost function
- ✅ ID persistence metrics (switches, fragmentation)

**Visualization & Analysis:**
- ✅ Per-phase timing breakdown
- ✅ Per-class detection stats
- ✅ Entity persistence analysis
- ✅ Visual overlays (bboxes, zones, trajectories)

---

### Quick Start

#### 1. Run Perception Analysis (Console Output)
```bash
# Quick mode - fast sampling for rapid iteration
python scripts/analyze_detections.py \
  --video data/examples/video.mp4 \
  --mode quick \
  --tracking

# Accurate mode - DINO embeddings, 3D perception, SLAM
python scripts/analyze_detections.py \
  --video data/examples/video.mp4 \
  --mode accurate \
  --tracking \
  --save-entities

# Custom settings
python scripts/analyze_detections.py \
  --video my_video.mp4 \
  --mode balanced \
  --tracking \
  --target-fps 2.0 \
  --conf 0.3
```

**Output:** Comprehensive console report with:
- Detection counts per class
- Avg confidence & duration
- Entity persistence (lifetime)
- Spatial zone distribution
- Tracking quality (ID switches, fragmentation)
- Performance breakdown (detection, embedding, tracking, clustering, description)

---

#### 2. Generate Visual Snapshots
```bash
# Extract annotated frames
python scripts/snapshot_vis.py \
  --video data/examples/video.mp4 \
  --num-frames 5 \
  --output-dir results/snapshots

# View snapshots
open results/snapshots/
```

**Output:** Annotated frames with bounding boxes, class labels, confidence scores.

---

#### 3. Create Full Video Visualization (Coming Soon)
```bash
# Full annotated video with tracking IDs, zones, trajectories
python scripts/visualize_detections.py \
  --video data/examples/video.mp4 \
  --mode quick \
  --tracking \
  --zones \
  --trails \
  --output results/vis_quick.mp4

# View
vlc results/vis_quick.mp4
```

**Output:** Video with:
- Bounding boxes + class labels
- Tracking IDs (if enabled)
- Spatial zone overlays
- Motion trails/trajectories
- Depth heatmap (optional)

---

#### 4. Export Timed Metrics
```bash
# Lightweight metrics for iteration speed
python scripts/eval_perception_run.py \
  --video data/examples/video.mp4 \
  --mode quick \
  --tracking

# Saves to results/perception_run.json
cat results/perception_run.json
```

---

### Results Summary (Example Video)

**Quick Mode (CLIP, yolo11n, 0.5 FPS):**
```
Unique entities: 6
Total detections: 45
Sampled frames: 21
Timings: detect=26s, embed=9s, track=0.6s, cluster=0.01s, describe=25s, total=64s
Tracking: 5 total tracks, 0 ID switches, 0.83 fragmentation ratio
```

**Accurate Mode (DINO, yolo11x, 0.5 FPS):**
```
Unique entities: 8
Total detections: 46
Sampled frames: 21
Timings: detect=31s, embed=170s, track=3s, cluster=0.02s, describe=43s, total=251s
Tracking: 5 total tracks, 0 ID switches, 0.62 fragmentation ratio
```

**Top Detected Objects:**
1. Person (13 detections, 1534 frames avg duration)
2. Book (10 detections, 531 frames)
3. Bed (6 detections, 590 frames)
4. TV (5 detections, 1947 frames)
5. Mouse, Keyboard (4-3 detections)

**Spatial Distribution:**
- Center: 33%
- Bottom-right: 20%
- Floor: 15%
- Top: 11%

---

### Config Presets

**Quick:**
- YOLO11n (fastest)
- CLIP 512-dim
- 0.5 FPS sampling
- Confidence: 0.5

**Balanced:**
- YOLO11m
- CLIP 512-dim
- 4 FPS sampling
- Confidence: 0.25

**Accurate:**
- YOLO11x (most accurate)
- **DINOv3 1024-dim** (instance Re-ID)
- 8 FPS sampling
- Confidence: 0.15
- **3D perception enabled** (DepthAnythingV2)
- **Tracking enabled** (EnhancedTracker)

---

### Programmatic Usage

```python
from orion.perception.config import get_accurate_config
from orion.perception.engine import run_perception

# Configure
cfg = get_accurate_config()
cfg.target_fps = 1.0  # sample every 1s
cfg.enable_tracking = True  # for ID persistence

# Run
result = run_perception('video.mp4', config=cfg)

# Access results
print(f"Detected {result.unique_entities} entities")
print(f"Metrics: {result.metrics}")

# Per-entity analysis
for entity in result.entities:
    print(f"{entity.entity_id}: {entity.object_class} "
          f"({entity.appearance_count} appearances, "
          f"frames {entity.first_seen_frame}-{entity.last_seen_frame})")
```

---

### Architecture Notes

**Pipeline Flow:**
1. **Detection** (YOLO) → frame sampling, object detection, cropping
2. **Embedding** (DINO/CLIP) → visual features + semantic checks
3. **Tracking** (EnhancedTracker) → Hungarian matching, spatial-semantic cost
4. **Clustering** (HDBSCAN) → group detections into unique entities
5. **Description** (FastVLM) → generate natural language descriptions

**DINO vs CLIP:**
- DINO: Instance-level discrimination (same object across viewpoints)
- CLIP: Semantic understanding (verify labels, detect misclassifications)
- In accurate mode: DINO for Re-ID, CLIP for verification

**Tracking Metrics:**
- `total_tracks`: Raw tracks created
- `confirmed_tracks`: Tracks with >3 detections
- `id_switches`: Identity confusion events
- `fragmentation`: tracks/entities ratio (1.0 = perfect)

---

### File Outputs

```
results/
├── perception_run.json      # Timed metrics + config
├── entities.json            # Per-entity details (--save-entities)
├── snapshots/               # Annotated frames
│   ├── frame_00000.jpg
│   └── ...
└── vis_accurate.mp4         # Full annotated video
```

---

### Tips

**Speed up iterations:**
- Use `--mode quick` and `--target-fps 0.5`
- Raise `--conf 0.5` to reduce detections
- Test on shorter clips first

**Improve accuracy:**
- Use `--mode accurate` for DINO embeddings
- Lower `--conf 0.15` for more detections
- Enable `--tracking` for temporal consistency

**Debug specific issues:**
- Use `snapshot_vis.py` to check a few frames visually
- Check `results/entities.json` for per-entity lifetimes
- Review tracking metrics: high fragmentation = too many duplicate tracks

---

### Next Steps

**Spatial Analytics:**
- Add zone occupancy heatmaps
- Track object movement between zones
- Detect spatial relationships (on, near, above)

**SLAM Integration:**
- Camera trajectory visualization
- World-coordinate object anchoring
- Loop closure for long videos

**SAM Masks:**
- Precise object boundaries
- Mask-guided embedding pooling
- Occlusion-aware Re-ID

**Performance:**
- Batch DINO embeddings for speed
- GPU acceleration for DepthAnythingV2
- Frame-skipping adaptive sampling
