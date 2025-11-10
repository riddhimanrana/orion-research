# Rerun.io Visualization Integration 🎨

**Phase 4 Week 3 - Interactive 3D Visualization**

## Overview

Rerun.io provides **professional-grade 3D visualization** for the Orion SLAM system, replacing OpenCV's limited `cv2.imshow()` windows with an interactive, browser-based interface.

## What is Rerun?

Rerun is a purpose-built visualization tool for robotics and computer vision:
- **Interactive 3D viewer** with rotation, zoom, pan
- **Timeline scrubbing** - go back/forward in time
- **Multiple synchronized views** (camera feed, depth, 3D space, metrics)
- **Real-time updates** via efficient logging
- **Export to .rrd files** for later replay
- **Web-based** - accessible from any device

## Features Logged

### 1. Video Frames (`camera/image`)
- RGB frames from the camera feed
- Real-time playback with timeline control

### 2. Depth Maps (`camera/depth`)
- Colorized depth visualization
- Automatically scaled to meters

### 3. Object Detections (`detections/boxes`)
- 2D bounding boxes overlaid on video
- Class labels with confidence scores
- Color-coded by class

### 4. Tracked Entities (`entities/current`)
- 3D positions of tracked objects (colored spheres)
- Persistent IDs across frames
- Entity trajectories showing movement paths (`entities/trajectories/{id}`)
- 10 distinct colors for different entities

### 5. SLAM Trajectory (`slam/camera`, `slam/trajectory`)
- Camera position and orientation in 3D
- Trajectory path with **color gradient**:
  - **Blue** → Start
  - **Green** → Middle  
  - **Yellow** → End
- Shows loop closures and camera movement

### 6. Spatial Zones (`zones/{zone_id}`)
- 3D boxes representing semantic zones
- Color-coded by room type:
  - **Cornflower Blue**: Bedroom
  - **Dark Orange**: Kitchen
  - **Sea Green**: Living Room
  - **Sky Blue**: Bathroom
  - **Gray**: Unknown
- Labels with zone type

### 7. System Metrics (`metrics/*`)
- FPS (frames per second)
- Number of tracked entities
- Number of spatial zones
- Total SLAM poses
- Loop closures detected

## Usage

### Basic Usage

```bash
# Run with Rerun visualization
python scripts/run_slam_complete.py --video data/examples/video.mp4 --rerun

# Limit frames for quick test
python scripts/run_slam_complete.py --video data/examples/video_short.mp4 --rerun --max-frames 100
```

### Test Rerun Separately

```bash
# Quick test without full SLAM pipeline
python scripts/test_rerun.py
```

This creates a simple visualization with random points to verify Rerun is working.

## Rerun Viewer Controls

### Timeline
- **Drag slider**: Scrub through timeline
- **Space**: Play/Pause
- **Arrow keys**: Step forward/backward

### 3D View
- **Left mouse + drag**: Rotate camera
- **Right mouse + drag**: Pan view
- **Scroll wheel**: Zoom in/out
- **Double-click entity**: Focus and select

### Panels
- **Blueprint**: Toggle visibility of entities
- **Selection**: View details of selected object
- **Time**: Control playback speed
- **Plot**: View metric graphs over time

## Code Architecture

### `RerunLogger` Class

Located in `orion/visualization/rerun_logger.py`

**Key Methods:**
- `log_frame()` - Log video frame
- `log_depth()` - Log depth map
- `log_detections()` - Log 2D bounding boxes
- `log_entities()` - Log 3D tracked entities with trajectories
- `log_slam_pose()` - Log camera pose and trajectory
- `log_zones()` - Log spatial zones as 3D boxes
- `log_metrics()` - Log system performance metrics
- `log_text()` - Log text annotations

### Integration in `run_slam_complete.py`

```python
# Initialize (in __init__)
if self.use_rerun:
    rerun_config = RerunConfig(
        app_name="orion-slam",
        spawn_viewer=True,
        log_video=True,
        log_depth=True,
        log_detections=True,
        log_entities=True,
        log_slam_trajectory=True,
        log_zones=True,
        log_metrics=True,
    )
    self.rerun_logger = RerunLogger(config=rerun_config)

# Log each frame (in run loop)
if self.rerun_logger:
    self.rerun_logger.log_frame(frame, self.frame_count)
    self.rerun_logger.log_depth(depth_map, self.frame_count)
    self.rerun_logger.log_detections(frame, detections, self.frame_count)
    self.rerun_logger.log_entities(tracks, self.frame_count, show_trajectories=True)
    self.rerun_logger.log_slam_pose(slam_pose, self.frame_count, show_trajectory=True)
    self.rerun_logger.log_zones(self.zone_manager.zones, self.frame_count)
    self.rerun_logger.log_metrics(...)
```

## Installation

```bash
# Install Rerun (compatible with numpy 1.x)
pip install 'rerun-sdk<0.20' 'numpy<2.0.0,>=1.26.0'
```

**Note:** Rerun 0.26+ requires numpy>=2, which conflicts with some dependencies (ultralytics, tensorflow). Version 0.19.1 works with numpy 1.26.x.

## Benefits Over OpenCV Windows

| Feature | OpenCV (`cv2.imshow`) | Rerun.io |
|---------|----------------------|----------|
| 3D Visualization | ❌ No | ✅ Interactive 3D |
| Timeline Scrubbing | ❌ No | ✅ Full timeline control |
| Multiple Views | 😐 Separate windows | ✅ Synchronized panels |
| Web Access | ❌ Local only | ✅ Browser-based |
| Export/Replay | ❌ No | ✅ .rrd files |
| Performance | 😐 Limited | ✅ Optimized |
| Entity Selection | ❌ Manual click | ✅ Click + inspect |
| Metrics Plots | ❌ No | ✅ Real-time graphs |

## Examples

### Example 1: View Entity Trajectories

1. Run with `--rerun` flag
2. In Rerun viewer, navigate to `entities/trajectories`
3. See colored paths showing how each entity moved through space

### Example 2: Inspect Loop Closures

1. Run SLAM with `--rerun`
2. View `slam/trajectory` - look for trajectory segments that come close together
3. Check `metrics/loop_closures` to see when they were detected
4. Observe zones merging in `zones/*`

### Example 3: Analyze Zone Classification

1. Run with `--rerun --zone-mode dense`
2. View `zones/*` in 3D
3. Colors indicate room types (bedroom=blue, kitchen=orange, etc.)
4. Click zones to see entity membership

## Troubleshooting

### Viewer doesn't open

```bash
# Manually open viewer
rerun
```

Then run script with Rerun enabled - it will connect to the viewer.

### Numpy version conflicts

```bash
# Force compatible versions
pip install 'rerun-sdk<0.20' 'numpy<2.0.0,>=1.26.0' --force-reinstall
```

### Slow startup

The first time you run, imports take ~30 seconds due to TensorFlow/MediaPipe loading. This is normal.

### Connection refused

If port 9876 is in use:
```bash
# Kill existing Rerun processes
killall rerun
```

## Future Enhancements

**Week 4 (Planned)**:
- [ ] 3D spatial map with zone boundaries
- [ ] Entity velocity vectors
- [ ] Uncertainty visualization (confidence ellipsoids)
- [ ] Semantic labels in 3D space
- [ ] Export annotated videos

## Resources

- **Rerun Docs**: https://www.rerun.io/docs
- **Examples**: https://www.rerun.io/examples
- **Python API**: https://ref.rerun.io/docs/python/

---

**Status**: ✅ Fully integrated (Phase 4 Week 3)  
**Last Updated**: November 9, 2025
