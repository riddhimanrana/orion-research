# SLAM Demo - Quick Start

## Run the Demo

```bash
# Basic usage - interactive window
python scripts/run_slam_demo.py --video data/examples/video.mp4

# Save output video
python scripts/run_slam_demo.py --video data/examples/video.mp4 --output results/slam_demo.mp4
```

## Interactive Controls

During playback:

- **m** - Toggle SLAM trajectory mini-map
- **d** - Toggle depth heatmap overlay
- **h** - Show/hide help
- **Space** - Pause/resume playback
- **q** - Quit
- **Click entity** - Inspect entity details (ID, class, depth, 3D position)

## What You'll See

### Main Window
- **Current frame detections only** (no trails/history clutter)
- Bounding boxes with entity ID, class, confidence
- Depth distance for each entity
- SLAM tracking status (TRACKING/LOST)
- Semantic landmark rescue indicator (when active)
- Interactive status bar showing toggle states
- Entity details panel on click

### SLAM Mini-Map (press 'm')
- Top-down view of camera trajectory
- Blue (start) → Red (current) color gradient
- Waypoint markers
- Distance traveled
- Grid for scale reference

### Hover Highlights
- Yellow dashed border on mouse hover
- Entity ID and class label near cursor

## Architecture

Clean integration of three SLAM enhancements:

1. **Semantic SLAM** (`orion/slam/semantic_slam.py`)
   - Hybrid visual ORB + object landmark tracking
   - Rescues tracking in texture-less areas
   - 18 stable object classes (beds, furniture, doors, etc.)

2. **Trajectory Visualization** (`orion/perception/visualization.py`)
   - Real-time camera path overlay
   - Separate mini-map window
   - Distance calculation

3. **Interactive Controls** (`orion/perception/interactive_visualizer.py`)
   - Mouse click entity inspection
   - Keyboard shortcuts for toggles
   - Real-time overlay management

## Performance

- **Target FPS**: 1.0-1.5 FPS (real-time capable)
- **Current**: ~1.65 FPS on Apple Silicon
- **Overhead**: <3% for all enhancements combined

## Output

The demo shows:
- ✓ Current frame detections (clean, no history)
- ✓ Real-time depth estimation
- ✓ SLAM camera tracking
- ✓ Interactive inspection
- ✓ Clean, production-ready visualization

No clutter, no old tracking artifacts, just current state!
