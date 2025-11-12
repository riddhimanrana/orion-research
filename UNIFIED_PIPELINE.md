# Orion Unified 9-Modality Perception Pipeline

Production-ready CLI for unified video perception with 5-phase detection reduction pipeline.

## Quick Start

```bash
# Process a video through all 5 phases
python -m orion run --video data/examples/video_short.mp4 --max-frames 60

# With benchmarking
python -m orion run --video data/examples/video_short.mp4 --max-frames 60 --benchmark

# Disable Rerun 3D visualization logging (faster)
python -m orion run --video data/examples/video_short.mp4 --max-frames 60 --no-rerun
```

## Pipeline Phases

The `orion run` command executes a complete 5-phase unified perception pipeline:

1. **Phase 1: UnifiedFrame** - Merge 10 modalities (YOLO, Depth, SLAM, etc.)
2. **Phase 2: Rerun Visualization** - Real-time 3D interactive visualization
3. **Phase 3: Scale Estimation** - SLAM-based metric scale recovery
4. **Phase 4: Object Tracking** - Temporal deduplication across frames
5. **Phase 5: Re-ID + CLIP** - Semantic deduplication using CLIP embeddings

### Example Results
- Input: 72 raw detections across 20 frames
- After Phase 4 (Tracking): 5 tracked objects
- After Phase 5 (Re-ID): 5 unified entities
- **Total Reduction: 14.4x**

## Command Options

```
usage: orion run [-h] [--video VIDEO] [--max-frames MAX_FRAMES] 
                 [--benchmark] [--no-rerun] [--runtime RUNTIME]

options:
  --video VIDEO             Path to input video (default: data/examples/video_short.mp4)
  --max-frames MAX_FRAMES   Maximum frames to process (default: 60)
  --benchmark               Show detailed timing breakdown per phase
  --no-rerun                Disable Rerun 3D visualization
  --runtime RUNTIME         Select runtime backend (auto or torch)
```

## Architecture

- **Detection**: YOLOv11x (real-time, 80 classes)
- **Depth**: MiDaS v2 (metric scale)
- **SLAM**: OpenCV Feature-based Visual Odometry + Depth
- **Tracking**: Centroid-based temporal assignment (Phase 4)
- **Re-ID**: CLIP embeddings + cosine similarity (Phase 5)

## Components

All pipeline components are in `orion/perception/`:

- `unified_pipeline.py` - Main CLI integration
- `unified_frame.py` - Unified data structure
- `pipeline_adapter.py` - Component adapter
- `rerun_visualizer.py` - 3D visualization
- `object_tracker.py` - Phase 4 tracking
- `reid_matcher.py` - Phase 5 Re-ID

## Testing

Run the test script to validate all phases:

```bash
python scripts/test_pipeline.py
```

Expected output shows detection → tracking → unification reduction chain.

## See Also

- `orion analyze` - Full semantic pipeline with knowledge graph
- `orion research slam` - Advanced SLAM debugging with visualization
- `orion config show` - View system configuration
