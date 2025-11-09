# Research Toolkit: Debugging, Evaluation, Profiling

Complete guide to Orion's research toolkit for debugging components, running benchmarks, and analyzing performance.

---

## Overview

The research toolkit (`orion/research/`) is designed for:
- **🔍 Debugging**: Inspect internal state of perception, tracking, semantic components
- **📊 Profiling**: Measure latency, memory usage, GPU utilization
- **📈 Evaluation**: Benchmark on Ego4D, ActionGenome datasets with full metrics
- **📉 Ablation Studies**: Compare 2D vs 3D CIS, with/without hands, occlusion models

---

## Part 1: Debugging & Visualization

### 1.1 Debug Visualizer

Visualize internal states of pipeline components.

```python
from orion.research.debug.visualizer import DebugVisualizer
import cv2
import numpy as np

# === Visualize Depth Map ===
depth_map = perceiver.estimate_depth(frame)

# Render as colorized heatmap
depth_viz = DebugVisualizer.render_depth_map(depth_map, colormap="turbo")
cv2.imshow("Depth Map", depth_viz)

# Custom colormap options: "turbo", "viridis", "hot", "cool", "spring", etc.


# === Visualize Hand Tracking ===
hands = hand_tracker.detect(frame)

# Draw hand landmarks + pose
hand_viz = DebugVisualizer.draw_hands(frame, hands, color=(0, 255, 0))
cv2.imshow("Hands", hand_viz)

# Each hand shows:
# - 21 landmarks connected by skeleton
# - Pose label: OPEN, CLOSED, PINCH
# - Confidence scores


# === Visualize Entity Tracking ===
tracking_result = tracker.track(frame, entities)

# Draw bounding boxes + entity IDs
track_viz = DebugVisualizer.draw_tracking(
    frame,
    tracking_result.entities,
    show_ids=True,
    show_confidence=True
)
cv2.imshow("Tracking", track_viz)
```

### 1.2 Frame Logger

Log per-frame data for post-hoc analysis.

```python
from orion.research.debug.frame_logger import FrameLogger
from pathlib import Path

logger = FrameLogger(Path("debug_output/frame_logs/"))

# Process video
for frame_id, frame in enumerate(video_stream):
    perception = perceiver.process_frame(frame)
    tracking = tracker.track(perception)
    semantic = semantics.process(tracking)
    
    # Log this frame
    logger.log_frame(frame_id, {
        'timestamp': frame_id / fps,
        'depth_map': perception.depth_map,
        'depth_confidence': perception.depth_confidence,
        'hands': perception.hands,
        'num_entities': len(tracking.entities),
        'entity_ids': [e.entity_id for e in tracking.entities],
        'scene_context': semantic.scene_context,
        'cis_matrix': semantic.cis_matrix,
        'causal_links_count': len(semantic.causal_links),
    })

# Save summary (JSON)
logger.save_summary("frame_summary.json")

# Output: debug_output/frame_logs/frame_summary.json
# {
#   "frame_0": {"timestamp": 0.0, "depth_confidence": 0.95, "num_entities": 3, ...},
#   "frame_1": {"timestamp": 0.033, "depth_confidence": 0.93, ...},
# }
```

### 1.3 Interactive HTML Debugger

Web-based inspector for frame-by-frame analysis.

```python
from orion.research.debug.html_debugger import HTMLDebugger

debugger = HTMLDebugger("debug_output/html_viewer/")

# Generate interactive HTML viewer
debugger.generate_viewer(
    frame_logs=frame_data,
    video_path="video.mp4"
)

# Open in browser:
# open debug_output/html_viewer/debug_viewer.html
```

**Features**:
- 🎬 Video timeline with frame slider
- 📊 Depth map heatmap rendering
- 🖐️ Hand landmark overlay with pose labels
- 📍 Entity tracking bounding boxes + IDs
- 📋 Frame metadata panel
- 🔗 JSON inspector for raw frame state
- ⏯️ Play/pause and frame-by-frame navigation

**Example**: Jump to frame 150, inspect depth confidence, examine entity IDs, etc.

### 1.4 Debug Inspector (Breakpoints)

Interactive breakpoint-based debugging.

```python
from orion.research.debug.inspector import DebugInspector

inspector = DebugInspector(paused=False)

# === Set Conditional Breakpoint ===
def check_depth_uncertainty(state):
    """Trigger breakpoint on low depth confidence"""
    confidence = state.get('depth_confidence', 1.0)
    return confidence < 0.5  # Unreliable depth

inspector.set_breakpoint('depth_estimator', check_depth_uncertainty)

# === Set Frame Breakpoint ===
def check_frame_number(state):
    return state.get('frame_id') == 150  # Pause at frame 150

inspector.set_breakpoint('frame_processor', check_frame_number)

# === Use During Pipeline ===
for frame_id, frame in enumerate(video_stream):
    perception = perceiver.process_frame(frame)
    
    # Check breakpoints
    if inspector.check_breakpoint('depth_estimator', {
        'frame_id': frame_id,
        'depth_confidence': perception.depth_confidence,
    }):
        print(f"Breakpoint triggered at frame {frame_id}")
        print(f"Depth confidence: {perception.depth_confidence}")
        # Drop into debugger
        import pdb; pdb.set_trace()
```

### 1.5 Entity Trajectory Inspector

```python
from orion.research.debug.inspector import DebugInspector

inspector = DebugInspector()

# Inspect single entity's history
entity_history = [
    {'frame': 0, 'class': 'cup', 'bbox': (100, 100, 150, 150), 'confidence': 0.92},
    {'frame': 1, 'class': 'cup', 'bbox': (105, 98, 155, 148), 'confidence': 0.89},
    {'frame': 2, 'class': 'cup', 'bbox': (110, 95, 160, 145), 'confidence': 0.87},
    # ... ID switch happens here?
    {'frame': 10, 'class': 'cup', 'bbox': (200, 150, 250, 200), 'confidence': 0.56},  # Confidence drop
]

inspector.inspect_entity("cup_0", entity_history)

# Output:
# === Entity cup_0 ===
# Frame 0: class=cup, bbox=(100, 100, 150, 150), confidence=0.92
# Frame 1: class=cup, bbox=(105, 98, 155, 148), confidence=0.89
# Frame 2: class=cup, bbox=(110, 95, 160, 145), confidence=0.87
# Frame 10: class=cup, bbox=(200, 150, 250, 200), confidence=0.56  ⚠️ Low confidence
```

---

## Part 2: Performance Profiling

### 2.1 Latency & Memory Profiling

Measure per-component performance across devices.

```python
from orion.research.profiling.profiler import PerformanceProfiler
from orion.utils.device import detect_device

profiler = PerformanceProfiler(device=detect_device())

# Profile depth estimation
depth_result = profiler.profile_component(
    "depth_estimation",
    perceiver.estimate_depth,
    frame,
    iterations=100  # Run 100 times, compute mean/std
)

# Output:
# {
#   "component": "depth_estimation",
#   "device": "cuda",
#   "latency_mean_ms": 45.2,
#   "latency_std_ms": 2.1,
#   "latency_min_ms": 42.5,
#   "latency_max_ms": 51.3,
#   "memory_peak_mb": 892,
#   "memory_mean_mb": 750,
#   "throughput_fps": 22.1,  # 1000 / latency_mean_ms
# }

# Profile hand tracking
hand_result = profiler.profile_component(
    "hand_tracking",
    hand_tracker.detect,
    frame,
    iterations=50
)

# Profile tracking
tracking_result = profiler.profile_component(
    "entity_tracking",
    tracker.track,
    perception_result,
    iterations=50
)

# Generate report
profiler.generate_report("profiling_report.md")
```

### 2.2 Device Information

```python
from orion.utils.device import get_device_info, TensorDevice

info = get_device_info()
print(f"Device: {info['device']}")
print(f"GPU: {info.get('gpu_name', 'N/A')}")
print(f"GPU Memory: {info.get('gpu_memory_gb', 'N/A')} GB")
print(f"CPU Threads: {info['cpu_count']}")
print(f"PyTorch: {info['torch_version']}")

# Example output on GPU:
# Device: cuda
# GPU: NVIDIA RTX 4090
# GPU Memory: 24.0 GB
# CPU Threads: 16
# PyTorch: 2.0.0

# Example output on Apple Silicon:
# Device: mps
# GPU: Apple Silicon (M3)
# GPU Memory: unified memory
# CPU Threads: 8
# PyTorch: 2.1.0
```

### 2.3 Full Pipeline Profiling

```bash
# Profile entire pipeline
python scripts/profile_performance.py \
  --device cuda \
  --output results/profile_cuda.json \
  --iterations 10

# Results:
# {
#   "device": "cuda",
#   "timestamp": "2025-11-02T10:30:00",
#   "components": {
#     "depth_estimation": {"latency_ms": 45.2, "memory_mb": 892, ...},
#     "hand_tracking": {"latency_ms": 15.3, "memory_mb": 234, ...},
#     "tracking": {"latency_ms": 5.1, "memory_mb": 50, ...},
#     "semantic": {"latency_ms": 120.5, "memory_mb": 500, ...},
#     "qa": {"latency_ms": 450.2, "memory_mb": 1200, ...},
#   },
#   "total_latency_ms": 636.3,
#   "total_throughput_fps": 1.57,
# }
```

---

## Part 3: Benchmarking & Evaluation

### 3.1 Ego4D Adapter

```python
from orion.research.evaluation.ego4d_adapter import Ego4DAdapter

adapter = Ego4DAdapter("/path/to/ego4d/root")

# Get list of clips
clips = adapter.get_clips(split="val", limit=10)
# Returns: ["clip_001", "clip_002", ...]

# Load single clip
clip = adapter.load_clip("clip_001")

# clip contains:
# {
#   'video_path': '/path/to/clip_001/video.mp4',
#   'objects': [...],  # Ground truth objects
#   'actions': [...],  # Ground truth actions
#   'hand_interactions': [...],  # Ground truth interactions
# }

# Process and evaluate
for clip_id in clips:
    clip = adapter.load_clip(clip_id)
    video_path = clip['video_path']
    
    # Run Orion pipeline
    result = pipeline.process_video(video_path)
    
    # Compare with GT
    metrics = compute_metrics(result, clip)
    print(f"{clip_id}: mAP={metrics['detection_map']:.3f}, interaction_F1={metrics['interaction_f1']:.3f}")
```

### 3.2 ActionGenome Adapter

```python
from orion.research.evaluation.action_genome_adapter import ActionGenomeAdapter

adapter = ActionGenomeAdapter("/path/to/action_genome/root")

# Get videos
videos = adapter.get_videos(limit=5)

# Load video with causal ground truth
video = adapter.load_video("video_001")

# video contains:
# {
#   'video_path': '/path/to/video_001.mp4',
#   'objects': [...],
#   'actions': [...],
#   'causal_links': [  # Key for this dataset!
#       {'cause_action_id': 0, 'effect_action_id': 1},
#       {'cause_action_id': 1, 'effect_action_id': 2},
#   ]
# }

# Evaluate causal reasoning
for video_id in videos:
    video = adapter.load_video(video_id)
    result = pipeline.process_video(video['video_path'])
    
    # Compute CIS accuracy against GT causality
    cis_precision, cis_recall, cis_f1 = compute_cis_metrics(
        result.cis_matrix,
        video['causal_links'],
        threshold=0.5
    )
    print(f"{video_id}: CIS F1={cis_f1:.3f}")
```

### 3.3 Evaluation Pipeline

```python
from orion.research.evaluation.evaluator import EvaluationPipeline

evaluator = EvaluationPipeline(config)

# Run full evaluation on Ego4D
ego4d_metrics = evaluator.evaluate_on_ego4d(split="val", limit=50)

# Returns:
# {
#   'detection_map': 0.65,
#   'detection_ap50': 0.75,
#   'detection_ap75': 0.45,
#   'mota': 0.82,
#   'motp': 0.91,
#   'id_switches': 3,
#   'hand_object_interaction_f1': 0.71,
# }

# Run full evaluation on ActionGenome
ag_metrics = evaluator.evaluate_on_action_genome()

# Returns causal evaluation metrics
# {
#   'causal_precision': 0.68,
#   'causal_recall': 0.73,
#   'causal_f1': 0.70,
#   'causal_auc_roc': 0.82,
# }
```

### 3.4 Ablation Studies

```bash
# Run ablations: 2D vs 3D, with/without hands, etc.
python scripts/run_ablation.py \
  --dataset ego4d \
  --output results/ablation_results.json

# Generates comparisons:
# - 2D CIS: causal_f1 = 0.55
# - 3D CIS (no hands): causal_f1 = 0.68 (+23% improvement)
# - 3D CIS (with hands): causal_f1 = 0.73 (+32% improvement)
# - Full system: causal_f1 = 0.73 (occlusion model adds small benefit)
```

**Ablation Output**:
```json
{
  "ablations": {
    "2d_cis": {
      "detection_map": 0.58,
      "causal_f1": 0.55,
      "interaction_f1": 0.48,
      "latency_ms": 420
    },
    "3d_cis_no_hand": {
      "detection_map": 0.62,
      "causal_f1": 0.68,
      "interaction_f1": 0.55,
      "latency_ms": 500
    },
    "3d_cis_with_hand": {
      "detection_map": 0.65,
      "causal_f1": 0.73,
      "interaction_f1": 0.71,
      "latency_ms": 530
    }
  },
  "improvements": {
    "3d_vs_2d_causal_f1": "+32.7%",
    "hand_bonus_interaction_f1": "+29.1%",
    "occlusion_detection_id_switches": "-38%"
  }
}
```

---

## Part 4: Using Research Toolkit in Practice

### Scenario 1: Debug Hand Detection Failures

```python
# Problem: Hand detection is failing on some frames

from orion.research.debug import FrameLogger, DebugVisualizer
from orion.research.debug.inspector import DebugInspector

logger = FrameLogger("debug_hands/")
inspector = DebugInspector()

# Set breakpoint: detect when no hands found
def no_hands_detected(state):
    return len(state.get('hands', [])) == 0

inspector.set_breakpoint('hand_tracker', no_hands_detected)

# Process video
for frame_id, frame in enumerate(video_stream):
    hands = hand_tracker.detect(frame)
    
    if inspector.check_breakpoint('hand_tracker', {'hands': hands}):
        # Debug this frame
        print(f"No hands at frame {frame_id}")
        
        # Try lower confidence
        hand_tracker.confidence_threshold = 0.3
        hands_retry = hand_tracker.detect(frame)
        
        # Visualize
        hand_viz = DebugVisualizer.draw_hands(frame, hands_retry)
        cv2.imwrite(f"debug_hands/frame_{frame_id}_no_hands.png", hand_viz)
        
        # Log for analysis
        logger.log_frame(frame_id, {
            'no_hands_at_default': True,
            'hands_at_lower_threshold': len(hands_retry),
        })
    
    logger.log_frame(frame_id, {'hand_count': len(hands)})

logger.save_summary("hand_debug.json")
```

### Scenario 2: Compare Performance Across Devices

```bash
# Profile on CPU
python scripts/profile_performance.py --device cpu --output results/profile_cpu.json

# Profile on GPU
python scripts/profile_performance.py --device cuda --output results/profile_gpu.json

# Profile on MPS
python scripts/profile_performance.py --device mps --output results/profile_mps.json

# Generate comparison
python -c "
import json
from orion.research.profiling.reporter import generate_comparison_report

cpu = json.load(open('results/profile_cpu.json'))
gpu = json.load(open('results/profile_gpu.json'))
mps = json.load(open('results/profile_mps.json'))

report = generate_comparison_report({'CPU': cpu, 'GPU': gpu, 'MPS': mps})
print(report)
"

# Output:
# Device Performance Comparison
# ===========================
#
# Component: depth_estimation
# - CPU:  45.2 ms (1x baseline)
# - GPU:  5.3 ms (8.5x speedup)
# - MPS:  12.1 ms (3.7x speedup)
#
# Component: hand_tracking
# - CPU:  15.3 ms (1x baseline)
# - GPU:  2.1 ms (7.3x speedup)
# - MPS:  4.5 ms (3.4x speedup)
```

### Scenario 3: Validate CIS Formula on Ground Truth

```python
from orion.research.evaluation.metrics import compute_cis_metrics

# Load Ego4D video
video = ego4d_adapter.load_clip("clip_001")

# Run Orion
result = pipeline.process_video(video['video_path'])

# Compute CIS accuracy across thresholds
precisions = []
recalls = []
thresholds = np.linspace(0, 1, 11)

for threshold in thresholds:
    p, r, f1 = compute_cis_metrics(
        result.cis_matrix,
        video.get('causal_links', set()),
        threshold=threshold
    )
    precisions.append(p)
    recalls.append(r)
    print(f"Threshold {threshold:.1f}: precision={p:.3f}, recall={r:.3f}, F1={f1:.3f}")

# Plot precision-recall curve
import matplotlib.pyplot as plt
plt.plot(recalls, precisions, 'b-o')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("CIS Causal Link Detection")
plt.savefig("cis_pr_curve.png")
```

---

## Part 5: CLI Commands for Research

### Run Benchmarks
```bash
python scripts/run_benchmark.py --dataset ego4d --limit 20 --output results/bench.json
```

### Run Ablations
```bash
python scripts/run_ablation.py --dataset ego4d --output results/ablation.json
```

### Profile Performance
```bash
python scripts/profile_performance.py --device auto --output results/profile.json
```

### Generate Reports
```bash
python scripts/generate_report.py \
  --results results/bench.json \
  --output results/benchmark_report.md
```

### Extract Debug Video
```bash
python scripts/debug_video.py \
  --input video.mp4 \
  --output debug_output/
  --mode full  # Extract depth, hands, tracking, CIS
```

---

## Toolkit Summary Table

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| **DebugVisualizer** | Render depth, hands, tracking | Frame + model outputs | Annotated images |
| **FrameLogger** | Log per-frame state | Pipeline results | JSON + pickle |
| **HTMLDebugger** | Interactive frame browser | Frame logs + video | HTML viewer |
| **DebugInspector** | Breakpoint debugging | Condition + state | Console + debugger |
| **PerformanceProfiler** | Latency & memory | Component + inputs | JSON metrics |
| **EvaluationPipeline** | Benchmark on datasets | Video + GT | Metrics (mAP, F1, etc.) |
| **AblationRunner** | Compare configurations | Dataset + configs | Comparison tables |

---

## Next Steps

1. **Explore**: Use DebugVisualizer to visualize pipeline output
2. **Inspect**: Use FrameLogger to capture per-frame state
3. **Benchmark**: Run EvaluationPipeline on Ego4D
4. **Ablate**: Compare 2D vs 3D CIS performance
5. **Report**: Generate markdown report for paper

