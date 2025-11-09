# Developer Guide: Orion Research Backend Architecture

Complete guide for contributing to Orion, understanding the codebase structure, running tests, and using the research toolkit.

---

## Quick Start

### 1. Clone & Install

```bash
# Clone repository
git clone https://github.com/riddhimanrana/orion-research.git
cd orion-research

# Install in development mode
make install-dev

# For research/benchmarking
make install-research
```

### 2. Verify Installation

```bash
# Run quick test
make ci-local

# Or individually:
make lint          # Code quality
make type-check    # Type safety
make test          # Unit tests (CPU)
```

### 3. Run Demo

```bash
# Download sample video (if available)
make demo
```

---

## Project Structure Explained

### Backend Organization (Production Code)

```
orion/backend/
├── perception/       Phase 1: 3D Perception
│   ├── depth.py             DepthEstimator (ZoeDepth, MiDaS)
│   ├── hand_tracking.py     HandTracker (MediaPipe + 3D projection)
│   ├── occlusion.py         OcclusionDetector (depth-based masking)
│   ├── camera_intrinsics.py CameraIntrinsics, backprojection utils
│   └── types.py             Hand, DepthResult, PerceptionOutput
│
├── tracking/         Phase 2: Entity Tracking & Permanence
│   ├── bayesian.py          BayesianEntityBelief (posterior tracking)
│   ├── tracker_3d.py        EntityTracker3D (Hungarian matching)
│   ├── permanence.py        ObjectPermanenceTracker (re-identification)
│   └── types.py             TrackingResult, EntityTrajectory
│
├── semantic/         Phase 3: Causal Reasoning & Scene Graphs
│   ├── context.py           ContextDetector (scene classification)
│   ├── causal_scorer.py     CausalInfluenceScorer3D (CIS formula)
│   ├── scene_graph.py       DenseSceneGraph, SparseSceneGraph
│   └── types.py             SemanticEntity, CausalLink
│
├── qa/               Phase 4: Question Answering
│   ├── question_classifier.py  QuestionClassifier
│   ├── qa_engine.py            VideoQAEngine
│   ├── clip_extractor.py       VideoClipExtractor (ffmpeg)
│   └── types.py                Answer, QAResult
│
└── pipeline.py       Main orchestrator (VideoPipeline)
```

### Research Toolkit (Debug & Analysis)

```
orion/research/
├── debug/
│   ├── visualizer.py        Render depth, hands, tracking
│   ├── frame_logger.py      Per-frame state logging
│   ├── inspector.py         Interactive debugging
│   └── html_debugger.py     Web-based frame inspector
│
├── evaluation/
│   ├── ego4d_adapter.py     Ego4D dataset interface
│   ├── action_genome_adapter.py  AG dataset interface
│   ├── metrics.py           CIS metrics, tracking metrics
│   └── evaluator.py         End-to-end evaluation pipeline
│
└── profiling/
    ├── profiler.py          Latency & memory profiling
    ├── hardware_info.py     Device detection (GPU/CPU/MPS)
    └── reporter.py          Report generation
```

---

## Device Support & Testing

### Supported Platforms

| Platform | Device | Status | Test Target |
|----------|--------|--------|-------------|
| Ubuntu 22.04 | CPU | ✅ Fully tested | `test-ubuntu-cpu` |
| Ubuntu 22.04 | CUDA GPU | ✅ Fully tested | `test-ubuntu-gpu` |
| macOS (Intel/M1/M2) | MPS | ✅ Tested | `test-macos-mps` |
| Windows 10/11 | CPU | ✅ Tested | `test-windows-cpu` |

### Device Auto-Detection

```python
# The system auto-detects best device
from orion.utils.device import detect_device, get_device_info

device = detect_device()  # Returns "cuda", "mps", or "cpu"
info = get_device_info()  # {'device': 'cuda', 'gpu_memory_gb': 16, ...}
```

### Running Tests on Specific Device

```bash
# CPU (default)
pytest orion/tests/ -v --device cpu

# GPU
pytest orion/tests/ -v --device cuda

# MPS (Apple Silicon)
pytest orion/tests/ -v --device mps
```

### GitHub Actions CI

All tests run automatically on **every push**:
- **Ubuntu CPU**: Tests perception, tracking, semantic, QA
- **Ubuntu GPU**: CUDA-specific tests
- **macOS MPS**: Apple Silicon tests
- **Windows CPU**: Windows compatibility

View results: GitHub Actions → Workflows → CI/CD

---

## Configuration System

### Hierarchical Configuration

```yaml
# config/default.yaml - Base configuration
perception:
  depth:
    model_name: "zoe"
    input_size: 384
  hand_tracking:
    enable: true

semantic:
  cis:
    temporal_weight: 0.30
    spatial_weight: 0.44

# config/profiles/demo.yaml - Override for demo
perception:
  depth:
    enable: true  # Fast path, good quality
  hand_tracking:
    max_hands: 2
```

### Loading Configuration

```python
from orion.utils.config_loader import load_config

# From YAML
config = load_config("config/profiles/demo.yaml")

# Programmatically
from orion.backend.config import PipelineConfig
config = PipelineConfig(
    input_video_path="video.mp4",
    output_dir="results/",
    depth=DepthConfig(model_name="zoe"),
    hand_tracking=HandTrackingConfig(enable=True),
)
```

---

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/my-new-feature
```

### 2. Make Changes & Test Locally

```bash
# Run linting & type checking
make lint
make type-check

# Run affected tests
pytest orion/tests/test_perception.py -v

# Or run all tests
make test
```

### 3. Format Code

```bash
make format
```

### 4. Commit & Push

```bash
git add .
git commit -m "Add feature: description"
git push origin feature/my-new-feature
```

### 5. GitHub Actions Will:
- ✅ Run lint checks
- ✅ Run type checking
- ✅ Run tests on all platforms
- ✅ Upload coverage
- ✅ Build package

If all pass, you're good to submit a PR!

---

## Using the Research Toolkit

### A. Debug Visualizations

```python
from orion.research.debug.visualizer import DebugVisualizer
import cv2

# Render depth map
depth_map = perceiver.estimate_depth(frame)
depth_colored = DebugVisualizer.render_depth_map(depth_map)
cv2.imshow("Depth", depth_colored)

# Draw hand tracking
hands = hand_tracker.detect(frame)
annotated = DebugVisualizer.draw_hands(frame, hands)

# Draw entity tracking
entities = tracker.track(annotated)
annotated = DebugVisualizer.draw_tracking(annotated, entities)
```

### B. Frame-by-Frame Logging

```python
from orion.research.debug.frame_logger import FrameLogger

logger = FrameLogger("debug_output/")

for frame_id, frame in enumerate(video_stream):
    perception = perceiver.process_frame(frame)
    tracking = tracker.track(perception)
    semantic = semantics.process(tracking)
    
    # Log this frame's data
    logger.log_frame(frame_id, {
        'timestamp': frame_id / fps,
        'depth_map': perception.depth_map,
        'hands': perception.hands,
        'entities': tracking.entities,
        'cis_matrix': semantic.cis_matrix,
    })

# Save summary
logger.save_summary("frame_log.json")
```

### C. Interactive HTML Debugger

```python
from orion.research.debug.html_debugger import HTMLDebugger

debugger = HTMLDebugger("debug_output/")
debugger.generate_viewer(frame_logs, video_path="video.mp4")

# Open in browser: open debug_output/debug_viewer.html
```

**Features**:
- 🎬 Video timeline slider
- 📊 Depth map heatmap
- 🖐️ Hand landmark overlay
- 📍 Entity tracking boxes
- 📋 Frame metadata inspector
- 🔗 JSON state viewer

### D. Interactive Breakpoint Inspector

```python
from orion.research.debug.inspector import DebugInspector

inspector = DebugInspector(paused=False)

# Set conditional breakpoint
def check_low_confidence(state):
    return state.get('depth_confidence', 1.0) < 0.3

inspector.set_breakpoint('depth_estimator', check_low_confidence)

# During pipeline:
if inspector.check_breakpoint('depth_estimator', depth_result):
    print(f"Breakpoint triggered! Confidence: {depth_result['confidence']}")
    import pdb; pdb.set_trace()  # Drop into debugger
```

---

## Running Benchmarks

### 1. Profile Performance (Latency & Memory)

```bash
# Auto-detect device
make profile

# Specific device
python scripts/profile_performance.py --device cuda --output results/profile.json
```

**Outputs**:
- `results/profile.json`: Latency breakdown per component
- `results/profile_report.md`: Human-readable report

### 2. Benchmark on Ego4D

```bash
# Download Ego4D subset (if configured)
make benchmark-ego4d

# Or manually:
python scripts/run_benchmark.py \
  --dataset ego4d \
  --split val \
  --limit 10 \
  --output results/ego4d_results.json
```

**Outputs**:
- Detection mAP, MOTA (tracking accuracy)
- Causal F1 score
- Interaction detection precision/recall
- Latency per component

### 3. Run Ablation Studies

```bash
make ablation

# Or manually:
python scripts/run_ablation.py \
  --dataset ego4d \
  --output results/ablation_results.json

# Generates comparisons:
# - 2D CIS vs 3D CIS (expect 15-25% F1 improvement)
# - With/without hand signals (expect 20-30% improvement)
# - With/without occlusion detection (expect 30-40% ID switch reduction)
```

**Report Generation**:
```bash
python scripts/generate_report.py \
  --results results/ablation_results.json \
  --output results/ablation_report.md
```

---

## Adding a New Component

### Example: Add a New Hand Pose Classification Model

**1. Create types** (`orion/backend/perception/types.py`):

```python
from dataclasses import dataclass

@dataclass
class HandPose:
    label: str  # "OPEN", "CLOSED", "PINCH", "POINTING"
    confidence: float
    action: str = "IDLE"  # "GRASPING", "RELEASING", "MOVING"
```

**2. Implement component** (`orion/backend/perception/hand_pose.py`):

```python
from orion.backend.base import PerceptionModule

class HandPoseClassifier(PerceptionModule):
    def __init__(self, model_name: str = "mediapipe"):
        self.model_name = model_name
    
    def process_frame(self, hand_landmarks: np.ndarray) -> HandPose:
        """Classify hand pose from 21 landmarks"""
        # Implementation
        pass
    
    def get_config(self) -> Dict:
        return {"model_name": self.model_name}
```

**3. Add unit test** (`orion/tests/test_hand_pose.py`):

```python
import pytest
from orion.backend.perception.hand_pose import HandPoseClassifier

@pytest.fixture
def classifier():
    return HandPoseClassifier()

def test_hand_pose_classification(classifier):
    # Mock hand landmarks (21, 3)
    mock_landmarks = np.random.rand(21, 3)
    pose = classifier.process_frame(mock_landmarks)
    
    assert pose.label in ["OPEN", "CLOSED", "PINCH", "POINTING"]
    assert 0 <= pose.confidence <= 1
```

**4. Integrate** (`orion/backend/perception/__init__.py`):

```python
from .hand_pose import HandPoseClassifier

__all__ = ["HandPoseClassifier", ...]
```

**5. Update pipeline** (`orion/backend/pipeline.py`):

```python
class VideoPipeline:
    def __init__(self, config):
        # ... existing code
        self.hand_pose_classifier = HandPoseClassifier(config.hand_pose_model)
    
    def process_frame(self, frame):
        # ... detect hands
        hand_poses = [
            self.hand_pose_classifier.process_frame(hand.landmarks_3d)
            for hand in hands
        ]
        # ... use poses in tracking/semantic modules
```

**6. Add to configuration** (`config/default.yaml`):

```yaml
perception:
  hand_pose:
    model_name: "mediapipe"
    enable: true
```

---

## Debugging Common Issues

### Issue: GPU Out of Memory

```bash
# Solution 1: Reduce batch size
python -m orion.cli --batch-size 1 --video video.mp4

# Solution 2: Use CPU
python -m orion.cli --device cpu --video video.mp4

# Solution 3: Profile memory usage
python scripts/profile_performance.py --device cuda --profile-memory
```

### Issue: Slow on macOS

```bash
# Use MPS (Metal Performance Shaders) if available
pytest --device mps

# Check device
from orion.utils.device import get_device_info
print(get_device_info())  # Should show MPS
```

### Issue: Hand Detection Failing

```python
# Debug with visualizer
from orion.research.debug.visualizer import DebugVisualizer

hand_tracker = HandTracker()
hands = hand_tracker.detect(frame)

if not hands:
    print(f"No hands detected. Frame min/max: {frame.min()}/{frame.max()}")
    print(f"Hand confidence threshold: {hand_tracker.confidence_threshold}")
    
    # Lower threshold and retry
    hand_tracker.confidence_threshold = 0.3
    hands = hand_tracker.detect(frame)

# Visualize
annotated = DebugVisualizer.draw_hands(frame, hands)
cv2.imshow("Hands", annotated)
```

### Issue: Type Checking Errors

```bash
# Run mypy to see type errors
make type-check

# Fix: Add type hints to function signatures
def process_frame(self, frame: np.ndarray) -> PerceptionResult:
    ...

# Or add type stub for third-party library
# Create stubs/ directory with .pyi files
```

---

## Performance Optimization Tips

### 1. Profile Before Optimizing

```bash
# Identify bottleneck
make profile

# Look at latency breakdown:
# depth_estimation: 45ms (bottleneck)
# hand_tracking: 15ms
# tracking: 5ms
```

### 2. Reduce Model Size

```python
# Use lightweight depth model
config.depth.model_name = "midas_small"  # vs "zoe" (500MB)

# Use quantized hand detector (future work)
config.hand_tracking.quantized = True
```

### 3. Enable Multi-Processing

```python
# Process multiple frames in parallel
from concurrent.futures import ProcessPoolExecutor

executor = ProcessPoolExecutor(max_workers=4)
futures = [executor.submit(process_frame, frame) for frame in frames]
results = [f.result() for f in futures]
```

### 4. Cache Repeated Computations

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_scene_embedding(scene_context: str):
    # Expensive CLIP embedding
    return clip_model.encode(scene_context)
```

---

## Contributing

### Code Style

We use:
- **Black** for formatting (line length: 120)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format before committing
make format

# Verify code quality
make ci-local
```

### Documentation

Add docstrings following Google style:

```python
def process_video(video_path: str, config: Dict) -> VideoResult:
    """
    Process video through full Orion pipeline.
    
    Args:
        video_path: Path to input video file
        config: Configuration dictionary with keys:
            - depth_model: Depth model to use ("zoe" or "midas")
            - device: "cuda", "cpu", or "mps"
    
    Returns:
        VideoResult containing perception, tracking, semantic, QA results
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If GPU out of memory
    
    Example:
        >>> result = process_video("video.mp4", {"device": "cuda"})
        >>> print(f"Detected {len(result.entities)} entities")
    """
```

### Testing

Aim for >80% coverage:

```bash
# Check coverage
make test

# View detailed coverage report
open htmlcov/index.html
```

---

## Resources

- 📖 **Phase Documentation**: See `docs/PHASE_*.md`
- 🏗️ **Architecture**: See `docs/SYSTEM_ARCHITECTURE_AND_STRUCTURE.md`
- 🔧 **API Reference**: Generated at `docs/api/` (run `make docs`)
- 🚀 **Research Toolkit**: See `docs/RESEARCH_TOOLKIT.md`
- 📊 **Benchmarks**: Results in `research/results/benchmarks/`

---

## Getting Help

- **Issues**: https://github.com/riddhimanrana/orion-research/issues
- **Discussions**: https://github.com/riddhimanrana/orion-research/discussions
- **Email**: riddhiman.rana@gmail.com

