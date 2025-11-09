# System Architecture & Project Structure

Complete guide to Orion's filesystem, backend design, CI/CD pipelines, and research toolkit.

---

## Part 1: Project Filesystem Structure

```
orion-research/
├── .github/
│   ├── workflows/
│   │   ├── ci-test.yml                 # Main CI/CD pipeline (all platforms)
│   │   ├── performance-profile.yml     # Benchmark on GPU/CPU
│   │   ├── research-eval.yml           # Run Phase 5 benchmarks
│   │   └── lint-and-format.yml         # Code quality checks
│   ├── copilot-instructions.md
│   └── CONTRIBUTING.md
│
├── orion/
│   ├── __init__.py
│   ├── __version__.py                  # Version string for CI/CD tagging
│   ├── cli.py                          # Entry point (unchanged)
│   ├── settings.py                     # Config schema (Pydantic)
│   │
│   ├── backend/                        # ⭐ PRODUCTION BACKEND (Phase 1-4)
│   │   ├── __init__.py
│   │   ├── base.py                     # Base classes, interfaces
│   │   │
│   │   ├── perception/                 # Phase 1: 3D Perception
│   │   │   ├── __init__.py
│   │   │   ├── depth.py                # DepthEstimator (ZoeDepth)
│   │   │   ├── hand_tracking.py        # HandTracker (MediaPipe)
│   │   │   ├── occlusion.py            # OcclusionDetector
│   │   │   ├── camera_intrinsics.py    # CameraIntrinsics, backprojection
│   │   │   └── types.py                # Dataclasses (Hand, DepthResult, etc.)
│   │   │
│   │   ├── tracking/                   # Phase 2: Tracking & Permanence
│   │   │   ├── __init__.py
│   │   │   ├── bayesian.py             # BayesianEntityBelief
│   │   │   ├── tracker_3d.py           # EntityTracker3D (Hungarian matching)
│   │   │   ├── permanence.py           # ObjectPermanenceTracker
│   │   │   ├── metrics.py              # TrackingMetrics, ID-switch rate
│   │   │   └── types.py                # Dataclasses (TrackingResult, etc.)
│   │   │
│   │   ├── semantic/                   # Phase 3: CIS & Scene Graphs
│   │   │   ├── __init__.py
│   │   │   ├── context.py              # ContextDetector (INDOOR/OUTDOOR)
│   │   │   ├── causal_scorer.py        # CausalInfluenceScorer3D (5-component CIS)
│   │   │   ├── scene_graph.py          # DenseSceneGraph, SparseSceneGraph
│   │   │   ├── entities.py             # SemanticEntity, CausalLink, etc.
│   │   │   └── types.py                # Dataclasses (SemanticResult, etc.)
│   │   │
│   │   ├── qa/                         # Phase 4: QA & Visualization
│   │   │   ├── __init__.py
│   │   │   ├── question_classifier.py  # QuestionClassifier (regex + LLM)
│   │   │   ├── qa_engine.py            # VideoQAEngine (5 answer methods)
│   │   │   ├── clip_extractor.py       # VideoClipExtractor (ffmpeg)
│   │   │   ├── html_viewer.py          # HTMLViewer (interactive output)
│   │   │   └── types.py                # Dataclasses (Answer, QAResult, etc.)
│   │   │
│   │   ├── pipeline.py                 # VideoPipeline (orchestrates P1-P4)
│   │   └── config.py                   # Pydantic config schemas (phase-specific)
│   │
│   ├── research/                       # ⭐ RESEARCH TOOLKIT (Phase 5)
│   │   ├── __init__.py
│   │   ├── debug/
│   │   │   ├── __init__.py
│   │   │   ├── visualizer.py           # render_depth_map(), draw_hands(), draw_tracking()
│   │   │   ├── inspector.py            # DebugInspector (pause, step, inspect state)
│   │   │   ├── frame_logger.py         # FrameLogger (save per-frame debug data)
│   │   │   └── html_debugger.py        # HTMLDebugger (interactive web UI for analysis)
│   │   │
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   ├── ego4d_adapter.py        # Ego4DAdapter dataset interface
│   │   │   ├── action_genome_adapter.py# ActionGenomeAdapter dataset interface
│   │   │   ├── metrics.py              # compute_cis_metrics(), tracking_metrics()
│   │   │   ├── evaluator.py            # EvaluationPipeline (end-to-end)
│   │   │   └── ablation.py             # AblationRunner (2D vs 3D, etc.)
│   │   │
│   │   ├── profiling/
│   │   │   ├── __init__.py
│   │   │   ├── profiler.py             # PerformanceProfiler (latency, memory)
│   │   │   ├── hardware_info.py        # detect_gpu(), gpu_memory(), cpu_count()
│   │   │   └── reporter.py             # generate_profile_report()
│   │   │
│   │   └── reporting/
│   │       ├── __init__.py
│   │       ├── report_generator.py     # ReportGenerator (results → markdown/JSON)
│   │       └── visualization.py        # plot_metrics(), save_ablation_plots()
│   │
│   ├── managers/                       # Database & Index Managers
│   │   ├── __init__.py
│   │   ├── in_memory_index.py          # InMemoryIndex (primary index)
│   │   ├── memgraph_adapter.py         # MemgraphAdapter (optional)
│   │   └── cache.py                    # LRU cache for embeddings, depth maps
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config_loader.py            # load_yaml_config(), validate_config()
│   │   ├── logging.py                  # structured logging setup
│   │   ├── device.py                   # detect_device() → "cuda"/"mps"/"cpu"
│   │   └── video_io.py                 # VideoReader, VideoWriter wrappers
│   │
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py                 # pytest fixtures (sample video, mock data)
│       ├── test_perception.py          # Unit tests for Phase 1
│       ├── test_tracking.py            # Unit tests for Phase 2
│       ├── test_semantic.py            # Unit tests for Phase 3
│       ├── test_qa.py                  # Unit tests for Phase 4
│       └── integration/
│           ├── __init__.py
│           ├── test_e2e_pipeline.py    # Full pipeline test
│           └── test_video_qa.py        # QA end-to-end test
│
├── research/
│   ├── notebooks/
│   │   ├── 1_explore_depth_models.ipynb    # Comparison: ZoeDepth, MiDaS, DPT
│   │   ├── 2_hand_tracking_analysis.ipynb  # Hand pose accuracy analysis
│   │   ├── 3_cis_ablation_study.ipynb      # Visualize CIS component contributions
│   │   ├── 4_scene_graph_visualization.ipynb # Inspect causal links
│   │   └── 5_ego4d_evaluation.ipynb        # Benchmark results analysis
│   │
│   ├── datasets/
│   │   ├── ego4d/                      # Ego4D video clips + annotations
│   │   ├── action_genome/              # ActionGenome causal ground truth
│   │   ├── internal_labeled/           # In-house test videos with annotations
│   │   └── README.md                   # Dataset download & setup instructions
│   │
│   ├── results/
│   │   ├── benchmarks/
│   │   │   ├── ego4d_results.json      # Evaluation results
│   │   │   ├── action_genome_results.json
│   │   │   └── ablation_study.json     # 2D vs 3D, with/without hands, etc.
│   │   │
│   │   ├── ablations/
│   │   │   ├── cis_2d_vs_3d/          # Comparison plots
│   │   │   ├── hand_signal_impact/    # With/without hand bonus
│   │   │   └── occlusion_model/       # With/without occlusion detection
│   │   │
│   │   └── profiling/
│   │       ├── latency_breakdown.json  # Per-component latency (GPU/CPU/MPS)
│   │       ├── memory_usage.json       # Peak memory by phase
│   │       └── hardware_comparison.md  # Results across platforms
│   │
│   └── papers/
│       ├── draft.md                    # Conference paper draft
│       ├── figures/                    # Generated figures
│       └── supplementary.md            # Supplementary material
│
├── scripts/
│   ├── setup_env.sh                    # Install dependencies, download models
│   ├── download_ego4d.py               # Script to download Ego4D subset
│   ├── download_action_genome.py       # Script to download ActionGenome
│   ├── run_benchmark.py                # CLI: run Phase 5 evaluation
│   ├── run_ablation.py                 # CLI: run ablation studies
│   ├── profile_performance.py          # CLI: profile latency & memory
│   └── generate_report.py              # CLI: generate markdown report
│
├── config/
│   ├── default.yaml                    # Default config (all phases)
│   ├── profiles/
│   │   ├── demo.yaml                   # WACV demo config (fast, high quality)
│   │   ├── research.yaml               # Research config (all ablations enabled)
│   │   ├── mobile.yaml                 # Phase 3 roadmap: mobile optimization
│   │   └── real_time.yaml              # Phase 2 roadmap: streaming mode
│   │
│   ├── datasets/
│   │   ├── ego4d.yaml                  # Ego4D evaluation settings
│   │   └── action_genome.yaml          # ActionGenome evaluation settings
│   │
│   └── models/
│       ├── depth.yaml                  # ZoeDepth settings, fallback to MiDaS
│       ├── hand.yaml                   # MediaPipe hand tracking settings
│       └── llm.yaml                    # Small LLM (Llama2-7B, etc.)
│
├── docs/
│   ├── PHASE_1_README.md               # ✅ Created
│   ├── PHASE_2_README.md               # ✅ Created
│   ├── PHASE_3_README.md               # ✅ Created
│   ├── PHASE_4_README.md               # ✅ Created
│   ├── PHASE_5_README.md               # ✅ Created
│   ├── SYSTEM_ARCHITECTURE_AND_STRUCTURE.md # ⭐ THIS FILE
│   ├── INSTALLATION.md                 # Setup guide
│   ├── DEVELOPER_GUIDE.md              # For contributors
│   ├── RESEARCH_TOOLKIT.md             # How to use debugging/analysis tools
│   └── API_REFERENCE.md                # Auto-generated (pdoc or Sphinx)
│
├── .github/
│   └── copilot-instructions.md
│
├── pyproject.toml                      # Package definition (with GPU extras)
├── setup.py                            # Installation script
├── requirements.txt                    # Base dependencies
├── requirements-dev.txt                # Dev: testing, linting, profiling
├── requirements-research.txt           # Research: Ego4D, AG, notebooks
├── Makefile                            # Common commands (test, lint, build)
├── pytest.ini                          # pytest config
├── mypy.ini                            # Type checking config
├── .flake8                             # Linting config
├── pyproject.toml                      # Black, isort config
├── README.md                           # (unchanged, kept as-is)
└── LICENSE

```

---

## Part 2: Backend Module Design

### Principle: Layered Architecture with Clear Interfaces

```python
# orion/backend/base.py - Define all interfaces

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    """Base result type for all pipeline stages"""
    timestamp: float
    duration_ms: float
    metadata: Dict[str, Any]

class PerceptionModule(ABC):
    """Abstract base for perception components"""
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> ProcessingResult:
        """Process single frame, return typed result"""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return current configuration"""
        pass

class TrackingModule(ABC):
    """Abstract base for tracking components"""
    
    @abstractmethod
    def track_frame(self, entities, frame_id: int) -> ProcessingResult:
        """Update tracks, return updated entities"""
        pass

class SemanticModule(ABC):
    """Abstract base for semantic reasoning"""
    
    @abstractmethod
    def process(self, tracking_result: ProcessingResult) -> ProcessingResult:
        """Compute semantics (CIS, scene graph)"""
        pass

class QAModule(ABC):
    """Abstract base for question answering"""
    
    @abstractmethod
    def answer_question(self, question: str, semantic_result: ProcessingResult) -> str:
        """Answer question given semantic index"""
        pass
```

### Device Abstraction (CPU/GPU/MPS)

```python
# orion/utils/device.py

import torch
import numpy as np
from typing import Literal
from pathlib import Path

def detect_device() -> Literal["cuda", "mps", "cpu"]:
    """Auto-detect best device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"

def get_device_info() -> Dict[str, Any]:
    """Return detailed device information"""
    device = detect_device()
    info = {"device": device}
    
    if device == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["cuda_version"] = torch.version.cuda
    elif device == "mps":
        info["gpu_name"] = "Apple Silicon"
        info["gpu_memory_gb"] = "unified memory"
    
    info["cpu_count"] = torch.get_num_threads()
    info["torch_version"] = torch.__version__
    info["numpy_version"] = np.__version__
    
    return info

class TensorDevice:
    """Wrapper for device-agnostic tensor operations"""
    
    def __init__(self, device: str = None):
        self.device = device or detect_device()
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to device, handle MPS special cases"""
        if self.device == "mps":
            # MPS doesn't support some ops, fallback to CPU if needed
            return tensor.to(self.device)
        else:
            return tensor.to(self.device)
    
    def numpy_to_device(self, arr: np.ndarray) -> torch.Tensor:
        """Convert numpy to device tensor"""
        tensor = torch.from_numpy(arr).float()
        return self.to_device(tensor)
```

### Configuration Management (Pydantic)

```python
# orion/backend/config.py

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from pathlib import Path

class DepthConfig(BaseModel):
    """Depth estimation settings"""
    model_name: Literal["zoe", "midas"] = "zoe"
    model_path: Optional[Path] = None
    enable: bool = True
    input_size: int = 384
    confidence_threshold: float = 0.1
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    
    @validator("model_path", pre=True, always=True)
    def resolve_model_path(cls, v, values):
        if v is None and values.get("model_name") == "zoe":
            return Path.home() / ".cache" / "orion" / "zoe_depth.pt"
        return v

class HandTrackingConfig(BaseModel):
    """Hand tracking settings"""
    enable: bool = True
    confidence_threshold: float = 0.5
    max_hands: int = 2
    depth_to_3d: bool = True
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"

class CISConfig(BaseModel):
    """Causal Influence Score settings"""
    temporal_weight: float = Field(0.30, ge=0, le=1)
    spatial_weight: float = Field(0.44, ge=0, le=1)
    motion_weight: float = Field(0.21, ge=0, le=1)
    semantic_weight: float = Field(0.06, ge=0, le=1)
    
    temporal_tau_ms: int = 4000  # decay time constant
    spatial_max_distance_mm: int = 600
    
    hand_grasp_bonus: float = 0.30
    hand_touch_bonus: float = 0.15
    hand_near_bonus: float = 0.05
    
    @validator("*_weight")
    def weights_sum_to_one(cls, v, values):
        # Validation logic
        return v

class PipelineConfig(BaseModel):
    """Top-level pipeline config"""
    depth: DepthConfig = Field(default_factory=DepthConfig)
    hand_tracking: HandTrackingConfig = Field(default_factory=HandTrackingConfig)
    cis: CISConfig = Field(default_factory=CISConfig)
    
    input_video_path: Path
    output_dir: Path
    
    def to_dict(self) -> Dict:
        """Serialize to dict for logging"""
        return self.dict()
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "PipelineConfig":
        """Load from YAML file"""
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

---

## Part 3: CI/CD Pipeline on GitHub Actions

### `.github/workflows/ci-test.yml` - Main Pipeline

```yaml
name: Continuous Integration & Testing

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-and-format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      
      - name: Install dev dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Lint with flake8
        run: |
          flake8 orion/ --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 orion/ --count --exit-zero --max-complexity=10 --max-line-length=120
      
      - name: Type check with mypy
        run: |
          mypy orion/ --ignore-missing-imports --no-incremental
      
      - name: Format check with black
        run: |
          black --check orion/

  test-ubuntu-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      
      - name: Install dependencies
        run: |
          pip install -e ".[all]"
      
      - name: Run unit tests (CPU)
        run: |
          pytest orion/tests/ -v --cov=orion --cov-report=xml
        env:
          DEVICE: cpu
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  test-ubuntu-cuda:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.1.0-runtime-ubuntu22.04
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      
      - name: Install dependencies (with CUDA support)
        run: |
          pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
          pip install -e ".[all]"
      
      - name: Run unit tests (CUDA)
        run: |
          pytest orion/tests/ -v --device cuda
        env:
          CUDA_VISIBLE_DEVICES: 0

  test-macos-mps:
    runs-on: macos-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies (Apple Silicon)
        run: |
          pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
          pip install -e ".[all]"
      
      - name: Run unit tests (MPS)
        run: |
          pytest orion/tests/ -v --device mps
        continue-on-error: true  # MPS may not be available on CI runner

  test-windows-cpu:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      
      - name: Install dependencies (Windows)
        run: |
          pip install -e ".[all]"
      
      - name: Run unit tests (Windows CPU)
        run: |
          pytest orion/tests/ -v --device cpu
        shell: pwsh

  integration-test:
    runs-on: ubuntu-latest
    needs: [lint-and-format, test-ubuntu-cpu]
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      
      - name: Install dependencies
        run: |
          pip install -e ".[all]"
      
      - name: Download test video
        run: |
          mkdir -p data/test
          # Download a small test video (e.g., from a GitHub release)
          wget -O data/test/sample.mp4 https://github.com/riddhimanrana/orion-research/releases/download/v0.1.0/sample_video.mp4
      
      - name: Run end-to-end pipeline test
        run: |
          pytest orion/tests/integration/ -v
      
      - name: Run CLI demo
        run: |
          python -m orion.cli --input data/test/sample.mp4 --mode perception_3d --output results/
          python -m orion.cli analyze-with-qa --video data/test/sample.mp4 --questions "What did I hold?" --output-dir results/

  build-and-publish:
    runs-on: ubuntu-latest
    needs: [test-ubuntu-cpu, test-ubuntu-cuda, test-macos-mps, test-windows-cpu]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      
      - name: Build distribution
        run: |
          pip install build
          python -m build
      
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

### `.github/workflows/performance-profile.yml` - Benchmarking

```yaml
name: Performance Profiling

on:
  push:
    branches: [main]
  schedule:
    - cron: "0 0 * * 0"  # Weekly on Sunday

jobs:
  profile-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      
      - name: Install dependencies
        run: |
          pip install -e ".[research]"
      
      - name: Run performance profiler (CPU)
        run: |
          python scripts/profile_performance.py --device cpu --output results/profile_cpu.json
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: profile-results-cpu
          path: results/profile_cpu.json

  profile-gpu:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.1.0-runtime-ubuntu22.04
    steps:
      - uses: actions/checkout@v3
      - name: Install Python & dependencies
        run: |
          apt-get update && apt-get install -y python3.10 python3-pip
          pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
          pip install -e ".[research]"
      
      - name: Run performance profiler (GPU)
        run: |
          python scripts/profile_performance.py --device cuda --output results/profile_gpu.json
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: profile-results-gpu
          path: results/profile_gpu.json

  compare-results:
    runs-on: ubuntu-latest
    needs: [profile-cpu, profile-gpu]
    steps:
      - uses: actions/checkout@v3
      - name: Download artifacts
        uses: actions/download-artifact@v3
      
      - name: Generate comparison report
        run: |
          python scripts/compare_profiles.py \
            --cpu-profile profile-results-cpu/profile_cpu.json \
            --gpu-profile profile-results-gpu/profile_gpu.json \
            --output results/performance_report.md
      
      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: performance-report
          path: results/performance_report.md
```

---

## Part 4: Research Toolkit

### Debug Visualizer

```python
# orion/research/debug/visualizer.py

import cv2
import numpy as np
from typing import Tuple, List, Dict
from pathlib import Path

class DebugVisualizer:
    """Visualize internal state for debugging & analysis"""
    
    @staticmethod
    def render_depth_map(depth_map: np.ndarray, colormap: str = "turbo") -> np.ndarray:
        """
        Render depth map as colorized image
        
        Args:
            depth_map: (H, W) depth in mm or normalized [0,1]
            colormap: cv2 colormap name
        
        Returns:
            (H, W, 3) RGB image
        """
        # Normalize to [0, 255]
        depth_norm = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6) * 255).astype(np.uint8)
        
        # Apply colormap
        colormap_id = getattr(cv2, f"COLORMAP_{colormap.upper()}")
        colored = cv2.applyColorMap(depth_norm, colormap_id)
        
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def draw_hands(frame: np.ndarray, hands: List[Dict], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw hand landmarks and pose on frame
        
        Args:
            frame: (H, W, 3) RGB image
            hands: list of Hand objects with landmarks_2d, pose
        
        Returns:
            annotated frame
        """
        annotated = frame.copy()
        
        for hand in hands:
            landmarks = hand.landmarks_2d  # (21, 2)
            pose = hand.pose  # "OPEN", "CLOSED", "PINCH"
            
            # Draw skeleton
            HAND_CONNECTIONS = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # index
                # ... rest of connections
            ]
            
            for start, end in HAND_CONNECTIONS:
                pt1 = tuple(landmarks[start].astype(int))
                pt2 = tuple(landmarks[end].astype(int))
                cv2.line(annotated, pt1, pt2, color, 2)
            
            # Draw landmarks
            for i, lm in enumerate(landmarks):
                x, y = int(lm[0]), int(lm[1])
                cv2.circle(annotated, (x, y), 3, color, -1)
            
            # Add pose label
            cv2.putText(annotated, pose, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return annotated
    
    @staticmethod
    def draw_tracking(frame: np.ndarray, entities: List[Dict], show_ids: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and entity IDs
        """
        annotated = frame.copy()
        
        for entity in entities:
            bbox = entity['bbox_2d']  # (x1, y1, x2, y2)
            entity_id = entity['id']
            class_label = entity['class']
            
            color = (0, 255, 0)
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            if show_ids:
                label = f"ID:{entity_id} {class_label}"
                cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return annotated
```

### Frame-by-Frame Logger

```python
# orion/research/debug/frame_logger.py

import json
import pickle
from pathlib import Path
from typing import Dict, Any
import numpy as np

class FrameLogger:
    """Log per-frame debug data for post-hoc analysis"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_data = []
    
    def log_frame(self, frame_id: int, data: Dict[str, Any]):
        """
        Log frame-level data
        
        Data may include:
        - depth_map: (H, W) depth
        - hand_detections: list of Hand
        - tracked_entities: list of entities
        - cis_matrix: (n, n) CIS scores
        - scene_context: INDOOR/OUTDOOR
        """
        
        # Serialize carefully (numpy arrays → lists, etc.)
        serializable = {
            'frame_id': frame_id,
            'timestamp': data.get('timestamp'),
            'depth_map': data.get('depth_map').tolist() if isinstance(data.get('depth_map'), np.ndarray) else None,
            'hand_count': len(data.get('hands', [])),
            'entity_count': len(data.get('entities', [])),
            'scene_context': data.get('scene_context'),
            'cis_matrix_shape': data.get('cis_matrix').shape if isinstance(data.get('cis_matrix'), np.ndarray) else None,
        }
        
        self.frame_data.append(serializable)
    
    def save_summary(self, filename: str = "frame_log.json"):
        """Save JSON summary"""
        with open(self.output_dir / filename, 'w') as f:
            json.dump(self.frame_data, f, indent=2)
    
    def save_detailed(self, frame_id: int, data: Dict[str, Any]):
        """Save detailed per-frame pickle for full reconstruction"""
        with open(self.output_dir / f"frame_{frame_id:06d}.pkl", 'wb') as f:
            pickle.dump(data, f)
```

### Debug Inspector (Interactive Stepping)

```python
# orion/research/debug/inspector.py

class DebugInspector:
    """Interactive debugger for stepping through pipeline"""
    
    def __init__(self, paused: bool = False):
        self.paused = paused
        self.step_count = 0
        self.breakpoints = {}
    
    def set_breakpoint(self, component: str, condition: callable):
        """Set conditional breakpoint"""
        self.breakpoints[component] = condition
    
    def check_breakpoint(self, component: str, state: Dict) -> bool:
        """Check if breakpoint triggered"""
        if component in self.breakpoints:
            return self.breakpoints[component](state)
        return False
    
    def pause_on_frame(self, frame_id: int):
        """Pause on specific frame"""
        if self.paused and frame_id == self.target_frame:
            print(f"Paused at frame {frame_id}. Inspect state:")
            import pdb; pdb.set_trace()
    
    def inspect_entity(self, entity_id: str, history: List[Dict]):
        """Inspect entity trajectory and state"""
        print(f"\n=== Entity {entity_id} ===")
        for i, state in enumerate(history):
            print(f"Frame {i}: class={state['class']}, bbox={state['bbox']}, embedding_sim={state.get('embedding_sim', 'N/A')}")
```

### HTML Debugger (Interactive Web UI)

```python
# orion/research/debug/html_debugger.py

from pathlib import Path
import json

class HTMLDebugger:
    """Generate interactive HTML for exploring frame-level debug data"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
    
    def generate_viewer(self, frame_logs: List[Dict], video_path: str = None):
        """
        Generate interactive HTML viewer for frame-by-frame inspection
        
        Features:
        - Timeline slider to jump to frame
        - Depth map visualization
        - Hand detection overlay
        - Entity tracking visualization
        - CIS heatmap
        - JSON inspector for detailed state
        """
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Orion Debug Viewer</title>
            <style>
                body { font-family: monospace; background: #1e1e1e; color: #ddd; padding: 20px; }
                .container { max-width: 1400px; margin: 0 auto; }
                .controls { margin-bottom: 20px; }
                .frame-viewer { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .panel { background: #2d2d2d; padding: 10px; border-radius: 5px; }
                canvas { max-width: 100%; border: 1px solid #444; }
                #timeline { width: 100%; height: 60px; }
                .json-viewer { background: #1a1a1a; padding: 10px; border-radius: 3px; max-height: 300px; overflow-y: auto; font-size: 11px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Orion Debug Viewer</h1>
                
                <div class="controls">
                    <label>Frame:</label>
                    <input type="range" id="frameSlider" min="0" max="0" value="0">
                    <span id="frameLabel">0 / 0</span>
                    <button onclick="previousFrame()">← Prev</button>
                    <button onclick="nextFrame()">Next →</button>
                    <button onclick="togglePlayback()">▶ Play</button>
                </div>
                
                <div class="frame-viewer">
                    <div class="panel">
                        <h3>Video Frame</h3>
                        <video id="video" width="100%" controls></video>
                    </div>
                    
                    <div class="panel">
                        <h3>Depth Map</h3>
                        <canvas id="depthCanvas"></canvas>
                    </div>
                    
                    <div class="panel">
                        <h3>Hand Tracking</h3>
                        <canvas id="handCanvas"></canvas>
                    </div>
                    
                    <div class="panel">
                        <h3>Entity Tracking</h3>
                        <canvas id="trackingCanvas"></canvas>
                    </div>
                </div>
                
                <div class="panel" style="margin-top: 20px;">
                    <h3>Frame Metadata</h3>
                    <div class="json-viewer" id="metadataViewer"></div>
                </div>
            </div>
            
            <script>
                const frameData = """ + json.dumps(frame_logs) + """;
                let currentFrame = 0;
                let playing = false;
                
                function updateFrame(frameIdx) {
                    currentFrame = Math.min(frameIdx, frameData.length - 1);
                    const data = frameData[currentFrame];
                    
                    document.getElementById('frameLabel').textContent = `${currentFrame} / ${frameData.length - 1}`;
                    document.getElementById('frameSlider').value = currentFrame;
                    
                    // Update metadata viewer
                    document.getElementById('metadataViewer').textContent = JSON.stringify(data, null, 2);
                    
                    // Render visualizations
                    renderDepthMap(data);
                    renderHands(data);
                    renderTracking(data);
                }
                
                function nextFrame() {
                    updateFrame(currentFrame + 1);
                }
                
                function previousFrame() {
                    updateFrame(currentFrame - 1);
                }
                
                function togglePlayback() {
                    playing = !playing;
                    if (playing) {
                        const interval = setInterval(() => {
                            if (currentFrame >= frameData.length - 1) {
                                clearInterval(interval);
                                playing = false;
                            } else {
                                nextFrame();
                            }
                        }, 33);  // ~30 fps
                    }
                }
                
                document.getElementById('frameSlider').addEventListener('input', (e) => {
                    updateFrame(parseInt(e.target.value));
                });
                
                // Initialize
                document.getElementById('frameSlider').max = frameData.length - 1;
                updateFrame(0);
            </script>
        </body>
        </html>
        """
        
        with open(self.output_dir / "debug_viewer.html", 'w') as f:
            f.write(html)
        
        print(f"Debug viewer saved to {self.output_dir / 'debug_viewer.html'}")
```

---

## Part 5: Research Adapters

### Ego4D Adapter

```python
# orion/research/evaluation/ego4d_adapter.py

from pathlib import Path
from typing import Dict, List
import json

class Ego4DAdapter:
    """Interface to Ego4D dataset"""
    
    def __init__(self, ego4d_root: Path):
        self.ego4d_root = Path(ego4d_root)
        self.split_files = {
            'train': self.ego4d_root / 'annotations' / 'ego4d_train_clips.json',
            'val': self.ego4d_root / 'annotations' / 'ego4d_val_clips.json',
        }
    
    def get_clips(self, split: str = 'val', limit: int = None) -> List[str]:
        """Get list of clip IDs for split"""
        with open(self.split_files[split]) as f:
            data = json.load(f)
        
        clip_ids = [c['clip_id'] for c in data['clips']]
        return clip_ids[:limit] if limit else clip_ids
    
    def load_clip(self, clip_id: str) -> Dict:
        """Load clip with annotations"""
        video_path = self.ego4d_root / 'clips' / clip_id / 'video.mp4'
        annotations_path = self.ego4d_root / 'annotations' / clip_id / 'annotations.json'
        
        with open(annotations_path) as f:
            annotations = json.load(f)
        
        return {
            'video_path': str(video_path),
            'clip_id': clip_id,
            'annotations': annotations,
            'objects': annotations.get('objects', []),
            'actions': annotations.get('actions', []),
            'hand_interactions': annotations.get('hand_interactions', []),
        }
```

### ActionGenome Adapter

```python
# orion/research/evaluation/action_genome_adapter.py

class ActionGenomeAdapter:
    """Interface to ActionGenome causal dataset"""
    
    def __init__(self, ag_root: Path):
        self.ag_root = Path(ag_root)
    
    def get_videos(self, limit: int = None) -> List[str]:
        """Get list of video IDs"""
        video_dir = self.ag_root / 'videos'
        videos = sorted([v.stem for v in video_dir.glob('*.mp4')])
        return videos[:limit] if limit else videos
    
    def load_video(self, video_id: str) -> Dict:
        """Load video with causal GT"""
        video_path = self.ag_root / 'videos' / f'{video_id}.mp4'
        annotations_path = self.ag_root / 'annotations' / f'{video_id}.json'
        
        with open(annotations_path) as f:
            annotations = json.load(f)
        
        return {
            'video_path': str(video_path),
            'video_id': video_id,
            'objects': annotations['objects'],
            'actions': annotations['actions'],
            'causal_links': annotations['causality'],  # [(cause_action_id, effect_action_id), ...]
        }
```

---

## Part 6: Makefile & Common Tasks

```makefile
# Makefile

.PHONY: help install test lint format type-check profile benchmark clean docs

help:
	@echo "Orion Research: Common Commands"
	@echo ""
	@echo "  make install           Install dependencies"
	@echo "  make test              Run unit tests"
	@echo "  make lint              Lint code (flake8, mypy)"
	@echo "  make format            Format code (black, isort)"
	@echo "  make type-check        Run mypy type checking"
	@echo "  make profile           Profile performance"
	@echo "  make benchmark         Run benchmarks on Ego4D"
	@echo "  make demo              Run WACV demo"
	@echo "  make clean             Remove build artifacts"
	@echo "  make docs              Build documentation"

install:
	pip install -e ".[all]"
	python scripts/setup_env.sh

test:
	pytest orion/tests/ -v --cov=orion

lint:
	flake8 orion/ --max-line-length=120
	mypy orion/ --ignore-missing-imports

format:
	black orion/
	isort orion/

type-check:
	mypy orion/ --strict --ignore-missing-imports

profile:
	python scripts/profile_performance.py --device auto --output results/profile.json

benchmark:
	python scripts/run_benchmark.py --dataset ego4d --output results/benchmark.json

demo:
	python -m orion.cli analyze-with-qa --video data/test/sample.mp4 --questions "What did I hold?" --output-dir results/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info

docs:
	pdoc --html orion/ -o docs/api/
```

---

## Summary

**Filesystem**: Organized by phase (perception/, tracking/, semantic/, qa/) + research toolkit (debug/, evaluation/, profiling/)

**Backend**: Layered architecture with abstract base classes, device abstraction (CPU/GPU/MPS), Pydantic configs

**CI/CD**: GitHub Actions with tests on Ubuntu (CPU/CUDA), macOS (MPS), Windows (CPU)

**Research Toolkit**: Debug visualizer, frame logger, interactive HTML viewer, dataset adapters

**Tools**: Makefile for common tasks, performance profiler, ablation runner

This ensures **production-quality code** that's also **research-friendly** for deep debugging and scientific validation.

