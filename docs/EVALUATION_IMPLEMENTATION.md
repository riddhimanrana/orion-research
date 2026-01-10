# PVSG + ActionGenome Evaluation Implementation Guide

## Overview

This guide implements the plan discussed with your team for improving Orion's evaluation on video scene understanding datasets.

**Key Decisions:**
- **Paper version**: DINOv3/Faster-RCNN + Gemini 3.5-Flash (stronger results for publication)
- **Lightweight version**: YOLO-World + FastVLM (faster, lower quality for deployment)
- **Primary metric**: Recall@K (matching HyperGLM baseline)
- **Datasets**: PVSG (scene graphs) + ActionGenome (scene graph anticipation)

## Completed Implementation

### 1. ✅ PVSG Scene Graph Evaluation (`orion/evaluation/pvsg_evaluator.py`)

**What it does:**
- Loads PVSG dataset with 4 video sources (Ego4D, EpicKitchen, VidOR, etc.)
- Evaluates predicted scene graphs against ground truth
- Implements Recall@K, Precision, F1-Score metrics

**Key Classes:**
- `SceneGraphTriplet`: Represents (subject, predicate, object) relationship
- `PVSGEvaluator`: Main evaluation engine
- `PVSGEvaluationResult`: Per-frame results with all metrics

**Usage:**
```python
from orion.evaluation.pvsg_evaluator import PVSGEvaluator, SceneGraphTriplet

evaluator = PVSGEvaluator()

# Evaluate predictions
result = evaluator.evaluate_predictions(
    video_id="ego4d_sample",
    frame_id=0,
    predicted_triplets=[
        SceneGraphTriplet("person", "holding", "cup", confidence=0.95),
        SceneGraphTriplet("cup", "on", "table", confidence=0.87),
    ]
)

print(f"Recall@10: {result.recall_at_k[10]:.3f}")
print(f"Precision: {result.precision:.3f}")
print(f"F1-Score: {result.f1_score:.3f}")
```

**Metrics Explained:**
- **Recall@K**: What % of ground truth triplets appear in model's top-K predictions
  - Formula: (# matching GT triplets in top-K) / (total GT triplets)
  - Score range: 0.0 - 1.0 (higher is better)
  
- **Precision**: What % of model predictions are correct
  - Formula: (# correct predictions) / (total predictions)
  - Score range: 0.0 - 1.0 (higher is better)
  
- **F1-Score**: Harmonic mean of Recall@10 and Precision
  - Formula: 2 × (Precision × Recall) / (Precision + Recall)
  - Balances precision and recall

### 2. ✅ ActionGenome SGA Evaluator (`orion/evaluation/sga_evaluator.py`)

**What it does:**
- Evaluates Scene Graph Anticipation: predict future SGs given pruned video
- Given frames 0 to t (first 50%), predict frames t+1 onwards
- Measures ability to anticipate scene evolution

**Key Classes:**
- `AnticipationMetrics`: Results for anticipation task
- `ActionGenomeDataLoader`: Loads ActionGenome annotations
- `SGAEvaluator`: Anticipation evaluation engine

**Usage:**
```python
from orion.evaluation.sga_evaluator import SGAEvaluator

evaluator = SGAEvaluator()

# Anticipate future scene graphs
result = evaluator.evaluate_video_anticipation(
    video_id="video_123",
    prune_ratio=0.5,  # Given first 50% of frames
    predicted_future_sgs={
        50: [SceneGraphTriplet("person", "moving_to", "kitchen")],
        51: [SceneGraphTriplet("person", "entering", "kitchen")],
        # ... more future predictions
    }
)

print(f"Recall@10: {result.mean_recall_at_k[10]:.3f}")
print(f"Success Rate: {result.anticipation_success_rate:.3f}")
```

**Task:**
1. Given: Frames 0-49 (50% of 100-frame video)
2. Observed scene graphs for frames 0-49
3. Predict: Scene graphs for frames 50-99
4. Evaluate: How well predictions match ground truth

### 3. ✅ Gemini VLM Integration (`orion/backends/gemini_vlm.py`)

**What it does:**
- Replaces FastVLM with Gemini 3.5-Flash for paper evaluations
- Provides stronger scene understanding for better results
- Two backends: Gemini (paper) and FastVLM (lightweight)

**Key Classes:**
- `GeminiVLMBackend`: Uses Gemini API for scene understanding
- `FastVLMBackend`: Lightweight fallback for deployment
- `create_vlm_backend()`: Factory function

**Features:**
1. **Object Descriptions**: Rich text descriptions of detected objects
2. **Relationship Understanding**: Infers spatial relationships (on, near, holding, etc.)
3. **Scene Graph Anticipation**: Predicts future scene graphs

**Usage:**
```python
from orion.backends.gemini_vlm import create_vlm_backend
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "your_api_key"

# Create Gemini backend (paper version)
vlm = create_vlm_backend("gemini")

# Get object descriptions
descriptions = vlm.describe_objects(
    frame=numpy_frame,
    detections=[{"class_name": "person", "bbox": [10, 20, 100, 200]}],
    context="kitchen scene"
)

# Understand relationships
relationships = vlm.understand_relationships(frame, detections)
# Returns: [(person_idx, "holding", cup_idx), ...]

# Anticipate future scene graphs
future_sgs = vlm.anticipate_scene_graphs(
    frames_so_far=[frame0, frame1, ...],
    scene_graphs_so_far=[sg0, sg1, ...],
    num_future_frames=10
)
# Returns: List of anticipated scene graphs for next 10 frames
```

**API Setup:**
```bash
# Get API key from https://aistudio.google.com/apikey
export GOOGLE_API_KEY="sk-..."

# Install dependency
pip install google-generativeai pillow
```

### 4. ✅ End-to-End Pipeline (`orion/evaluation/scene_graph_pipeline.py`)

**What it does:**
- Unified pipeline: Video → Detections → Scene Graphs → Evaluation
- Supports two modes: PAPER (stronger) and LIGHTWEIGHT (faster)
- Integrates detection, VLM, and evaluation components

**Key Classes:**
- `FrameSceneGraph`: Scene graph for single frame
- `VideoSceneGraphs`: All scene graphs for a video
- `SceneGraphGenerator`: Main pipeline orchestrator
- `PipelineMode`: Enum for mode selection (PAPER or LIGHTWEIGHT)

**Usage:**
```python
from orion.evaluation.scene_graph_pipeline import create_pipeline, PipelineMode

# Create pipeline
pipeline = create_pipeline(PipelineMode.PAPER)  # For paper results

# Process video
video_sgs = pipeline.process_video(
    video_path="path/to/video.mp4",
    sample_rate=1,  # Process every frame
    context="kitchen scene"
)

# Evaluate on PVSG
pvsg_metrics = pipeline.evaluate_on_pvsg(video_sgs)
print(f"Recall@10: {pvsg_metrics['recall@10']:.3f}")
print(f"Mean Precision: {pvsg_metrics['mean_precision']:.3f}")

# Evaluate on ActionGenome (anticipation)
sga_metrics = pipeline.evaluate_on_actiongenome(video_sgs, prune_ratio=0.5)
print(f"Anticipation Success Rate: {sga_metrics['anticipation_success_rate']:.3f}")
```

**Pipeline Architecture:**

```
Video Input
    ↓
Frame Sampling (every N frames)
    ↓
Object Detection (YOLO-World or DINOv3)
    ↓
VLM Processing (Gemini or FastVLM)
    ├─ Object Descriptions
    ├─ Relationship Detection
    └─ Scene Graph Assembly
    ↓
Frame Scene Graphs
    ↓
Evaluation
    ├─ PVSG Evaluation (Recall@K, Precision)
    └─ ActionGenome SGA (Anticipation Accuracy)
```

### 5. ✅ Setup & Testing Script (`scripts/setup_evaluation_datasets.py`)

**What it does:**
- Validates dataset setup (PVSG, ActionGenome)
- Tests all evaluators and backends
- Provides comprehensive setup instructions

**Usage:**
```bash
# Check datasets
python scripts/setup_evaluation_datasets.py --check-datasets

# Test evaluators
python scripts/setup_evaluation_datasets.py --test-evaluators

# Benchmark pipeline
python scripts/setup_evaluation_datasets.py --benchmark-pipeline

# Run all checks
python scripts/setup_evaluation_datasets.py --all
```

## Implementation Status

### ✅ Completed

1. **PVSG Evaluator** (277 lines)
   - Loads PVSG JSON with scene graph annotations
   - Implements Recall@K, Precision, F1-Score
   - Evaluates batch predictions
   - Status: TESTED ✓

2. **ActionGenome SGA Evaluator** (198 lines)
   - Scene graph anticipation evaluation
   - Measures prediction quality for future frames
   - Implements success rates and per-frame metrics
   - Status: READY

3. **Gemini VLM Backend** (352 lines)
   - Object description generation
   - Spatial relationship understanding
   - Future scene graph anticipation
   - Fallback to FastVLM
   - Status: READY (requires API key)

4. **Scene Graph Pipeline** (386 lines)
   - Paper mode: DINOv3 + Gemini
   - Lightweight mode: YOLO-World + FastVLM
   - PVSG evaluation
   - ActionGenome SGA evaluation
   - Status: READY

5. **Setup & Testing** (325 lines)
   - Comprehensive test suite
   - Dataset validation
   - Setup instructions
   - Status: TESTED ✓

### 🔄 In Progress

1. **Faster-RCNN Detection Backend**
   - Not yet integrated
   - Can add as alternative to DINOv3
   - Would replace YOLO in paper mode

### 📋 TODO

1. **Download ActionGenome Dataset**
   ```bash
   mkdir -p datasets/ActionGenome
   # Follow instructions to download from OpenPVSG
   ```

2. **Set up Gemini API**
   ```bash
   # Get key from https://aistudio.google.com/apikey
   export GOOGLE_API_KEY="your_key"
   pip install google-generativeai pillow
   ```

3. **Run Baseline Evaluations**
   ```bash
   python scripts/setup_evaluation_datasets.py --all
   ```

4. **Add Faster-RCNN Backend** (Optional)
   - Create `orion/backends/faster_rcnn.py`
   - Implement detection interface
   - Integrate into pipeline

5. **Benchmark Against HyperGLM**
   - Compare Recall@K scores
   - Document results
   - Create comparison table

## Dataset Locations

```
/Users/yogeshatluru/orion-research/
├── datasets/
│   ├── PVSG/               # ✓ Already downloaded
│   │   ├── pvsg.json       # 3.9 MB scene graph annotations
│   │   ├── Ego4D/          # (videos not downloaded)
│   │   ├── EpicKitchen/    # (videos not downloaded)
│   │   └── VidOR/          # (videos not downloaded)
│   │
│   └── ActionGenome/       # ⏳ TODO: Download
│       └── (annotations and videos)
│
└── orion/
    └── evaluation/
        ├── pvsg_evaluator.py      # ✓ Scene graph evaluation
        ├── sga_evaluator.py       # ✓ Anticipation evaluation
        └── scene_graph_pipeline.py # ✓ End-to-end pipeline
```

## Configuration Files

### For PVSG Evaluation
```python
# File: orion/evaluation/pvsg_evaluator.py

# Default settings (no tuning needed)
evaluator = PVSGEvaluator()

# Override dataset path if needed
evaluator = PVSGEvaluator(
    pvsg_root=Path("path/to/PVSG")
)
```

### For Pipeline Execution
```python
# File: orion/evaluation/scene_graph_pipeline.py

# Paper mode (recommended for results)
pipeline = create_pipeline(PipelineMode.PAPER)

# Lightweight mode (for deployment)
pipeline = create_pipeline(PipelineMode.LIGHTWEIGHT)

# Process with custom settings
video_sgs = pipeline.process_video(
    video_path="video.mp4",
    sample_rate=1,      # Every frame
    context="kitchen"   # Optional scene context
)
```

## Key Differences from Current Setup

| Aspect | Current | New (PVSG+ActionGenome) |
|--------|---------|---------------------------|
| **Detection** | YOLO11, YOLO-World | YOLO-World (lightweight), DINOv3 (paper) |
| **VLM** | FastVLM, OLLAMA | FastVLM (lightweight), Gemini (paper) |
| **Evaluation** | Re-ID metrics | Recall@K, Precision, F1, Anticipation |
| **Datasets** | Custom videos | PVSG, ActionGenome, (later VSGR) |
| **Metrics** | Custom | Standard (Recall@K like HyperGLM) |
| **Release Strategy** | Single | Dual: Paper + Lightweight |

## Next Steps

1. **Immediate (This Week):**
   - Download ActionGenome dataset
   - Set up Gemini API key
   - Run baseline evaluations on PVSG

2. **Short Term (Next 2 Weeks):**
   - Benchmark paper pipeline (DINOv3 + Gemini) vs HyperGLM
   - Compare Recall@K scores
   - Optimize detection thresholds

3. **Medium Term (Next Month):**
   - Add Faster-RCNN backend if needed
   - Integrate with VSGR dataset
   - Create comprehensive benchmark table

4. **Long Term:**
   - Deploy lightweight version
   - Document differences
   - Release dual versions

## Testing Checklist

- [x] PVSG evaluator loads and works
- [x] ActionGenome SGA evaluator structure ready
- [x] Gemini VLM integration code complete
- [x] Scene graph pipeline implemented
- [x] Setup validation script works
- [ ] Gemini API key configured
- [ ] ActionGenome dataset downloaded
- [ ] Full end-to-end pipeline tested
- [ ] Baseline benchmarks completed
- [ ] Compared against HyperGLM

## References

- **PVSG Dataset**: https://huggingface.co/datasets/Jingkang/PVSG
- **ActionGenome**: https://github.com/jingkang50/OpenPVSG
- **Gemini API**: https://aistudio.google.com
- **HyperGLM Paper**: Related work on scene graph generation
- **VSGR Dataset**: (Videos to be sourced - noted for later)
