# Orion Research: Evaluation Framework

This document describes the evaluation framework for comparing the Orion CIS+LLM knowledge graph construction method against baselines.

## Overview

The Orion research system implements a novel **two-stage causal inference** approach:

1. **Stage 1: Mathematical Causal Influence Score (CIS)**
   - Calculates spatial proximity, directed motion, temporal decay, and visual similarity
   - Prunes spurious correlations before LLM reasoning
   - Formula: `CIS = w1·f_prox + w2·f_motion + w3·f_temporal + w4·f_embedding`

2. **Stage 2: LLM-based Event Verification**
   - Only high-scoring CIS pairs are passed to Gemma 3 4B
   - LLM provides semantic labels and generates Cypher queries
   - Constrained verification task vs. open-ended reasoning

## Architecture

```
Video Input
    ↓
┌─────────────────────────────────────┐
│  Asynchronous Perception Engine     │
│  • Intelligent frame selection      │
│  • YOLO11m object detection         │
│  • OSNet Re-ID embeddings           │
│  • Motion tracking (velocity)       │
│  • FastVLM descriptions (async)     │
└─────────────────────────────────────┘
    ↓
Structured Perception Log
    ↓
┌─────────────────────────────────────┐
│  Semantic Uplift Engine             │
│  • HDBSCAN entity tracking          │
│  • State change detection           │
│  • CIS causal scoring               │
│  • LLM event composition            │
│  • Neo4j graph construction         │
└─────────────────────────────────────┘
    ↓
Knowledge Graph (CIS+LLM)
```

## Components

### 1. Motion Tracker (`src/orion/motion_tracker.py`)
- Tracks object centroids across frames
- Estimates velocity vectors using linear regression
- Detects directional movement patterns
- Essential for CIS `f_motion` component

### 2. Causal Inference Engine (`src/orion/causal_inference.py`)
- Implements the CIS mathematical scoring function
- Scores agent-patient pairs for each state change
- Filters candidates before LLM verification
- Components:
  - `f_prox`: Spatial proximity (inverse distance)
  - `f_motion`: Directed motion towards patient
  - `f_temporal`: Temporal decay function
  - `f_embedding`: Visual similarity bonus

### 3. Heuristic Baseline (`src/orion/evaluation/heuristic_baseline.py`)
- Receives the same perception log as main system
- Uses hand-crafted if/then rules:
  - **Proximity**: distance < 50px for 10 frames → `IS_NEAR`
  - **Containment**: bbox overlap > 95% → `IS_INSIDE`
  - **Simple Causal**: `IS_NEAR` + state change → `CAUSED`
- No ML or AI components for fair comparison

### 4. Evaluation Metrics (`src/orion/evaluation/metrics.py`)
- **Structural**: Precision, Recall, F1 for edges
- **Semantic**: Label accuracy, description richness
- **Causal**: True positive rate for causal links
- **Graph**: Density, average degree, connectivity

### 5. VSGR Benchmark Loader (`src/orion/evaluation/benchmarks/vsgr_loader.py`)
- Loads Video Scene Graph Recognition dataset
- Converts VSGR annotations to Orion format
- Batch evaluation across multiple clips
- Aggregates metrics for statistical significance

## Usage

### Single Video Evaluation

Compare CIS+LLM vs. Heuristic Baseline on a video:

```bash
python scripts/run_evaluation.py --mode video --video path/to/video.mp4
```

This will:
1. Run perception engine and save perception log
2. Build knowledge graph with CIS+LLM method
3. Build knowledge graph with heuristic baseline
4. Compare the two graphs and generate report

Output files:
- `evaluation_output/perception_log.json` - Raw perception data
- `evaluation_output/graph_cis_llm.json` - Our method's graph
- `evaluation_output/graph_heuristic.json` - Baseline graph
- `evaluation_output/comparison_report.json` - Detailed comparison

### VSGR Benchmark Evaluation

Evaluate on the full VSGR dataset:

```bash
python scripts/run_evaluation.py \
    --mode benchmark \
    --benchmark vsgr \
    --dataset-path /path/to/vsgr_dataset
```

Expected VSGR structure:
```
vsgr_dataset/
├── videos/
│   ├── clip_001.mp4
│   └── clip_002.mp4
├── annotations/
│   ├── clip_001.json
│   └── clip_002.json
└── metadata.json
```

Output:
- Per-clip graphs in `evaluation_output/clip_XXX/`
- Aggregated metrics in `evaluation_output/vsgr_evaluation_results.json`

### Python API

```python
from orion.evaluation import HeuristicBaseline, GraphComparator
from orion.evaluation.benchmarks import VSGRBenchmark

# Compare two graphs
comparator = GraphComparator()
comparator.load_from_json("method_a", "graph_a.json")
comparator.load_from_json("method_b", "graph_b.json")
comparator.print_summary()

# Evaluate on VSGR
benchmark = VSGRBenchmark("/path/to/vsgr")
clip = benchmark.get_clip("clip_001")
metrics = benchmark.run_evaluation("clip_001", predicted_graph)
```

## Evaluation Metrics

### Edge-Level Comparison

```python
{
  "edges": {
    "precision": 0.85,   # TP / (TP + FP)
    "recall": 0.78,      # TP / (TP + FN)
    "f1": 0.81           # Harmonic mean
  }
}
```

### Event-Level Comparison

```python
{
  "events": {
    "precision": 0.72,
    "recall": 0.68,
    "f1": 0.70
  }
}
```

### Causal Link Accuracy

```python
{
  "causal": {
    "precision": 0.79,   # Correct causal links / predicted causal links
    "recall": 0.65,      # Correct causal links / ground truth causal links
    "f1": 0.71
  }
}
```

## Expected Results

Based on our research hypothesis, we expect:

1. **CIS+LLM should outperform Heuristic Baseline** on:
   - Causal link precision (fewer false positives)
   - Semantic richness (more detailed event labels)
   - Edge F1 (better overall graph quality)

2. **Heuristic Baseline limitations**:
   - Brittle rules fail on edge cases
   - Cannot handle ambiguous situations
   - No semantic understanding (generic labels only)
   - High false positive rate for causal links

3. **CIS+LLM advantages**:
   - Mathematical filtering reduces spurious correlations
   - LLM provides rich semantic labels
   - Handles novel event types not explicitly programmed
   - Better precision through two-stage verification

## Configuration

### CIS Parameters (`src/orion/causal_inference.py`)

```python
class CausalConfig:
    proximity_weight = 0.45      # w1 in CIS formula
    motion_weight = 0.25         # w2 in CIS formula
    temporal_weight = 0.20       # w3
    embedding_weight = 0.10      # w4
    
    max_pixel_distance = 600.0   # Proximity cutoff
    temporal_decay = 4.0         # Temporal influence decay (seconds)
    min_score = 0.55             # Minimum CIS to pass to LLM
    top_k_per_event = 5          # Max agents per state change
```

### Heuristic Parameters (`src/orion/evaluation/heuristic_baseline.py`)

```python
class HeuristicConfig:
    proximity_distance_threshold = 50.0   # pixels
    proximity_duration_threshold = 10     # frames
    containment_overlap_threshold = 0.95  # 95% overlap
    causal_max_time_gap = 2.0            # seconds
```

## Interpreting Results

### High Precision, Lower Recall
- System is conservative (fewer false alarms)
- May miss some subtle causal relationships
- Good for high-confidence applications

### Lower Precision, High Recall
- System is aggressive (captures more events)
- Higher false positive rate
- May need additional filtering

### Balanced F1
- Good tradeoff between precision and recall
- Optimal for most use cases

### Label Accuracy
- Measures semantic quality
- Should be higher for CIS+LLM vs. heuristic
- Indicates value of LLM reasoning

## Troubleshooting

### HDBSCAN not available
```bash
pip install hdbscan
```

### OSNet not loading
The system will automatically fall back to ResNet50 if OSNet is unavailable. For optimal Re-ID performance, ensure `timm` is up to date.

### Ollama connection errors
Ensure Ollama is running:
```bash
ollama serve
```

And that `gemma3:4b` is available:
```bash
ollama pull gemma3:4b
```

### Neo4j connection issues
Neo4j is optional for evaluation. The system will work without it, exporting graphs to JSON instead.

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@software{orion_evaluation_2025,
  title={Orion: Two-Stage Causal Inference for Video Scene Graphs},
  author={Orion Research Team},
  year={2025},
  url={https://github.com/riddhimanrana/orion-research}
}
```

## Future Work

- [ ] Additional benchmarks (Action Genome, Visual Genome Video)
- [ ] Cross-dataset generalization experiments
- [ ] Ablation studies on CIS components
- [ ] Human evaluation of semantic richness
- [ ] Runtime performance benchmarks
- [ ] Active learning for weight tuning

## Contact

For questions about the evaluation framework, please open an issue on GitHub.
