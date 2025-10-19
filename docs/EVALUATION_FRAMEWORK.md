# Orion Evaluation Framework

## Overview

The Orion Evaluation Framework provides comprehensive tools for benchmarking Orion's **causal understanding**, **event detection**, and **relationship extraction** capabilities against standard video scene graph datasets.

This framework is designed for research paper evaluation, comparing Orion's performance against:
- **Action Genome**: Dense spatio-temporal scene graphs with actions
- **VSGR (Video Scene Graph)**: Multi-object tracking with relationships
- **PVSG (Panoptic Video Scene Graph)**: Panoptic segmentation + scene graphs

---

## Quick Start

### 1. Run Orion on a Benchmark Video

First, analyze your video with Orion to build the knowledge graph:

```bash
# Analyze a video from Action Genome
orion analyze /path/to/action_genome/videos/clip_001.mp4 \
    --output data/testing/clip_001 \
    --neo4j-uri bolt://localhost:7687
```

This creates entities, relationships, events, and causal links in Neo4j.

### 2. Run Benchmark Evaluation

Compare Orion's output against ground truth:

```bash
# Evaluate on Action Genome dataset
orion benchmark \
    --dataset action-genome \
    --data-dir /path/to/action_genome \
    --output-dir results/action_genome \
    --max-videos 10
```

This will:
1. Load ground truth annotations from Action Genome
2. Export Orion's knowledge graph from Neo4j
3. Align entities/relationships/events between prediction and ground truth
4. Compute precision/recall/F1 for relationships, events, and causal links
5. Generate JSON results and print summary tables

### 3. View Results

Results are saved in JSON format:

```bash
cat results/action_genome/action-genome_results.json
```

Example output:

```json
{
  "dataset": "action_genome",
  "num_videos": 10,
  "aggregate": {
    "relationships": {
      "precision": { "mean": 0.782, "std": 0.124 },
      "recall": { "mean": 0.691, "std": 0.143 },
      "f1": { "mean": 0.734, "std": 0.118 }
    },
    "events": {
      "precision": { "mean": 0.856, "std": 0.089 },
      "recall": { "mean": 0.743, "std": 0.112 },
      "f1": { "mean": 0.796, "std": 0.095 },
      "temporal_iou": { "mean": 0.621, "std": 0.154 }
    },
    "causal": {
      "precision": { "mean": 0.712, "std": 0.167 },
      "recall": { "mean": 0.634, "std": 0.189 },
      "f1": { "mean": 0.671, "std": 0.172 },
      "temporal_accuracy": { "mean": 0.893, "std": 0.087 }
    }
  }
}
```

---

## Metrics Explained

### 1. **Relationship Detection**

Measures how well Orion detects spatial and semantic relationships between entities.

- **Precision**: `TP / (TP + FP)` — What fraction of predicted relationships are correct?
- **Recall**: `TP / (TP + FN)` — What fraction of ground truth relationships were detected?
- **F1 Score**: Harmonic mean of precision and recall
- **Per-Type Breakdown**: Metrics for each relationship type (e.g., `on`, `holding`, `next_to`)

**What's measured:**
- Entity pair matching (by IoU threshold)
- Relationship predicate correctness

### 2. **Event/Action Detection**

Measures how well Orion detects events and their temporal boundaries.

- **Precision**: `TP / (TP + FP)` — What fraction of predicted events are correct?
- **Recall**: `TP / (TP + FN)` — What fraction of ground truth events were detected?
- **F1 Score**: Harmonic mean of precision and recall
- **Temporal IoU (tIoU)**: Overlap between predicted and ground truth event time ranges

**What's measured:**
- Event type classification accuracy
- Temporal boundary alignment (start/end frames)
- Involved entities overlap

### 3. **Causal Understanding**

Measures how well Orion infers causal relationships between events.

- **Precision**: `TP / (TP + FP)` — What fraction of predicted causal links are correct?
- **Recall**: `TP / (TP + FN)` — What fraction of ground truth causal links were detected?
- **F1 Score**: Harmonic mean of precision and recall
- **Temporal Accuracy**: Fraction of causal links with correct temporal ordering (cause before effect)

**What's measured:**
- Correct identification of cause-effect pairs
- Temporal ordering preservation
- Causal chain completeness

### 4. **Entity Matching**

Measures entity detection and tracking accuracy.

- **Precision**: `TP / (TP + FP)` — What fraction of predicted entities are correct?
- **Recall**: `TP / (TP + FN)` — What fraction of ground truth entities were detected?
- **Class Accuracy**: Fraction of matched entities with correct class labels

**What's measured:**
- Spatial IoU between predicted and ground truth bounding boxes
- Temporal overlap across frames
- Class label agreement

### 5. **Graph Edit Distance (GED)**

Approximate metric for overall graph similarity:

```
GED = |nodes_diff| + 0.5 * |edges_diff| + 0.3 * |events_diff|
```

Lower is better. Measures structural differences between predicted and ground truth graphs.

---

## Architecture

### Modules

1. **`benchmark_evaluator.py`**  
   Core evaluation logic:
   - `BenchmarkEvaluator`: Computes all metrics
   - `GroundTruthGraph`: Standardized ground truth representation
   - `PredictionGraph`: Standardized Orion output representation
   - `EvaluationMetrics`: Result dataclass

2. **`orion_adapter.py`**  
   Exports Orion's Neo4j knowledge graph to evaluation format:
   - `OrionKGAdapter`: Queries Neo4j, extracts entities/relationships/events/causal links
   - Handles temporal and spatial info extraction

3. **`ag_adapter.py`**  
   Converts Action Genome annotations to `GroundTruthGraph`:
   - Maps AG objects → entities
   - Maps AG relationships → standardized relationships
   - Maps AG actions → events
   - Infers causal links from temporal ordering

4. **`benchmark_runner.py`**  
   CLI tool that orchestrates evaluation:
   - Loads benchmark dataset
   - Runs evaluation on multiple videos
   - Aggregates metrics
   - Saves results to JSON

### Data Flow

```
┌─────────────────┐      ┌──────────────────┐
│ Action Genome   │──────▶│  GroundTruthGraph│
│ Annotations     │      │  (standardized)   │
└─────────────────┘      └──────────────────┘
                                    │
                                    │
                                    ▼
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│ Orion Neo4j KG  │──────▶│ PredictionGraph  │──────▶│ EvaluationMetrics│
└─────────────────┘      │  (standardized)   │      └─────────────────┘
                         └──────────────────┘
                                    │
                                    │
                                    ▼
                         ┌──────────────────┐
                         │BenchmarkEvaluator│
                         │  - Match entities │
                         │  - Align events   │
                         │  - Compute metrics│
                         └──────────────────┘
```

---

## Entity Matching Algorithm

Entities are matched using a weighted similarity score:

```python
score = 0.4 * class_match + 0.3 * temporal_iou + 0.3 * spatial_iou
```

Where:
- **class_match**: 1.0 if predicted and GT classes match, else 0.0
- **temporal_iou**: IoU of frame sets (predicted frames ∩ GT frames) / (predicted ∪ GT)
- **spatial_iou**: Average bbox IoU across common frames

Matching uses greedy assignment with threshold `iou_threshold` (default: 0.5).

---

## Event Matching Algorithm

Events are matched using:

```python
score = 0.5 * temporal_iou + 0.3 * type_match + 0.2 * entity_overlap
```

Where:
- **temporal_iou**: IoU of event time spans (start/end frames)
- **type_match**: 1.0 if event types match, else 0.0
- **entity_overlap**: Jaccard similarity of involved entities

Events match if `temporal_iou >= tiou_threshold` (default: 0.3).

---

## Causal Link Inference (Action Genome)

Since Action Genome doesn't explicitly annotate causal links, we infer them using heuristics:

1. **Temporal proximity**: Actions within 3 seconds are candidates
2. **Shared objects**: Actions involving the same objects are likely causal
3. **Causal patterns**:
   - `pick_up` → `put_down` (same object)
   - `open` → `enter` (same door)
   - `close` → `leave` (same room)

Confidence is set to 0.8 for inferred links.

---

## Command-Line Options

### `orion benchmark`

```
orion benchmark --dataset <dataset> --data-dir <path> [options]

Required:
  --dataset             Benchmark dataset: action-genome, vsgr, pvsg
  --data-dir           Path to dataset root directory

Optional:
  --output-dir         Results output directory (default: results/)
  --video-ids          Specific video IDs to evaluate
  --max-videos         Max number of videos (for quick testing)
  --iou-threshold      IoU threshold for entity matching (default: 0.5)
  --tiou-threshold     Temporal IoU threshold for events (default: 0.3)
```

### Example: Quick Test on 5 Videos

```bash
orion benchmark \
    --dataset action-genome \
    --data-dir ~/datasets/action_genome \
    --output-dir results/quick_test \
    --max-videos 5 \
    --iou-threshold 0.4
```

---

## Extending to Other Datasets

To add support for a new dataset (e.g., VSGR, PVSG):

1. **Create a loader** in `orion/evaluation/benchmarks/`:
   ```python
   # vsgr_loader.py
   class VSGRBenchmark:
       def __init__(self, dataset_root):
           # Load VSGR annotations
           pass
   ```

2. **Create an adapter** in `orion/evaluation/`:
   ```python
   # vsgr_adapter.py
   class VSGRAdapter:
       def convert_to_ground_truth(self, vsgr_data) -> GroundTruthGraph:
           # Convert VSGR to standardized format
           pass
   ```

3. **Update `benchmark_runner.py`**:
   ```python
   def _load_benchmark(self):
       if self.dataset_name == "vsgr":
           from orion.evaluation.benchmarks.vsgr_loader import VSGRBenchmark
           return VSGRBenchmark(str(self.data_dir))
   ```

---

## Research Paper Usage

### Reporting Results

For your research paper, report:

1. **Aggregate metrics** (mean ± std):
   - Relationship Detection: P/R/F1
   - Event Detection: P/R/F1 + Temporal IoU
   - Causal Understanding: P/R/F1 + Temporal Accuracy

2. **Comparison table**:
   ```
   | Method         | Rel F1 | Event F1 | Causal F1 |
   |----------------|--------|----------|-----------|
   | VidVRD         | 0.623  | 0.701    | —         |
   | TRACE          | 0.687  | 0.743    | —         |
   | Orion (Ours)   | 0.734  | 0.796    | 0.671     |
   ```

3. **Per-type breakdown** for key relationships (e.g., `holding`, `on`, `next_to`)

4. **Ablation studies**: Run with/without specific components (e.g., causal inference disabled)

### Generating Figures

See `orion/evaluation/visualize.py` (to be implemented) for:
- Confusion matrices (predicted vs GT relationship types)
- Temporal alignment plots (event boundaries)
- Precision-recall curves (at varying confidence thresholds)

---

## Troubleshooting

### Issue: "Cannot connect to Neo4j"

**Solution:** Ensure Neo4j is running and credentials are correct:
```bash
# Check Neo4j status
docker ps | grep neo4j

# Verify connection
orion status
```

### Issue: "Dataset not found"

**Solution:** Verify dataset directory structure matches expected format:
```
action_genome/
├── videos/
│   ├── clip_001.mp4
│   └── ...
├── annotations/
│   ├── person_bbox.pkl
│   └── object_bbox_and_relationship.pkl
└── metadata/
    └── object_classes.txt
```

### Issue: Low recall on relationships

**Possible causes:**
- IoU threshold too high → lower `--iou-threshold`
- Orion not detecting all entities → check perception settings
- Neo4j database incomplete → re-run `orion analyze`

### Issue: Low causal F1 score

**Possible causes:**
- Causal inference disabled in pipeline → check `orion/causal_inference.py`
- Ground truth has limited causal annotations → Action Genome uses inferred links
- Temporal windows too strict → adjust heuristics in `ag_adapter.py`

---

## Next Steps

1. **Run full evaluation** on Action Genome (all 10K clips)
2. **Add VSGR/PVSG support** by implementing loaders and adapters
3. **Generate visualizations** for paper figures
4. **Ablation studies**: Disable causal inference, adjust hyperparameters
5. **Comparison with baselines**: VidVRD, TRACE, Video Graph Transformer

---

## References

- **Action Genome**: [GitHub](https://github.com/JingweiJ/ActionGenome) | [Paper (CVPR 2020)](https://arxiv.org/abs/1912.06992)
- **VSGR**: [Paper](https://arxiv.org/abs/2004.11622)
- **PVSG**: [Paper](https://arxiv.org/abs/2206.01693)
- **Video Scene Graph Generation**: [Survey Paper](https://arxiv.org/abs/2107.07608)

---

**Author:** Orion Research Team  
**Version:** 1.0  
**Last Updated:** October 2025
