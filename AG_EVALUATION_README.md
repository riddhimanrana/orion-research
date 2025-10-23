# Action Genome Evaluation Pipeline

Run Orion's full pipeline on Action Genome and evaluate scene graph generation.

## Setup

### 1. Download Action Genome Dataset

```bash
# Download Charades 480p videos (required source)
# From: https://prior.allenai.org/projects/charades
# Extract to: dataset/ag/videos/

# Download Action Genome annotations (Google Drive link in AG repo)
# Extract to: dataset/ag/annotations/
#   - object_bbox_and_relationship.pkl
#   - person_bbox.pkl
#   - frame_list.txt
```

### 2. Extract Frames (Optional - speeds up pipeline)

```bash
# Modify tools/dump_frames.py to limit to first 50 videos
# Then run:
cd dataset/ag
python ../../tools/dump_frames.py
# Extracts to: dataset/ag/frames/
```

## Evaluation Pipeline

```bash
# Step 1: Load AG benchmark and convert to GroundTruthGraphs
python scripts/1_prepare_ag_data.py

# Step 3: Run full Orion pipeline (detection + semantic graph generation)
python scripts/3_run_orion_ag_eval.py

# Step 4: Evaluate predictions with standard metrics
python scripts/4_evaluate_ag_predictions.py
```

## Output Files

- `data/ag_50/ground_truth_graphs.json` - 50 AG clips in GroundTruthGraph format
- `data/ag_50/results/predictions.json` - Orion-generated scene graphs
- `data/ag_50/results/metrics.json` - Evaluation results

## Metrics Computed

**Relationship Detection (Edge-level):**
- Precision, Recall, F1-Score

**Event Detection:**
- Precision, Recall, F1-Score

**Causal Inference:**
- Precision, Recall, F1-Score

**Entity Detection:**
- Jaccard Similarity

## GroundTruthGraph Format

```python
{
  "video_id": str,
  "entities": {
    "entity_id": {
      "entity_id": str,
      "class": str,
      "label": str,
      "frames": [int],
      "first_frame": int,
      "last_frame": int,
      "bboxes": {frame_id: [x,y,w,h]},
      "attributes": {}
    }
  },
  "relationships": [
    {
      "subject": entity_id,
      "object": entity_id,
      "predicate": str,  # e.g., "holding", "looking_at"
      "frame_id": int,
      "confidence": float
    }
  ],
  "events": [
    {
      "event_id": str,
      "type": str,
      "start_frame": int,
      "end_frame": int,
      "entities": [entity_id],
      "agent": entity_id,
      "patients": [entity_id]
    }
  ],
  "causal_links": [
    {
      "cause": event_id,
      "effect": event_id,
      "time_diff": float,
      "confidence": float
    }
  ]
}
```

## Using Orion Modules

The pipeline uses these existing Orion modules:
- `orion.evaluation.benchmarks.action_genome_loader` - Load AG dataset
- `orion.evaluation.ag_adapter` - Convert AG to GroundTruthGraph
- `orion.run_pipeline` - Full perception + semantic graph
- `orion.evaluation.benchmark_evaluator` - Standard evaluation
- `orion.evaluation.metrics` - Precision/Recall/F1 computation
