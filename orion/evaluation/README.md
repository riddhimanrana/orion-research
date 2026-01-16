# Orion Evaluation Suite (SGG / SGA)

This folder contains a clean, standalone evaluation pipeline for:

- **SGG** on **PVSG** (Recall@K and mean Recall@K).
- **SGA** on **Action Genome** (future-frame Recall@K and mean Recall@K).

The design is intentionally modular:

- `core/` defines shared data types, metrics, and a runner.
- `pvsg/` provides PVSG loading + adapter + evaluator.
- `action_genome/` provides Action Genome loading + adapter + SGA evaluator.
- `scripts/evaluation/*.py` are the runnable entrypoints.

## Prediction file format (JSONL)

Both evaluators expect **prediction JSONL** with one line per frame:

```json
{
  "video_id": "vid_0001",
  "frame_index": 12,
  "relations": [
    {"subject_id": 0, "predicate": "on", "object_id": 1, "score": 0.83}
  ]
}
```

Only relations are required for evaluation. Object lists are optional (for future IoU-based matching).

## Plan summary

1. Load dataset annotations and convert to `VideoGraph` (frame → relations).
2. Load predictions from JSONL.
3. Compute R@K and mR@K for each frame, then average.
4. Write summary JSON artifacts for papers or tables.

See `scripts/evaluation/README.md` for full CLI usage and dataset layout details.
