# Evaluation Scripts

This folder contains runnable evaluation scripts that wrap the new evaluation suite.

## 1) PVSG SGG evaluation

```powershell
python scripts/evaluation/eval_pvsg_sgg.py \
  --pvsg-root data/pvsg \
  --split test \
  --predictions results/pvsg_predictions.jsonl \
  --topk 20 50 100 \
  --output eval_outputs/pvsg_sgg_summary.json
```

## 2) Action Genome SGA evaluation

```powershell
python scripts/evaluation/eval_action_genome_sga.py \
  --ag-root data/action_genome \
  --split test \
  --predictions results/ag_predictions.jsonl \
  --fraction 0.9 \
  --topk 10 20 50 \
  --output eval_outputs/ag_sga_summary.json
```

## Prediction JSONL format

Each line corresponds to one frame. Evaluation is performed only on frames where predictions exist (GT ∩ Pred):

```json
{
  "video_id": "vid_0001",
  "frame_index": 12,
  "relations": [
    {"subject_id": 0, "predicate": "on", "object_id": 1, "score": 0.83}
  ]
}
```

## Files created in this batch

- `orion/evaluation/core/types.py` – shared data structures.
- `orion/evaluation/core/metrics.py` – R@K/mR@K utilities.
- `orion/evaluation/core/runner.py` – evaluation runner + JSONL loader.
- `orion/evaluation/pvsg/*` – PVSG loader/adapter/evaluator.
- `orion/evaluation/action_genome/*` – AG loader/adapter/evaluator.
- `scripts/evaluation/eval_pvsg_sgg.py` – PVSG evaluator CLI.
- `scripts/evaluation/eval_action_genome_sga.py` – AG evaluator CLI.
