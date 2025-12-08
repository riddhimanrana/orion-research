# Video Scene Graph Evaluation Instructions

This guide explains how to extend Orion with the training and evaluation workflows needed to reproduce the HyperGLM Page 7 tables (Video Scene Graph Generation and Anticipation). You can implement the full Python stack before downloading datasets; dataset-specific code can be stubbed and filled in later.

---

## 1. Experiments We Aim to Reproduce

| Table | Task | Datasets | Notes |
| --- | --- | --- | --- |
| Table 2 | Scene Graph Anticipation (SGA) | Action Genome, VSGR | Input fraction `F` ∈ {0.3, 0.5, 0.7}; report `R@{10,20,50}` and `mR@{10,20,50}` |
| Table 3 | Scene Graph Generation (SGG) | PVSG, VSGR | Standard SGG (`R/mR@{20,50,100}`) |
| Table 4 | Scene Graph Anticipation (SGA) | Action Genome, VSGR | Focus on higher `F` (≥ 0.9) |

**Metrics:** Recall@K (`R@K`) and mean Recall@K (`mR@K`). These mirror the Action Genome evaluator already present in Orion.

---

## 2. Repository Components Ready for Use

The `orion/vsgg/` package is live and already wired into CLI utilities under `scripts/`:

```
orion/vsgg/
   datasets.py          # Action Genome + PVSG + VSGR loaders, synthetic fixtures
   metrics.py           # VideoRelationEvaluator with R/mR@K
   models.py            # Pairwise SGG head + temporal SGA forecaster
scripts/
   vsgg_train_sgg.py    # Train Recall/mRecall baselines
   vsgg_eval_sgg.py     # Evaluate checkpoints on any supported dataset
   vsgg_train_sga.py    # Train anticipation models with fraction F
   vsgg_eval_sga.py     # Evaluate anticipation checkpoints over fractions
```

### 2.1 Dataset helpers

- `FrameData` / `VideoSceneGraphSample` encapsulate frame-level annotations and whole videos.
- `ActionGenomeSceneGraphDataset` can emit per-frame samples or grouped videos; use `group_by_video=True` plus `min_frames_per_video`/`max_frames_per_video` to build SGA sequences. A `max_videos` limit keeps debugging runs lightweight.
- `PVSGSceneGraphDataset` and `VSGRSceneGraphDataset` parse JSON annotations (`annotations/<split>.json`) together with `metadata/object_classes.txt` and `metadata/predicate_classes.txt`. Pass `--max-videos` / `--max-frames-per-video` on the CLI to clip large corpora.
- `SyntheticSceneGraphDataset` still exists for smoke tests when real data is unavailable.

### 2.2 Metrics and models

- `metrics.VideoRelationEvaluator` mirrors the Action Genome evaluator and reports `R@K` and `mR@K` for any requested `topk` list.
- `models.PairwiseRelationModel` consumes pair features (`build_pair_feature`) for SGG, while `TemporalRelationForecaster` handles SGA sequences created from the grouped datasets.

### 2.3 Train/eval entry points

Use the new scripts directly (examples assume data has been placed in `data/`):

```powershell
# Train SGG baseline on PVSG
python scripts/vsgg_train_sgg.py --dataset pvsg --data-root data/pvsg --split train \
      --max-videos 50 --epochs 10 --eval

# Evaluate the checkpoint on VSGR validation split
python scripts/vsgg_eval_sgg.py --checkpoint checkpoints/vsgg_sgg.pt \
      --dataset vsgr --data-root data/vsgr --split val

# Train an SGA model on Action Genome with fraction 0.7
python scripts/vsgg_train_sga.py --dataset action_genome --data-root data/action_genome \
      --split train --fraction 0.7 --max-videos 100

# Evaluate SGA checkpoint for multiple fractions
python scripts/vsgg_eval_sga.py --checkpoint checkpoints/vsgg_sga.pt --dataset vsgr \
      --data-root data/vsgr --split val --fraction 0.5 --topk 10 20 50
```

Each CLI exposes `--max-videos`, `--max-frames-per-video`, and (for Action Genome) `--max-samples` so you can dry-run on a single GPU before downloading full datasets.

---

## 3. Environment & Dataset Preparation

### 3.1 Environment setup

1. Install Orion (editable mode keeps scripts in sync):

   ```powershell
   conda activate orion
   pip install -e .
   ```

2. (Optional) Run a quick loader smoke test. For example:

   ```powershell
   python - <<'PY'
   from orion.vsgg.datasets import ActionGenomeSceneGraphDataset
   ds = ActionGenomeSceneGraphDataset('data/action_genome', split='train', max_samples=8)
   print(f"Loaded {len(ds.samples)} samples, {ds.num_objects} objects, {ds.num_predicates} predicates")
   PY
   ```

### 3.2 Action Genome (AG)

Already partially supported via:

- `orion.data.action_genome.ActionGenomeFrameDataset`
- `orion.eval.action_genome.ActionGenomeRelationEvaluator`
- `scripts/eval_action_genome.py`

Layout example:

```
data/action_genome/
  object_bbox_and_relationship.pkl
  frame_list.txt
  object_classes.txt
  relationship_classes.txt
  videos/
  frames/            # extracted RGB frames (optional)
```

To prepare frames (if not pre-extracted):

```powershell
python -m orion.data.action_genome.extract_frames `
  --root data/action_genome `
  --output data/action_genome/frames
```

### 3.3 PVSG

Plan to download into:

```
data/pvsg/
  videos/<video_id>.mp4
  annotations/train.json
  annotations/val.json
  annotations/test.json
  metadata/object_classes.txt
  metadata/predicate_classes.txt
```

Each JSON entry should include:

- `video_id`
- `frames`: per-frame box list + class ids
- `relations`: `(frame_index, subj_idx, obj_idx, predicate_id)`

### 3.4 VSGR

Similar structure to PVSG:

```
data/vsgr/
  videos/
  annotations/train.json
  annotations/val.json
  annotations/test.json
  metadata/object_classes.txt
  metadata/predicate_classes.txt
```

Ensure annotations specify frame counts so you can derive prefix lengths for each fraction `F`.

**Important:** You can design the parser interfaces now; fill in actual file I/O once datasets are available.

---

## 4. Step-by-Step: Producing Tables 2–4

The following sections assume the datasets live under `data/` as described above. Adjust batch sizes/epochs to match your hardware; the commands here favor quick verification runs.

### 4.1 Table 3 – Scene Graph Generation (PVSG & VSGR)

**Train on PVSG**

```powershell
python scripts/vsgg_train_sgg.py `
   --dataset pvsg `
   --data-root data/pvsg `
   --split train `
   --epochs 10 `
   --batch-size 4 `
   --lr 1e-4 `
   --max-videos 200 `
   --max-frames-per-video 64 `
   --output checkpoints/pvsg_sgg_baseline.pt `
   --eval
```

**Evaluate on PVSG (val/test)**

```powershell
python scripts/vsgg_eval_sgg.py `
   --dataset pvsg `
   --data-root data/pvsg `
   --split test `
   --checkpoint checkpoints/pvsg_sgg_baseline.pt `
   --topk 20 50 100 `
   --max-videos 500 `
   --max-frames-per-video 64 `
   --output results/pvsg_sgg_baseline/sgg_metrics.json
```

**Repeat for VSGR** (swap `--dataset vsgr --data-root data/vsgr` everywhere). The two JSON files populate the PVSG and VSGR columns of Table 3 (metrics: `R@20/50/100`, `mR@20/50/100`).

### 4.2 Table 2 – Scene Graph Anticipation (Action Genome & VSGR, F ∈ {0.3,0.5,0.7})

**Train an SGA model (one checkpoint works for all F values):**

```powershell
python scripts/vsgg_train_sga.py `
   --dataset action_genome `
   --data-root data/action_genome `
   --split train `
   --fraction 0.9 `
   --epochs 10 `
   --batch-size 1 `
   --lr 1e-4 `
   --max-videos 200 `
   --min-frames-per-video 16 `
   --max-frames-per-video 128 `
   --output checkpoints/ag_sga_baseline.pt
```

**Evaluate for each fraction F (Action Genome):**

```powershell
foreach ($F in 0.3, 0.5, 0.7) {
   python scripts/vsgg_eval_sga.py `
      --dataset action_genome `
      --data-root data/action_genome `
      --split test `
      --checkpoint checkpoints/ag_sga_baseline.pt `
      --fraction $F `
      --topk 10 20 50 `
      --min-frames-per-video 16 `
      --max-frames-per-video 128 `
      --max-videos 500 `
      --output ("results/ag_sga_F$($F)/sga_metrics.json")
}
```

**Repeat training/eval for VSGR:**

```powershell
python scripts/vsgg_train_sga.py --dataset vsgr --data-root data/vsgr --split train --fraction 0.9 ...

foreach ($F in 0.3, 0.5, 0.7) {
   python scripts/vsgg_eval_sga.py --dataset vsgr --data-root data/vsgr --split test --checkpoint checkpoints/vsgr_sga_baseline.pt --fraction $F ...
}
```

Use the resulting `results/ag_sga_F*/` and `results/vsgr_sga_F*/` JSON files to fill the Action Genome and VSGR halves of Table 2 (`R/mR@{10,20,50}`).

### 4.3 Table 4 – Scene Graph Anticipation (Action Genome & VSGR, F = 0.9)

Reuse the same checkpoints, but evaluate with `--fraction 0.9`:

```powershell
python scripts/vsgg_eval_sga.py `
   --dataset action_genome `
   --data-root data/action_genome `
   --split test `
   --checkpoint checkpoints/ag_sga_baseline.pt `
   --fraction 0.9 `
   --topk 10 20 50 `
   --output results/ag_sga_F0.9/sga_metrics.json

python scripts/vsgg_eval_sga.py `
   --dataset vsgr `
   --data-root data/vsgr `
   --split test `
   --checkpoint checkpoints/vsgr_sga_baseline.pt `
   --fraction 0.9 `
   --topk 10 20 50 `
   --output results/vsgr_sga_F0.9/sga_metrics.json
```

Those metrics (Action Genome + VSGR, `R/mR@10/20/50`) form Table 4.

### 4.4 Converting metrics JSONs into Tables

All three tables can be generated by parsing the JSON outputs. A minimal pattern:

```powershell
python - <<'PY'
import json, pathlib
from pprint import pprint
root = pathlib.Path('results')
metrics = json.loads((root/'pvsg_sgg_baseline'/'sgg_metrics.json').read_text())
pprint(metrics)
PY
```

For production, write a helper script (e.g., `scripts/vsgg_render_tables.py`) that loads every metrics file, rounds to the desired precision, and prints markdown/LaTeX rows matching HyperGLM Tables 2–4.

---

## 5. Recommended Implementation Order

1. **Code First**
   - Implement `orion/vsgg/datasets.py`, `metrics.py`, `models.py`, and the train/eval scripts with synthetic stubs.
   - Use fake data to verify metric math and CLI wiring.

2. **Dataset Integration**
   - After code exists, download Action Genome, PVSG, VSGR into `data/` and complete the dataset parsers.
   - Validate one video per dataset manually to ensure box indices and labels align.

3. **Training/Evaluation Runs**
   - Run Action Genome SGG baseline (existing script) to get initial numbers.
   - Train SGG baseline on PVSG; evaluate and log metrics.
   - Train SGA baseline with `F=0.9` on Action Genome; evaluate across fractions for tables.

4. **HyperGLM Extensions (Optional)**
   - Add hypergraph construction, random walks, and LLM reasoning to mimic the full HyperGLM architecture once baselines are in place.

---

## 6. Logging and Results Storage

- Store experiment outputs under `results/<experiment_id>/`.
- Suggested files:
  - `sgg_metrics.json` with `R@20`, `R@50`, `R@100`, `mR@20`, `mR@50`, `mR@100`.
  - `sga_metrics_F0.3.json`, `sga_metrics_F0.5.json`, etc., each containing `R/mR@{10,20,50}`.
  - Training logs via TensorBoard or plain text.

Document the schema in `docs/results_schema.md` when ready.

---

## 7. Quick FAQ

**Q: Can we write all scripts before downloading datasets?**

Yes. Stub the dataset loaders and use synthetic data for unit tests. Real downloads are only needed once you’re ready to run actual experiments.

**Q: Which existing Orion code should we mimic for metrics?**

`orion.eval.action_genome.ActionGenomeRelationEvaluator`. The new `VideoRelationEvaluator` should match its behavior for R/mR.

**Q: How do we handle input fractions `F` for SGA?**

For each video with `T` frames, the dataset wrapper should provide:

- Observed prefix: frames `0 .. floor(F * T)`.
- Target future: remaining frames. Use only the target segment for computing metrics.

**Q: Where do we place dataset downloads?**

Under `data/`:

```
data/action_genome/
data/pvsg/
data/vsgr/
```

Controlled by CLI flags or config entries.

---

With these instructions, you can implement the full training/eval stack for HyperGLM-style SGG/SGA tables in Orion without delaying on dataset availability. Once the code skeletons are merged, clone the datasets into `data/`, finalize the parsers, and begin experimentation.
