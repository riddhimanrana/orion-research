# Dataset Download Instructions

This reference explains how to obtain and stage the datasets needed for Orion's Video Scene Graph workflows (Action Genome, PVSG, VSGR). Follow these steps before running the training/evaluation scripts documented in `docs/evaluation_instructions.md`.

---

## 1. Prerequisites

- Ensure you have sufficient storage (each dataset can require tens to hundreds of GB).
- Install the Orion environment so you can run sanity checks once downloads finish:
  ```powershell
  conda activate orion
  pip install -e .
  ```
- Create the target directories under the repo root:
  ```powershell
  New-Item -ItemType Directory -Force data/action_genome | Out-Null
  New-Item -ItemType Directory -Force data/pvsg | Out-Null
  New-Item -ItemType Directory -Force data/vsgr | Out-Null
  ```

---

## 2. Action Genome

- **Project page:** https://github.com/JingweiJ/ActionGenome
- **What to download:**
  - `object_bbox_and_relationship.pkl`
  - `frame_list.txt`
  - `object_classes.txt`
  - `relationship_classes.txt`
  - Videos and/or extracted frames (links provided in their README)
- **Layout in Orion:**
  ```text
  data/action_genome/
    object_bbox_and_relationship.pkl
    frame_list.txt
    object_classes.txt
    relationship_classes.txt
    videos/                # copy AG videos here (if available)
    frames/                # optional; create if you extract frames locally
  ```
- **Extract frames if needed:**
  ```powershell
  python -m orion.data.action_genome.extract_frames `
    --root data/action_genome `
    --output data/action_genome/frames
  ```
- **Smoke test:**
  ```powershell
  python - <<'PY'
  from orion.vsgg.datasets import ActionGenomeSceneGraphDataset
  ds = ActionGenomeSceneGraphDataset('data/action_genome', split='train', max_samples=8)
  print(f"Loaded {len(ds.samples)} samples; {ds.num_objects} objects, {ds.num_predicates} predicates")
  PY
  ```

---

## 3. PVSG (Panoptic Video Scene Graph)

- **Project page:** https://github.com/LilyDaytoy/OpenPVSG
- **What to download:**
  - PVSG videos (MP4 or the format provided)
  - Annotation JSONs for `train`, `val`, `test`
  - Object and predicate class definitions (text or JSON)
- **Typical download flow:**
  1. Clone the repo:
     ```powershell
     git clone https://github.com/LilyDaytoy/OpenPVSG.git
     ```
  2. Follow their README to pull the videos and annotations (often via scripts or direct links).
- **Layout in Orion:**
  ```text
  data/pvsg/
    videos/
      <video_id>.mp4
    annotations/
      train.json
      val.json
      test.json
    metadata/
      object_classes.txt
      predicate_classes.txt
  ```
  If the OpenPVSG metadata is stored as JSON lists, convert to simple `.txt` files (one class name per line) so `PVSGSceneGraphDataset` can read them.
- **Smoke test:**
  ```powershell
  python - <<'PY'
  from orion.vsgg.datasets import PVSGSceneGraphDataset
  ds = PVSGSceneGraphDataset('data/pvsg', split='val', max_videos=2)
  sample = ds[0]
  print(sample.video_id, len(sample.frames), ds.num_objects, ds.num_predicates)
  PY
  ```

---

## 4. VSGR (Video Scene Graph Reasoning)

- **Project page:** https://uark-cviu.github.io/projects/HyperGLM/#annotations
- **What to download:**
  - VSGR videos (links provided on the project page)
  - Annotation files for `train`, `val`, `test`
  - Object and predicate class definitions
- **Layout in Orion:**
  ```text
  data/vsgr/
    videos/
      <video_id>.mp4
    annotations/
      train.json
      val.json
      test.json
    metadata/
      object_classes.txt
      predicate_classes.txt
  ```
  As with PVSG, convert JSON class definitions into `.txt` lists if necessary.
- **Smoke test:**
  ```powershell
  python - <<'PY'
  from orion.vsgg.datasets import VSGRSceneGraphDataset
  ds = VSGRSceneGraphDataset('data/vsgr', split='val', max_videos=2)
  sample = ds[0]
  print(sample.video_id, len(sample.frames), ds.num_objects, ds.num_predicates)
  PY
  ```

---

## 5. After Downloads

1. Run the loader tests above for each dataset to ensure paths and annotations are valid.
2. Follow `docs/evaluation_instructions.md` to launch the SGG/SGA training and evaluation scripts.
3. Keep raw download archives outside the repo if possible to save disk space inside the working tree.
