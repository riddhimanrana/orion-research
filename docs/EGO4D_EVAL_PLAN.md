# Ego4D → Orion evaluation plan (Lambda)

This document outlines a **repeatable, non-overfit** way to stress-test Orion on a broad, real-world egocentric dataset (Ego4D) and then optionally run Gemini-based spot checks.

For benchmark/task background, baseline repos, and a concrete mapping from Ego4D benchmarks → Orion modules, see:

- `docs/EGO4D_BENCHMARK_RESEARCH.md`

## 0) Important constraints (read first)

### Dataset access & credentials

Ego4D data is hosted on S3 and **requires a license** and **AWS credentials** (provided after approval).

- Docs: [Start Here](https://ego4d-data.org/docs/start-here/)
- CLI README: [Ego4D Dataset Download CLI](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md)

The Ego4D CLI uses AWS credentials from `~/.aws/credentials` (or a named profile).

### External API / privacy note (Gemini)

Ego4D videos may contain identifiable people and the license may restrict redistribution / third-party processing.

Before sending any Ego4D frames to **Gemini API** (or any external service), **confirm the Ego4D license terms allow it**.

This repo’s Gemini evaluation scripts should be treated as **optional** and require an explicit confirmation flag in the benchmark harness.

### Storage & bandwidth

Full-scale Ego4D is multi-terabyte. On Lambda NFS, available space may be limited.

Recommended download targets for benchmarking:

- `clips` (benchmark clips)
- `video_540ss` (downscaled canonical videos)

## 1) Target layout on mounted filesystem

Create a dedicated dataset root on the mounted filesystem:

- `/lambda/nfs/orion-core-fs/data/ego4d/`  (raw Ego4D downloads)
- `/lambda/nfs/orion-core-fs/data/orion_eval/ego4d_subsets/` (UID lists + manifests)
- `/lambda/nfs/orion-core-fs/orion-research/results/ego4d_*` (Orion outputs)

## 2) Proposed experiment design

### Goal

Evaluate Orion’s end-to-end pipeline on a **diverse** subset rather than “tuning to one video.”

### Subset selection

Download a subset of **N=100** videos/clips via Ego4D CLI filters:

- Use `--video_uid_file` (UID list generated with a fixed seed)
- Prefer `--datasets clips` or `--datasets video_540ss` for size

We’ll store the UID list as:

- `.../ego4d_subsets/ego4d_100_seed42/uids.txt`

### Run Orion

Run Orion on all N videos with a consistent configuration:

- `fps=4` (or 2 for faster)
- `--no-overlay` (speed)
- `reid-threshold` fixed

### Gemini spot-check

Randomly pick **K=10** from the N videos and run:

- `scripts/gemini_accuracy_evaluation.py --model gemini-3-flash-preview`

Aggregate all Gemini reports into one Markdown summary.

## 3) Reproducibility

- Always record:
  - seed, subset UID list, CLI flags, Orion config, commit SHA
- Avoid modifying vocab/class lists per dataset.
- If we introduce label canonicalization, keep it **global** and validate on multiple sources.

## 4) Next actions checklist

1. Install Ego4D CLI (`pip install ego4d`) in Lambda venv
2. Configure `~/.aws/credentials`
3. Create dataset folders on `/lambda/nfs/orion-core-fs/`
4. Download metadata + chosen dataset (`clips` or `video_540ss`) for a sampled UID list
5. Run Orion batch pipeline
6. Run Gemini on K=10 (only if license allows)
7. Write aggregated findings
