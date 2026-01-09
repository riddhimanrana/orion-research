# Lambda-Orion Evaluation (Jan 8, 2026)

This document captures **what we actually ran on the remote `lambda-orion` machine** (via SSH), the resulting metrics, and the key architectural conclusions so far.

It’s meant to be a “ground truth log” of Orion v3 evaluation progress, with a special focus on:

- Neural Cypher RAG (Memgraph + Qwen reasoning)
- Re-ID batching (V-JEPA2)
- Detection quality / hallucinations (Gemini-3-Flash-Preview as judge)
- Scene graph relations (availability + usefulness)

> Artifacts (logs + JSON reports) referenced below are stored in this repo under:
>
> - `results/lambda_orion_20260108/`

---

## Remote environment (lambda-orion)

Collected via SSH:

- Hostname: `167-234-219-208`
- User: `ubuntu`
- Working dir: `/home/ubuntu/orion-research`
- Python:
  - `Python 3.12.3`
  - executable: `/usr/bin/python`
- PyTorch:
  - `torch==2.6.0+cu124`
  - CUDA available: `True`
  - GPU: `NVIDIA A10`
- Memgraph:
  - running as Docker container `orion-memgraph`
  - exposed ports include `7687` (Bolt)

LLMs present (Ollama):

- `qwen2.5:14b-instruct-q8_0` (used for RAG / Neural Cypher)
- `qwen3-vl:8b` (available, but **not used** in this evaluation)

Gemini SDK:

- `google-genai` import succeeded on lambda-orion
- `.env` exists on lambda-orion repo root and contains both `GOOGLE_API_KEY` and `GEMINI_API_KEY`

---

## Code state used on lambda-orion

We fast-forwarded lambda-orion to `origin/main`, then overlaid **local, uncommitted changes** by copying these files to lambda-orion:

- `scripts/eval_v3_architecture.py`
  - Uses `gemini-3-flash-preview`
  - Avoids repeated upload overhead by reusing a single in-memory video payload per run
  - Adds a *Gemini class audit* for top detected classes (detects likely false positives)
  - Adds `--run-showcase` option (not used in the final runs below; showcase was run separately)
- `orion/cli/run_showcase.py`
  - Adds `--reid-batch-size` and passes it into Phase 2 memory building
- `orion/perception/reid/matcher.py`
  - Always uses `embed_batch(...)` when supported
  - Adds `reid_batch_size` parameter to `build_memory_from_tracks(...)`

> NOTE: The evaluation JSON output format was later improved locally to include `gemini_model`, `top_detected_classes`, `class_audit`, but we did **not** rerun lambda eval after that small change.

---

## Evaluation harness

### End-to-end pipeline (Memgraph ingest)

We used:

- `python -m orion.cli.run_showcase`
  - Phase 1: YOLO detection via `FrameObserver` + tracking via `EnhancedTracker`
  - Phase 2: V-JEPA2 Re-ID embeddings → memory objects
  - Phase 3: scene graph building
  - Export: Memgraph ingest (`--memgraph --memgraph-clear`)

### QA evaluation (Gemini-ground-truth)

We used:

- `python scripts/eval_v3_architecture.py`
  - Questions: 12 fixed prompts (discovery / spatial / temporal / interaction / reasoning)
  - Evidence: Memgraph queries (template and Neural Cypher)
  - Reasoning: Qwen (`qwen2.5:14b-instruct-q8_0` via Ollama)
  - Judge: Gemini (`gemini-3-flash-preview`) with **full video input**

Gemini verdict classes:

- `CORRECT`, `PARTIAL`, `INCORRECT`, `HALLUCINATION`

---

## Runs executed on lambda-orion

### Run A: `test.mp4` end-to-end showcase (CUDA, batch=8)

Command conceptually equivalent to:

- `python -m orion.cli.run_showcase --episode lambda_eval_test_20260108 --video data/examples/test.mp4 --fps 4 --yolo-model yolo11m --device cuda --reid-batch-size 8 --force-phase1 --force-memory --force-graph --memgraph --memgraph-clear --no-overlay`

Key outputs (from logs):

- Duration: **~23.8s** wall time
- Detection (sampled frames):
  - video: `60.96s`, ~`261` sampled frames
  - total detections: `156`
- Tracking:
  - unique tracks: `30`
  - track observations: `314`
- Memory:
  - crops embedded: `57` crops from `12` tracks
  - memory objects: `9`
- Scene graph:
  - frames: `88`
  - **edges/frame: 0.00**
- Memgraph ingest:
  - observations: `314`
  - relations: `0`

Artifacts:

- `results/lambda_orion_20260108/lambda_eval_showcase_test_batch8_v2.log`

### Run B: `test.mp4` Phase 2 batching micro-benchmark

We rebuilt Phase 2 only (reuse tracks, skip graph) with two different batch sizes.

- batch=1:
  - wall time: **~15.22s**
  - artifact: `results/lambda_orion_20260108/lambda_eval_phase2_only_test_batch1.log`
- batch=8:
  - wall time: **~14.71s**
  - artifact: `results/lambda_orion_20260108/lambda_eval_phase2_only_test_batch8.log`

Conclusion:

- For `test.mp4`, batching speedup is **negligible** because the workload is tiny (57 crops) and the dominant cost is model load.

### Run C: `test.mp4` Gemini-validated QA

Key metrics (from JSON report):

- Avg latency: **4507ms**
- Neural Cypher “used” rate: **33.3%**
- Gemini outcomes: **1 correct**, **0 partial**, **11 incorrect/hallucination**

Notable Gemini class-audit finding (from log):

- top detected classes included `toilet`
- Gemini marked `toilet` **absent** → strong false-positive signal

Artifacts:

- JSON report: `results/lambda_orion_20260108/lambda_eval_v3_test_20260108.json`
- Log: `results/lambda_orion_20260108/lambda_eval_v3_test_20260108.log`

### Run D: `video.mp4` end-to-end showcase (CUDA, batch=8)

Key outputs (from logs):

- Duration: **~31.1s** wall time
- Detection:
  - video: `66.00s`, ~`283` sampled frames
  - total detections: `307`
- Tracking:
  - unique tracks: `20`
  - track observations: `879`
- Memory:
  - crops embedded: `147` crops from `33` tracks
  - memory objects: `10`
- Scene graph:
  - frames: `169`
  - edges/frame: **0.92**
- Memgraph ingest:
  - observations: `879`
  - relations: **155**

Artifacts:

- `results/lambda_orion_20260108/lambda_eval_showcase_video_batch8.log`

### Run E: `video.mp4` Gemini-validated QA

Key metrics (from JSON report):

- Avg latency: **4956ms**
- Neural Cypher “used” rate: **41.7%**
- Gemini outcomes: **0 correct**, **1 partial**, **11 incorrect/hallucination**

Gemini class-audit finding (from console output):

- Gemini marked **absent**: `remote`, `sink`, `suitcase`

Artifacts:

- JSON report: `results/lambda_orion_20260108/lambda_eval_v3_video_20260108.json`
- Log: `results/lambda_orion_20260108/lambda_eval_v3_video_20260108.log`

---

## What these results imply (architecture-level)

### 1) Detection quality is the #1 blocker for QA accuracy

Across both videos, the *top detected classes* include multiple items Gemini says are not present (e.g. `toilet`, `remote`, `sink`, `suitcase`). These false positives cascade into:

- wrong “objects detected” answers
- wrong object counts
- wrong temporal assertions (“X visible at 30s”)
- wrong inferred room/activity

**Key takeaway:** Without a strong semantic verification/filtering layer upstream, Neural Cypher RAG faithfully retrieves incorrect facts.

### 2) Semantic filtering / FastVLM is not yet exercised by the Memgraph showcase path

`orion.cli.run_showcase` Phase 1 currently uses `FrameObserver` + YOLO and does **not** run the `PerceptionEngine` semantic verification loop (FastVLM), even though that pipeline exists (`orion.cli.run_engine`).

**Implication:** We are evaluating Neural Cypher RAG against *raw detector output*, not the intended v3 architecture where FastVLM reduces false positives.

### 3) Neural Cypher success rate is mediocre and sometimes fails hard

We saw errors like:

- `Neural Cypher execution failed: All subqueries in an UNION must have the same column names.`

This both reduces coverage and injects irrelevant fallbacks (e.g. “No information found about 'tie'.”).

**Actionable fix direction:** add a Cypher safety layer in RAG:

- detect `UNION` and enforce consistent projections (same column names)
- auto-rewrite to `RETURN ... AS ...` normalization, or reject and retry with constraints

### 4) Scene graph relation density is highly video-dependent

- `test.mp4`: 0 relations
- `video.mp4`: 155 relations (~0.92 edges/frame)

This means the spatial reasoning pathway is fragile and won’t generalize without more robust relation extraction.

### 5) Interaction/held-object questions are currently mostly unsupported

The evaluation asked questions like “What did the person interact with?”

- We did not enable `--detect-hands` in showcase.
- The tracker/detector pipeline doesn’t reliably infer “held” events.

So interaction questions tend to be judged incorrect.

---

## Prioritized next steps (recommended)

1) **Unify Phase 1 ingestion** so Memgraph exports come from the FastVLM-enabled pipeline:
   - either modify `orion.cli.run_showcase` to optionally run `PerceptionEngine` instead of `run_tracks`/`FrameObserver`
   - or add a post-filter stage in `FrameObserver` using `SemanticFilterV2` / FastVLM

2) **Fix Neural Cypher robustness**
   - normalize `UNION` projections
   - add retry loop: “invalid Cypher → regenerate” with explicit schema hints

3) **Stop emitting technical errors as user answers**
   - e.g. “No embedding found for 'person'…” should become a graceful fallback response + recoverable path

4) **Improve interaction coverage**
   - enable hand detector in showcase runs (`--detect-hands`) and ingest those events
   - strengthen `held_by` relation derivation

5) **Evaluation enhancements**
   - persist class-audit results and top-class counts in the JSON outputs (so we can trend false positives)
   - expand evaluation to a “per-video regression suite” with fixed seeds and cached Gemini responses

---

## Appendix: local artifact index

All artifacts were copied from lambda-orion to this repo:

- `results/lambda_orion_20260108/lambda_eval_showcase_test_batch8_v2.log`
- `results/lambda_orion_20260108/lambda_eval_showcase_video_batch8.log`
- `results/lambda_orion_20260108/lambda_eval_phase2_only_test_batch1.log`
- `results/lambda_orion_20260108/lambda_eval_phase2_only_test_batch8.log`
- `results/lambda_orion_20260108/lambda_eval_v3_test_20260108.json`
- `results/lambda_orion_20260108/lambda_eval_v3_test_20260108.log`
- `results/lambda_orion_20260108/lambda_eval_v3_video_20260108.json`
- `results/lambda_orion_20260108/lambda_eval_v3_video_20260108.log`
