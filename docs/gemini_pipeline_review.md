# Gemini pipeline review

This repo includes a Gemini-powered audit script that critiques Orion outputs end-to-end (detection → classification → tracking → re-ID → semantics).

## Why this exists

Classic metrics (MOTA/IDF1) require ground truth. For rapid iteration, we use Gemini Vision as a *structured reviewer* to:

- spot obvious misses/false positives
- highlight label confusions
- detect likely ID-switches or track contamination
- suggest concrete fixes (prompt size, thresholds, tracker gating, re-id thresholds)

This is **not ground truth**, but it is very useful for quickly finding failure modes.

## Script

`scripts/gemini_pipeline_review.py`

### Inputs

- A results directory containing at least:
  - `tracks.jsonl`
- A video path for frame extraction (`--video ...`) or `episode_meta.json` inside the results directory with `video_path`.

Optional inputs (if present, they are included as context in the review):

- `entities.json`
- `pipeline_summary.json`
- `gemini_comparison.json`

### Output

- `<results>/gemini_pipeline_review.json`

This includes:

- `frame_audits`: per-frame critique + per-stage recommended fixes
- `track_identity_audits`: “same object?” judgments across distant crops for the same `track_id`
- `overall_review`: prioritized issues and recommended experiments

## Setup

Add your Gemini key to `.env` at repo root:

- `GOOGLE_API_KEY=...` (preferred)
- or `GEMINI_API_KEY=...`

Optional:

- `GEMINI_MODEL=gemini-3-flash-preview`

## Run

Example:

- Results: `results/validation` (must contain `tracks.jsonl`)
- Video: `data/examples/test.mp4`

Run the script with a small sample first:

- `--num-frames 6`
- `--track-pairs 6`

Then increase once it looks good.

## Interpreting the findings

Treat the output as a **debugging guide**:

- If Gemini repeatedly reports misses in the same categories → detection prompt/class set or thresholding is off.
- If Gemini reports the right object but calls it a different name → classification label set / canonicalization needs work.
- If track crops look like different objects → ID-switch, contaminated track, or wrong merge.

For rigorous evaluation, pair this with controlled benchmarks or manually annotated clips.

