# Phase 2 — Re-ID + Memory (Complete)

Date: 2025-11-16

## What we added

- DINO(v2/v3) embedder backend with ModelManager integration
- Re-ID pipeline to compute per-track embeddings from video crops
- Greedy cosine clustering per category to build persistent memory objects
- Outputs:
  - `results/<id>/memory.json` (objects, embeddings, statistics)
  - `results/<id>/reid_clusters.json` (memory_id → track_ids)
  - `results/<id>/tracks.jsonl` updated with `embedding_id`
- New CLI: `python -m orion.cli.run_reid ...`

## Key files

- `orion/perception/reid/reid.py` — core Phase 2 logic
- `orion/cli/run_reid.py` — CLI entrypoint

## Usage

```bash
# Build memory for existing tracks + video
python -m orion.cli.run_reid \
  --tracks results/<episode>/tracks.jsonl \
  --video data/examples/test.mp4 \
  --out results/<episode> \
  --threshold 0.78 \
  --max-crops-per-track 4
```

## Validation (test_validation)

- Memory objects: 19
- Updated `tracks.jsonl` with `embedding_id`
- Top examples:
  - `mem_016`: refrigerator, 18 observations (frames 1020–1105)
  - `mem_001`: tv, 12 observations (frames 10–1825 across multiple short tracks)

## Notes and next steps

- Threshold can be tuned per class; current default is 0.75 (validated at 0.78 here)
- Future work (Phase 3): Create `memory.json` lifecycle updates across longer gaps and establish entity persistence across restarts, plus event logging into `events.jsonl`.
