# Semantic Filtering (Phase 3)

Orion v2 introduces a semantic filtering stage to validate object detections using a Vision-Language Model (FastVLM) and Sentence Embeddings.

## Overview

The semantic filter verifies that the visual appearance of a tracked object matches its detected label. This helps remove false positives from the open-vocabulary detector (YOLO-World).

**Pipeline:**
1.  **Crop Extraction**: For each unique track, extract the "best" crop (highest confidence detection).
2.  **VLM Description**: Generate a visual description of the crop using FastVLM (e.g., "A wooden chair with four legs").
3.  **Semantic Embedding**: Embed both the description and the detected label (e.g., "chair") into a shared vector space using `sentence-transformers/all-mpnet-base-v2`.
4.  **Similarity Check**: Compute cosine similarity. If similarity < threshold (default 0.25), the track is rejected.

## Usage

### 1. Pre-requisites
Ensure the pre-baked YOLO-World model is available (for Phase 1 speedup):
```bash
python scripts/prebake_yoloworld_vocab.py
```

### 2. Running the Filter
The filter is typically run after tracking (Phase 1) and Re-ID (Phase 2).

```bash
python -m orion.cli.run_vlm_filter \
  --video data/examples/test.mp4 \
  --tracks results/test_demo/tracks.jsonl \
  --out-tracks results/test_demo/tracks_filtered.jsonl \
  --threshold 0.25 \
  --device cuda
```

### 3. Configuration
The filter can be configured via `SemanticFilterConfig` in `orion/perception/filters.py` or CLI arguments.

- `--threshold`: Similarity threshold (0.0-1.0). Higher = stricter. Default 0.25.
- `--device`: Inference device (`cuda`, `mps`, `cpu`).

## Performance
- **Batching**: Descriptions are generated sequentially (VLM limitation), but embeddings are computed in batches.
- **Caching**: Label embeddings are cached to avoid re-computation.
- **Speed**: ~0.5s per unique track on NVIDIA A10G.

## Integration
The filter is integrated into the `run_vlm_filter` CLI tool.
Future integration into `run_showcase` is planned.
