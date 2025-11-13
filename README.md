Orion Minimal Perception Pipeline
=================================

This pared-down version of Orion focuses on a lean egocentric perception stack.

Components:

- Detection: YOLO11 (Ultralytics) for 2D object proposals.
- Embeddings: Configurable backend (CLIP / DINO / DINOv3 stub) for appearance & Re-ID.
- Re-ID: Adaptive merging with per-class similarity thresholds and spatial/temporal heuristics.
- Descriptions: FastVLM (mlx-vlm local) with optional truncation for speed.
- Depth: DepthAnythingV2 (integrated internally) for spatial mapping utilities.

Key Scripts:

- `scripts/test_full_pipeline.py` – End-to-end sanity run.
- `scripts/test_dinov3_cluster.py` – DINOv3 + cluster embedding efficiency test.
- `scripts/print_reid_metrics.py` – Quick adaptive Re-ID metrics report.
- `scripts/test_clip_classification.py` – CLIP classification/embedding smoke test.
- `scripts/test_spatial_mapping_room.py` – Depth / spatial mapping validation.

Re-ID Metrics:

Adaptive metrics (class thresholds, similarity stats, merges) are attached to `PerceptionResult.metrics['reid']`.
Use `python scripts/print_reid_metrics.py <video>` to inspect.

Minimal Install (development):

```bash
pip install -e .[dev]
```

Run Examples:

```bash
python scripts/test_full_pipeline.py
python scripts/test_dinov3_cluster.py
python scripts/print_reid_metrics.py data/examples/room.mp4
```

Configuration:

Edit `orion/perception/config.py` or adjust fields programmatically (e.g., backend, clustering, `reid_debug`).

License: See `LICENSE`.
