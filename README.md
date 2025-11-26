<!-- markdownlint-disable-file -->

Orion Research: Memory-Centric Video Understanding
=================================================

Orion is a memory-centric video understanding system for persistent object tracking, state changes, and long-term scene reasoning. Built for Apple Silicon (M-series) with iPhone video capture workflows.

## Architecture

**Phase 1 (Current): Perception Pipeline**
- Detection: YOLO11 (Ultralytics) or GroundingDINO (zero-shot) for 2D proposals
- Embeddings: Configurable backend (CLIP / DINO / DINOv3) for appearance & Re-ID
- Re-ID: Adaptive merging with per-class similarity thresholds and spatial/temporal heuristics
- Descriptions: FastVLM (mlx-vlm local) for visual descriptions
- Depth: DepthAnythingV2 for spatial mapping (optional)

**Phase 2–4 (Roadmap): Memory Engine**
- Temporal Scene Memory: Long-range object persistence across occlusions
- Re-ID Index: DINOv3 embeddings + cosine similarity matching
- Temporal Scene Graph: Object states, spatial relations, event tracking
- Video QA: MLX-VLM powered reasoning over structured memory

## Episodes and Results

Orion uses a standardized episode/results format for reproducibility:

**Episode Structure:**
```
data/examples/episodes/<episode_id>/
├── meta.json          # Episode metadata (fps, resolution, device)
├── gt.json           # Optional ground truth annotations
└── video.mp4         # Source video
```

**Results Structure:**
```
results/<episode_id>/
├── tracks.jsonl       # Per-frame detections and tracks
├── memory.json       # Persistent object memory (Phase 3+)
├── events.jsonl      # Lifecycle events (Phase 3+)
├── scene_graph.jsonl # Temporal scene graph (Phase 4+)
└── entities.json     # Legacy clustered entities (Phase 1)
```

See [docs/episodes.md](docs/episodes.md) and [docs/results_schema.md](docs/results_schema.md) for full schemas.

**Quick Example:**
```bash
# List available episodes
python -c "from orion.config import list_episodes; print(list_episodes())"

# Run perception pipeline on demo episode
python scripts/test_full_pipeline.py --episode demo_room

# Results saved to: results/demo_room/
```

Key Scripts:

- `scripts/test_full_pipeline.py` – End-to-end sanity run.
- `scripts/test_dinov3_cluster.py` – DINOv3 + cluster embedding efficiency test.
- `scripts/print_reid_metrics.py` – Quick adaptive Re-ID metrics report.
- `scripts/test_clip_classification.py` – CLIP classification/embedding smoke test.
- `scripts/test_spatial_mapping_room.py` – Depth / spatial mapping validation.
- `python -m orion.cli.run_quality_sweep` – Automated Phase1→Phase2→graph→QA sweeps.
- `python -m orion.cli.run_showcase` – Full pipeline demo with overlay + Memgraph ingest.

### Showcase Overlay + Memgraph

End-to-end runnable demo that produces tracking/memory artifacts, a narrative-rich overlay, and (optionally) Memgraph ingest:

```bash
# Baseline demo on data/examples/test.mp4, overlay included
python -m orion.cli.run_showcase --episode test_demo --video data/examples/test.mp4

# Re-run but reuse tracks/memory, only rebuild scene graph + overlay
python -m orion.cli.run_showcase --episode test_demo --skip-phase1 --skip-memory

# Push everything into a running Memgraph (docker run -p 7687:7687 memgraph/memgraph)
python -m orion.cli.run_showcase \
	--episode test_demo --video data/examples/test.mp4 \
	--memgraph --memgraph-host 127.0.0.1 --memgraph-port 7687

# Standalone overlay regeneration once results exist
python scripts/render_video_overlay.py \
	--video data/examples/test.mp4 \
	--results results/test_demo \
	--output results/test_demo/overlay.mp4
```

Outputs land in `results/<episode_id>/`:

- `tracks.jsonl`, `memory.json`, `scene_graph.jsonl`, `graph_summary.json` – refreshed artifacts
- `video_overlay_insights.mp4` – overlay that callouts re-tracking, memory assignments, zones, and relation snippets
- Memgraph (optional) – `Entity`, `Frame`, `Zone`, and `NEAR/ON/HELD_BY` edges accessible via `mgconsole`

### Perception Quality Sweep

Use the consolidated CLI to regenerate perception artifacts, export graph samples, and optionally score relations with Gemini:

```bash
# Minimal single-episode run
python -m orion.cli.run_quality_sweep --episode demo_room

# Custom video override with Gemini validation and more samples
python -m orion.cli.run_quality_sweep \
	--episode kitchen_walkthrough \
	--video /path/to/kitchen_walkthrough.mov \
	--max-graph-samples 25 \
	--run-gemini --gemini-api-key $GOOGLE_API_KEY \
	--gemini-model gemini-2.0-pro-exp-02-05

# Batch plan (plan.json contains [{"episode": "demo_room", "video": "..."}, ...])
python -m orion.cli.run_quality_sweep --plan plan.json
```

Key outputs are written back into `results/<episode_id>/`, including an aggregated `quality_report.json`, refreshed `scene_graph.jsonl`, exported samples under `graph_samples/`, and optional `gemini_feedback.json` verdicts.

Re-ID Metrics:

Adaptive metrics (class thresholds, similarity stats, merges) are attached to `PerceptionResult.metrics['reid']`.
Use `python scripts/print_reid_metrics.py <video>` to inspect.

Minimal Install (development):

```bash
pip install -e .[dev]
```

Environment Activation:

```bash
conda activate orion
```

Run Examples:

```bash
python scripts/test_full_pipeline.py
python scripts/test_dinov3_cluster.py
python scripts/print_reid_metrics.py data/examples/room.mp4
```

Configuration:

Edit `orion/perception/config.py` or adjust fields programmatically (e.g., backend, clustering, `reid_debug`).

### Switching to GroundingDINO Detection

GroundingDINO can replace YOLO11 when you need stronger zero-shot coverage or text-conditioned prompts. Toggle it directly through the perception config:

```python
from orion.perception.config import PerceptionConfig
from orion.perception.engine import PerceptionEngine

config = PerceptionConfig()
config.detection.backend = "groundingdino"
config.detection.groundingdino_model_id = "IDEA-Research/grounding-dino-base"  # or tiny/large variants
config.detection.groundingdino_prompt = "person . chair . laptop . monitor . bottle ."

engine = PerceptionEngine(config=config)
result = engine.process_video("data/examples/test.mp4")
```

Key notes:
- Prompts are dot-separated phrases; keep them concise for best grounding results.
- `groundingdino_box_threshold` / `groundingdino_text_threshold` control aggressiveness (lower = more boxes).
- Pillow is now a core dependency (`pip install -e .` pulls it in) to support Hugging Face processors.


License: See `LICENSE`.
