"""Utility script: run perception (fast config) and print concise adaptive Re-ID metrics.

Usage:
    python scripts/print_reid_metrics.py [video_path]

Defaults to sample video if none provided.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.config import get_fast_config
from orion.perception.engine import PerceptionEngine

DEFAULT_VIDEO = "data/examples/room.mp4"


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    cfg = get_fast_config()
    # Configure for dinov3 adaptive Re-ID evaluation
    cfg.embedding.backend = "dinov3"
    cfg.embedding.use_cluster_embeddings = True
    cfg.embedding.cluster_similarity_threshold = 0.55
    cfg.embedding.embedding_dim = 512
    cfg.embedding.reid_debug = False
    # Minimize description overhead
    cfg.description.max_tokens = 1
    engine = PerceptionEngine(config=cfg, verbose=False)
    result = engine.process_video(video, save_visualizations=False)

    reid = (result.metrics or {}).get("reid") or {}
    print("\n=== Adaptive Re-ID Metrics ===")
    print(f"Video: {video}")
    print(f"Backend: {reid.get('backend')}  Base threshold: {reid.get('base_threshold')}  Reduction: {reid.get('reduction')}")
    print(f"Total merges: {reid.get('merges_total')}")
    ct = reid.get('class_thresholds', {})
    if ct:
        print("Class thresholds:")
        for cls, th in ct.items():
            print(f"  {cls}: {th:.3f}")
    stats = reid.get('similarity_stats', {})
    if stats:
        print("Similarity stats (median/mean/std/p95):")
        for cls, s in stats.items():
            print(f"  {cls}: {s.get('median'):.3f}/{s.get('mean'):.3f}/{s.get('std'):.3f}/{s.get('p95'):.3f}")
    pcm = reid.get('per_class_merges', {})
    if pcm:
        print("Per-class merges:")
        for cls, c in pcm.items():
            print(f"  {cls}: {c}")

    print("\nEntities:")
    for e in result.entities:
        cls = e.object_class.value if hasattr(e.object_class, 'value') else str(e.object_class)
        print(f"  {e.entity_id}: class={cls} obs={len(e.observations)} frames={e.first_seen_frame}-{e.last_seen_frame}")


if __name__ == "__main__":
    main()
