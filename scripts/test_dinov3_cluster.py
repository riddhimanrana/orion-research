"""Quick test for DINOv3 + cluster embeddings memory efficiency.

Runs perception with modified config (dinov3 backend + clustering) on sample video
and prints embedding / cluster stats.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.config import get_fast_config, EmbeddingConfig
from orion.perception.engine import PerceptionEngine

VIDEO = "data/examples/room.mp4"

cfg = get_fast_config()
# Override embedding backend to dinov3 + clustering
cfg.embedding.backend = "dinov3"
cfg.embedding.use_cluster_embeddings = True
cfg.embedding.cluster_similarity_threshold = 0.55  # slightly looser merge
cfg.embedding.embedding_dim = 512  # keep dimension consistent with placeholder
cfg.embedding.reid_debug = False  # suppress verbose pairwise output

# Disable descriptions for this speed/metrics test
cfg.description.describe_once = True
cfg.description.max_tokens = 1  # minimal to avoid VLM token mismatch

engine = PerceptionEngine(config=cfg, verbose=False)
result = engine.process_video(VIDEO, save_visualizations=False)

metrics = result.metrics or {}
print("\n=== DINOv3 Cluster Embedding Test ===")
print(f"Backend: {metrics.get('embedding_backend')}  Dim: {metrics.get('embedding_dim')}")
print(f"Detections processed: {metrics.get('sampled_frames')} frames, detections/frame: {metrics.get('detections_per_sampled_frame'):.2f}")
print(f"Cluster representatives: {metrics.get('cluster_embeddings')}  Avg cluster size: {metrics.get('avg_cluster_size'):.2f}")
print(f"Total entities: {result.unique_entities}")
print("Timings (s):")
for k,v in metrics.get('timings', {}).items():
    print(f"  {k}: {v:.3f}")

reid = metrics.get('reid') or {}
if reid:
    print("\nRe-ID Adaptive Metrics:")
    print(f"  Backend: {reid.get('backend')}  Base threshold: {reid.get('base_threshold')}")
    print(f"  Total merges: {reid.get('merges_total')}  Reduction: {reid.get('reduction')}")
    ct = reid.get('class_thresholds', {})
    if ct:
        print("  Class thresholds:")
        for cls, th in ct.items():
            print(f"    {cls}: {th:.3f}")
    stats = reid.get('similarity_stats', {})
    if stats:
        print("  Similarity distribution stats (per class):")
        for cls, s in stats.items():
            print(f"    {cls}: median={s.get('median'):.3f} mean={s.get('mean'):.3f} std={s.get('std'):.3f} p95={s.get('p95'):.3f}")
    pcm = reid.get('per_class_merges', {})
    if pcm:
        print("  Per-class merges:")
        for cls, mc in pcm.items():
            print(f"    {cls}: {mc}")
