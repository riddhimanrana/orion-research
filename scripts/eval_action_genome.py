from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[1]
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))

from orion.data.action_genome import ActionGenomeFrameDataset
from orion.eval import (
    ActionGenomeLabelMapper,
    ActionGenomeRelationEvaluator,
    build_ground_truth_triplets,
    build_prediction_triplets,
)
from orion.perception.config import PerceptionConfig
from orion.perception.engine import PerceptionEngine

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("eval_action_genome")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Orion on Action Genome frames")
    parser.add_argument("--dataset_root", required=True, help="Path to ActionGenome/dataset/ag")
    parser.add_argument("--split", default="test", help="Dataset split (train/val/test/all)")
    parser.add_argument("--limit", type=int, default=None, help="Optional max samples to evaluate")
    parser.add_argument("--output", default="results/action_genome_eval.json", help="Where to write metrics JSON")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for SGDET matching")
    parser.add_argument("--auto-extract", dest="auto_extract", action="store_true", help="Automatically extract missing frames with ffmpeg")
    parser.add_argument("--no-auto-extract", dest="auto_extract", action="store_false", help="Disable on-demand frame extraction")
    parser.add_argument("--ffmpeg-binary", default="ffmpeg", help="ffmpeg executable to use when extracting frames")
    parser.set_defaults(auto_extract=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = ActionGenomeFrameDataset(
        args.dataset_root,
        split=args.split,
        max_samples=args.limit,
        auto_extract_frames=args.auto_extract,
        ffmpeg_binary=args.ffmpeg_binary,
    )
    logger.info("Loaded %d Action Genome frames (split=%s)", len(dataset), args.split)

    config = PerceptionConfig()
    engine = PerceptionEngine(config=config)
    mapper = ActionGenomeLabelMapper()
    evaluator = ActionGenomeRelationEvaluator(iou_thresh=args.iou_thresh)

    per_sample: list[dict] = []
    processed = 0

    for sample in dataset:
        start = time.time()
        try:
            result = engine.process_image(
                str(sample.image_path),
                frame_number=0,
                timestamp=0.0,
                source_name=sample.tag,
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.exception("Failed to process frame %s", sample.tag)
            per_sample.append({
                "tag": sample.tag,
                "image_path": str(sample.image_path),
                "error": str(exc),
            })
            continue

        predictions = build_prediction_triplets(result, mapper)
        ground_truth = build_ground_truth_triplets(sample, dataset)
        evaluator.add_sample(predictions, ground_truth)

        per_sample.append({
            "tag": sample.tag,
            "image_path": str(sample.image_path),
            "pred_relations": len(predictions),
            "gt_relations": len(ground_truth),
            "processing_time_sec": round(time.time() - start, 3),
        })
        processed += 1

        if args.limit and processed >= args.limit:
            break

        if processed % 20 == 0:
            logger.info("Processed %d / %d frames", processed, len(dataset))

    metrics = evaluator.compute_metrics()
    output = {
        "split": args.split,
        "samples_evaluated": processed,
        "metrics": metrics,
        "samples": per_sample,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    logger.info("Saved metrics to %s", out_path)
    logger.info("R@20=%.3f | R@50=%.3f | R@100=%.3f", metrics.get("R@20", 0.0), metrics.get("R@50", 0.0), metrics.get("R@100", 0.0))


if __name__ == "__main__":
    main()
