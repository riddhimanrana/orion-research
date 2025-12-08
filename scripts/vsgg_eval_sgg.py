"""Evaluate a saved SGG checkpoint on a specified dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

from orion.vsgg.datasets import (
    ActionGenomeSceneGraphDataset,
    PVSGSceneGraphDataset,
    SyntheticSceneGraphDataset,
    VSGRSceneGraphDataset,
)
from orion.vsgg.metrics import RelationInstance, VideoRelationEvaluator
from orion.vsgg.models import PairwiseRelationModel, build_pair_feature

SUPPORTED_DATASETS = ("action_genome", "pvsg", "vsgr", "synthetic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Video Scene Graph Generation checkpoints.")
    parser.add_argument("--checkpoint", required=True, help="Path to the saved model (.pt)")
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="synthetic",
        help="Dataset to evaluate on.",
    )
    parser.add_argument("--data-root", default="data/action_genome", help="Dataset root directory.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument(
        "--topk",
        type=int,
        nargs="*",
        default=[20, 50, 100],
        help="Recall@K values to report.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Optional path to write metrics JSON.")
    return parser.parse_args()


def load_dataset(
    name: str,
    data_root: str,
    split: str,
    max_samples: int | None,
    max_videos: int | None,
    max_frames_per_video: int | None,
):
    if name == "action_genome":
        return ActionGenomeSceneGraphDataset(
            data_root,
            split=split,
            max_samples=max_samples,
            max_videos=max_videos,
        )
    if name == "pvsg":
        return PVSGSceneGraphDataset(
            data_root,
            split=split,
            max_videos=max_videos,
            max_frames_per_video=max_frames_per_video,
        )
    if name == "vsgr":
        return VSGRSceneGraphDataset(
            data_root,
            split=split,
            max_videos=max_videos,
            max_frames_per_video=max_frames_per_video,
        )
    if name == "synthetic":
        return SyntheticSceneGraphDataset()
    raise ValueError(f"Unsupported dataset: {name}")


def load_model(checkpoint_path: Path, device: torch.device, num_predicates: int, feature_dim: int) -> PairwiseRelationModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_dim = int(checkpoint.get("feature_dim", feature_dim))
    num_predicates = int(checkpoint.get("num_predicates", num_predicates))
    model = PairwiseRelationModel(feature_dim=feature_dim, num_predicates=num_predicates)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate(
    model: PairwiseRelationModel,
    samples,
    device: torch.device,
    topks: Sequence[int],
) -> dict:
    evaluator = VideoRelationEvaluator(top_ks=topks)
    max_topk = max(topks)
    with torch.inference_mode():
        for sample in samples:
            for frame in sample.frames:
                predictions = _predict_frame_relations(model, frame, device, max_topk)
                ground_truth = _gt_relations(frame)
                evaluator.add_sample(predictions=predictions, ground_truth=ground_truth)
    return evaluator.compute_metrics()


def _gt_relations(frame) -> List[RelationInstance]:
    rels: List[RelationInstance] = []
    for subj_idx, obj_idx, predicate in frame.relations.tolist():
        rels.append(
            RelationInstance(
                subject_idx=int(subj_idx),
                object_idx=int(obj_idx),
                predicate_id=int(predicate),
                subject_bbox=frame.boxes[subj_idx],
                object_bbox=frame.boxes[obj_idx],
                score=1.0,
            )
        )
    return rels


def _predict_frame_relations(
    model: PairwiseRelationModel,
    frame,
    device: torch.device,
    top_k_predicates: int,
) -> List[RelationInstance]:
    n = len(frame.boxes)
    preds: List[RelationInstance] = []
    if n <= 1:
        return preds
    boxes = frame.boxes
    labels = frame.labels
    for subj_idx in range(n):
        for obj_idx in range(n):
            if subj_idx == obj_idx:
                continue
            feat = torch.from_numpy(build_pair_feature(boxes, labels, subj_idx, obj_idx)).to(device).unsqueeze(0)
            logits = model(feat).squeeze(0)
            scores = torch.softmax(logits, dim=-1)
            k = min(top_k_predicates, scores.numel())
            top_scores, top_idx = torch.topk(scores, k=k)
            subj_bbox = boxes[subj_idx]
            obj_bbox = boxes[obj_idx]
            for score, predicate in zip(top_scores.tolist(), top_idx.tolist()):
                preds.append(
                    RelationInstance(
                        subject_idx=subj_idx,
                        object_idx=obj_idx,
                        predicate_id=int(predicate),
                        subject_bbox=subj_bbox,
                        object_bbox=obj_bbox,
                        score=float(score),
                    )
                )
    return preds


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = load_dataset(
        args.dataset,
        args.data_root,
        args.split,
        args.max_samples,
        args.max_videos,
        args.max_frames_per_video,
    )
    checkpoint_path = Path(args.checkpoint)
    model = load_model(
        checkpoint_path,
        device,
        num_predicates=dataset.num_predicates,
        feature_dim=len(build_pair_feature(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.int64), 0, 1)),
    )
    metrics = evaluate(model, dataset.samples, device, args.topk)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Metrics written to {output_path}")


if __name__ == "__main__":
    main()
