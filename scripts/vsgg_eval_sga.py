"""Evaluate Scene Graph Anticipation checkpoints."""

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
from orion.vsgg.models import TemporalRelationForecaster, build_pair_feature

SUPPORTED_DATASETS = ("action_genome", "pvsg", "vsgr", "synthetic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SGA checkpoints (currently synthetic).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="synthetic")
    parser.add_argument("--data-root", default="data/action_genome")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--min-frames-per-video", type=int, default=2)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument(
        "--topk",
        type=int,
        nargs="*",
        default=[10, 20, 50],
        help="Recall@K values to report.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_dataset(
    name: str,
    data_root: str,
    split: str,
    max_samples: int | None,
    max_videos: int | None,
    min_frames_per_video: int,
    max_frames_per_video: int | None,
):
    if name == "action_genome":
        return ActionGenomeSceneGraphDataset(
            data_root,
            split=split,
            max_samples=max_samples,
            group_by_video=True,
            min_frames_per_video=min_frames_per_video,
            max_frames_per_video=max_frames_per_video,
            max_videos=max_videos,
        )
    if name == "pvsg":
        return PVSGSceneGraphDataset(
            data_root,
            split=split,
            max_videos=max_videos,
            min_frames_per_video=min_frames_per_video,
            max_frames_per_video=max_frames_per_video,
        )
    if name == "vsgr":
        return VSGRSceneGraphDataset(
            data_root,
            split=split,
            max_videos=max_videos,
            min_frames_per_video=min_frames_per_video,
            max_frames_per_video=max_frames_per_video,
        )
    if name == "synthetic":
        return SyntheticSceneGraphDataset(num_videos=12, frames_per_video=4, max_objects=4, num_predicates=5)
    raise ValueError(f"Unsupported dataset: {name}")


def load_model(checkpoint_path: Path, device: torch.device, feature_dim: int, num_predicates: int) -> TemporalRelationForecaster:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_dim = int(checkpoint.get("feature_dim", feature_dim))
    num_predicates = int(checkpoint.get("num_predicates", num_predicates))
    model = TemporalRelationForecaster(feature_dim=feature_dim, num_predicates=num_predicates)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate(
    model: TemporalRelationForecaster,
    samples,
    device: torch.device,
    fraction: float,
    topks: Sequence[int],
) -> dict:
    evaluator = VideoRelationEvaluator(top_ks=topks)
    max_topk = max(topks)
    with torch.inference_mode():
        for sample in samples:
            total_frames = len(sample.frames)
            if total_frames < 2:
                continue
            obs_len = max(1, int(np.floor(total_frames * fraction)))
            obs_len = min(obs_len, total_frames - 1)
            obs_frames = sample.frames[:obs_len]
            target_frame = sample.frames[obs_len]
            predictions = _predict_future_relations(model, obs_frames, target_frame, device, max_topk)
            ground_truth = _gt_relations(target_frame)
            evaluator.add_sample(predictions=predictions, ground_truth=ground_truth)
    return evaluator.compute_metrics()


def _predict_future_relations(
    model: TemporalRelationForecaster,
    observed_frames,
    target_frame,
    device: torch.device,
    top_k_predicates: int,
) -> List[RelationInstance]:
    n = len(target_frame.boxes)
    preds: List[RelationInstance] = []
    for subj_idx in range(n):
        for obj_idx in range(n):
            if subj_idx == obj_idx:
                continue
            sequence = _pair_sequence(observed_frames, subj_idx, obj_idx)
            if sequence is None:
                continue
            seq_tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
            logits = model(seq_tensor).squeeze(0)
            scores = torch.softmax(logits, dim=-1)
            k = min(top_k_predicates, scores.numel())
            top_scores, top_idx = torch.topk(scores, k=k)
            subj_bbox = target_frame.boxes[subj_idx]
            obj_bbox = target_frame.boxes[obj_idx]
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


def _pair_sequence(frames, subj_idx: int, obj_idx: int) -> np.ndarray | None:
    sequence = []
    for frame in frames:
        if subj_idx >= len(frame.boxes) or obj_idx >= len(frame.boxes):
            return None
        sequence.append(build_pair_feature(frame.boxes, frame.labels, subj_idx, obj_idx))
    if not sequence:
        return None
    return np.stack(sequence, axis=0)


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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = load_dataset(
        args.dataset,
        args.data_root,
        args.split,
        args.max_samples,
        args.max_videos,
        args.min_frames_per_video,
        args.max_frames_per_video,
    )
    feature_dim = len(build_pair_feature(np.zeros((2, 4)), np.zeros(2), 0, 1))
    model = load_model(Path(args.checkpoint), device, feature_dim, dataset.num_predicates)
    metrics = evaluate(model, dataset.samples, device, args.fraction, args.topk)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
