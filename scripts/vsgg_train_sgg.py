"""Baseline trainer for Video Scene Graph Generation (SGG)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from orion.vsgg.datasets import (
    ActionGenomeSceneGraphDataset,
    PVSGSceneGraphDataset,
    SyntheticSceneGraphDataset,
    VSGRSceneGraphDataset,
    VideoSceneGraphSample,
)
from orion.vsgg.metrics import RelationInstance, VideoRelationEvaluator
from orion.vsgg.models import PairwiseRelationModel, build_pair_feature

SUPPORTED_DATASETS = ("action_genome", "pvsg", "vsgr", "synthetic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple SGG baseline on Orion datasets.")
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="synthetic",
        help="Dataset to use. Real datasets require assets under data/.",
    )
    parser.add_argument("--data-root", default="data/action_genome", help="Path to dataset root (if applicable).")
    parser.add_argument("--split", default="train", help="Dataset split (Action Genome only).")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for quick tests.")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos (PVSG/VSGR only).")
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="Subsample each video to at most this many frames (PVSG/VSGR).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="checkpoints/vsgg_sgg.pt", help="Checkpoint path to save model.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training.")
    parser.add_argument(
        "--topk",
        type=int,
        nargs="*",
        default=[20, 50, 100],
        help="Recall@K values to report (evaluation).",
    )
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
        return SyntheticSceneGraphDataset(num_videos=6)
    raise ValueError(f"Unsupported dataset: {name}")


def collect_relation_examples(samples: Sequence[VideoSceneGraphSample]) -> Tuple[np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    for sample in samples:
        for frame in sample.frames:
            if frame.relations.size == 0:
                continue
            for subj_idx, obj_idx, predicate in frame.relations.tolist():
                feat = build_pair_feature(frame.boxes, frame.labels, subj_idx, obj_idx)
                features.append(feat)
                labels.append(int(predicate))
    if not features:
        raise RuntimeError("Dataset produced no relation annotations to train on.")
    return np.stack(features, axis=0), np.array(labels, dtype=np.int64)


def make_dataloader(features: np.ndarray, labels: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def evaluate_model(
    model: PairwiseRelationModel,
    samples: Sequence[VideoSceneGraphSample],
    device: torch.device,
    topks: Sequence[int],
) -> dict:
    evaluator = VideoRelationEvaluator(top_ks=topks)
    model.eval()
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
    features, labels = collect_relation_examples(dataset.samples)
    dataloader = make_dataloader(features, labels, args.batch_size)

    feature_dim = features.shape[1]
    model = PairwiseRelationModel(feature_dim=feature_dim, num_predicates=dataset.num_predicates)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_features.size(0)
        avg_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch:02d}/{args.epochs} - loss: {avg_loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "feature_dim": feature_dim,
        "num_predicates": dataset.num_predicates,
    }
    torch.save(payload, output_path)
    print(f"Saved checkpoint to {output_path}")

    if args.eval:
        metrics = evaluate_model(model, dataset.samples, device, args.topk)
        metrics_path = output_path.with_suffix(".metrics.json")
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Evaluation metrics saved to {metrics_path}")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
