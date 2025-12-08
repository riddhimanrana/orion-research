"""Baseline trainer for Scene Graph Anticipation (SGA)."""

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
from orion.vsgg.models import TemporalRelationForecaster, build_pair_feature

SUPPORTED_DATASETS = ("action_genome", "pvsg", "vsgr", "synthetic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a toy Scene Graph Anticipation model.")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="synthetic")
    parser.add_argument("--data-root", default="data/action_genome", help="Dataset root (unused for synthetic).")
    parser.add_argument("--split", default="train", help="Dataset split for real corpora.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit Action Genome frames before grouping.")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of aggregated videos.")
    parser.add_argument(
        "--min-frames-per-video",
        type=int,
        default=2,
        help="Filter out videos with fewer than this many frames.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="Truncate each video to this many frames for efficiency.",
    )
    parser.add_argument("--fraction", type=float, default=0.5, help="Observed input fraction F in [0,1].")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="checkpoints/vsgg_sga.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--topk",
        type=int,
        nargs="*",
        default=[10, 20, 50],
        help="Recall@K values to report during evaluation.",
    )
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training.")
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
        return SyntheticSceneGraphDataset(num_videos=8, frames_per_video=4, max_objects=4, num_predicates=5)
    raise ValueError(f"Unsupported dataset: {name}")


def collect_sequences(
    samples: Sequence[VideoSceneGraphSample],
    fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    seqs: List[np.ndarray] = []
    labels: List[int] = []
    for sample in samples:
        total_frames = len(sample.frames)
        if total_frames < 2:
            continue
        obs_len = max(1, int(np.floor(total_frames * fraction)))
        obs_len = min(obs_len, total_frames - 1)
        obs_frames = sample.frames[:obs_len]
        target_frame = sample.frames[obs_len]
        if target_frame.relations.size == 0:
            continue
        for subj_idx, obj_idx, predicate in target_frame.relations.tolist():
            sequence = _pair_sequence(obs_frames, subj_idx, obj_idx)
            if sequence is None:
                continue
            seqs.append(sequence)
            labels.append(int(predicate))
    if not seqs:
        raise RuntimeError("No anticipation sequences were generated; adjust fraction or dataset size.")
    return np.stack(seqs, axis=0), np.array(labels, dtype=np.int64)


def _pair_sequence(frames, subj_idx: int, obj_idx: int) -> np.ndarray | None:
    sequence: List[np.ndarray] = []
    for frame in frames:
        if subj_idx >= len(frame.boxes) or obj_idx >= len(frame.boxes):
            return None
        sequence.append(build_pair_feature(frame.boxes, frame.labels, subj_idx, obj_idx))
    if not sequence:
        return None
    return np.stack(sequence, axis=0)


def evaluate_model(
    model: TemporalRelationForecaster,
    samples: Sequence[VideoSceneGraphSample],
    device: torch.device,
    fraction: float,
    topks: Sequence[int],
) -> dict:
    evaluator = VideoRelationEvaluator(top_ks=topks)
    max_topk = max(topks)
    model.eval()
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
    sequences, labels = collect_sequences(dataset.samples, args.fraction)

    seq_tensor = torch.from_numpy(sequences).float()
    label_tensor = torch.from_numpy(labels).long()
    loader = DataLoader(
        TensorDataset(seq_tensor, label_tensor),
        batch_size=args.batch_size,
        shuffle=True,
    )

    feature_dim = sequences.shape[-1]
    model = TemporalRelationForecaster(feature_dim=feature_dim, num_predicates=dataset.num_predicates)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_features.size(0)
        avg_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch:02d}/{args.epochs} - loss: {avg_loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "feature_dim": feature_dim,
        "num_predicates": dataset.num_predicates,
    }
    torch.save(payload, output_path)
    print(f"Saved SGA checkpoint to {output_path}")

    if args.eval:
        metrics = evaluate_model(model, dataset.samples, device, args.fraction, args.topk)
        metrics_path = output_path.with_suffix(".metrics.json")
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Evaluation metrics saved to {metrics_path}")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
