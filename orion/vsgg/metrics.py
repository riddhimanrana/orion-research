"""Generic R/mR evaluator for video scene graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RelationInstance:
    """Container describing a single relation prediction or ground truth."""

    subject_idx: int
    object_idx: int
    predicate_id: int
    subject_bbox: np.ndarray  # shape (4,)
    object_bbox: np.ndarray  # shape (4,)
    score: float = 1.0


class VideoRelationEvaluator:
    """Computes Recall/mRecall@K for arbitrary predicate vocabularies."""

    def __init__(
        self,
        *,
        top_ks: Sequence[int] = (20, 50, 100),
        iou_thresh: float = 0.5,
    ) -> None:
        self.top_ks = tuple(sorted(set(top_ks)))
        self.iou_thresh = float(iou_thresh)
        self._gt_total = 0
        self._hits: Dict[int, int] = {k: 0 for k in self.top_ks}
        self._predicate_totals: Dict[int, int] = {}
        self._predicate_hits: Dict[int, MutableMapping[int, int]] = {
            k: {} for k in self.top_ks
        }

    # ------------------------------------------------------------------
    def add_sample(
        self,
        *,
        predictions: Sequence[RelationInstance],
        ground_truth: Sequence[RelationInstance],
    ) -> None:
        if not ground_truth:
            return
        self._gt_total += len(ground_truth)
        for gt in ground_truth:
            self._predicate_totals[gt.predicate_id] = (
                self._predicate_totals.get(gt.predicate_id, 0) + 1
            )
        if not predictions:
            return
        preds_sorted = sorted(predictions, key=lambda inst: inst.score, reverse=True)
        for k in self.top_ks:
            matched, per_pred_hits = _match_relations(
                preds_sorted[:k], ground_truth, self.iou_thresh
            )
            self._hits[k] += matched
            bucket = self._predicate_hits[k]
            for predicate, count in per_pred_hits.items():
                bucket[predicate] = bucket.get(predicate, 0) + count

    # ------------------------------------------------------------------
    def compute_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self._gt_total == 0:
            for k in self.top_ks:
                metrics[f"R@{k}"] = 0.0
                metrics[f"mR@{k}"] = 0.0
            metrics["gt_relations"] = 0.0
            return metrics
        for k in self.top_ks:
            metrics[f"R@{k}"] = self._hits[k] / self._gt_total
            metrics[f"mR@{k}"] = _mean_recall(
                self._predicate_hits[k], self._predicate_totals
            )
        metrics["gt_relations"] = float(self._gt_total)
        return metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _match_relations(
    predictions: Sequence[RelationInstance],
    ground_truth: Sequence[RelationInstance],
    iou_thresh: float,
) -> Tuple[int, Dict[int, int]]:
    used = set()
    hits_by_predicate: Dict[int, int] = {}
    for pred in predictions:
        for idx, gt in enumerate(ground_truth):
            if idx in used:
                continue
            if pred.predicate_id != gt.predicate_id:
                continue
            if _bbox_iou(pred.subject_bbox, gt.subject_bbox) < iou_thresh:
                continue
            if _bbox_iou(pred.object_bbox, gt.object_bbox) < iou_thresh:
                continue
            used.add(idx)
            hits_by_predicate[gt.predicate_id] = hits_by_predicate.get(gt.predicate_id, 0) + 1
            break
    return len(used), hits_by_predicate


def _mean_recall(
    hits: Mapping[int, int],
    totals: Mapping[int, int],
) -> float:
    if not totals:
        return 0.0
    recalls: List[float] = []
    for predicate, total in totals.items():
        if total <= 0:
            continue
        recalls.append(hits.get(predicate, 0) / total)
    if not recalls:
        return 0.0
    return float(sum(recalls) / len(recalls))


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aw, ah = max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)
    bw, bh = max(0.0, bx2 - bx1), max(0.0, by2 - by1)
    union = aw * ah + bw * bh - inter + 1e-8
    return float(inter / union)


__all__ = ["RelationInstance", "VideoRelationEvaluator"]
