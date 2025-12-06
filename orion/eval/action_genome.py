"""Action Genome evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from orion.data.action_genome import ActionGenomeFrameDataset, ActionGenomeSample
from orion.perception.types import Observation, PerceptionResult


@dataclass(frozen=True)
class AGTriplet:
    """Represents either a ground-truth or predicted relation triplet."""

    subject_bbox: np.ndarray
    object_bbox: np.ndarray
    subject_class: str
    object_class: str
    predicate: str
    score: float = 1.0


@dataclass(frozen=True)
class _AGDetection:
    bbox: np.ndarray
    label: str
    confidence: float


class ActionGenomeLabelMapper:
    """Maps Orion detection classes (COCO) into Action Genome's taxonomy."""

    def __init__(self, overrides: Optional[Mapping[str, str]] = None) -> None:
        mapping: Dict[str, str] = {
            "person": "person",
            "backpack": "bag",
            "handbag": "bag",
            "suitcase": "bag",
            "bed": "bed",
            "book": "book",
            "bottle": "cupglassbottle",
            "wine glass": "cupglassbottle",
            "cup": "cupglassbottle",
            "chair": "chair",
            "couch": "sofacouch",
            "bench": "sofacouch",
            "dining table": "table",
            "laptop": "laptop",
            "tv": "television",
            "refrigerator": "refrigerator",
            "cell phone": "phonecamera",
            "remote": "phonecamera",
            "sandwich": "sandwich",
            "shoe": "shoe",
            "teddy bear": "pillow",
            "mirror": "mirror",
            "umbrella": "clothes",
        }
        if overrides:
            mapping.update({k.strip().lower(): v for k, v in overrides.items()})
        self._mapping = mapping

    def map_detection(self, class_name: str) -> Optional[str]:
        key = class_name.strip().lower()
        return self._mapping.get(key)


def build_prediction_triplets(
    result: PerceptionResult,
    mapper: ActionGenomeLabelMapper,
) -> List[AGTriplet]:
    detections: List[_AGDetection] = []
    for obs in result.raw_observations:
        ag_label = mapper.map_detection(obs.object_class.value)
        if not ag_label:
            continue
        bbox = np.array(obs.bounding_box.to_list(), dtype=np.float32)
        detections.append(
            _AGDetection(
                bbox=bbox,
                label=ag_label,
                confidence=float(obs.confidence),
            )
        )
    return _infer_relations(detections)


def build_ground_truth_triplets(
    sample: ActionGenomeSample,
    dataset: ActionGenomeFrameDataset,
) -> List[AGTriplet]:
    if sample.relations.numel() == 0:
        return []
    boxes = sample.boxes.detach().cpu().numpy()
    labels = sample.labels.detach().cpu().numpy()
    relations = sample.relations.detach().cpu().numpy()
    obj_classes = dataset.object_classes
    rel_classes = dataset.relationship_classes
    triplets: List[AGTriplet] = []
    for subj_idx, obj_idx, predicate_idx in relations.tolist():
        if subj_idx >= len(labels) or obj_idx >= len(labels):
            continue
        subject_cls = obj_classes[labels[subj_idx]] if labels[subj_idx] < len(obj_classes) else None
        object_cls = obj_classes[labels[obj_idx]] if labels[obj_idx] < len(obj_classes) else None
        if subject_cls is None or object_cls is None:
            continue
        predicate = rel_classes[predicate_idx] if predicate_idx < len(rel_classes) else None
        if predicate is None:
            continue
        triplets.append(
            AGTriplet(
                subject_bbox=boxes[subj_idx],
                object_bbox=boxes[obj_idx],
                subject_class=subject_cls,
                object_class=object_cls,
                predicate=predicate,
                score=1.0,
            )
        )
    return triplets


class ActionGenomeRelationEvaluator:
    """Computes SGDET Recall/mRecall for Action Genome triplets."""

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
        self._predicate_totals: Dict[str, int] = {}
        self._predicate_hits: Dict[int, MutableMapping[str, int]] = {
            k: {} for k in self.top_ks
        }

    def add_sample(
        self,
        predictions: Sequence[AGTriplet],
        ground_truth: Sequence[AGTriplet],
    ) -> None:
        if not ground_truth:
            return
        self._gt_total += len(ground_truth)
        for gt in ground_truth:
            self._predicate_totals[gt.predicate] = (
                self._predicate_totals.get(gt.predicate, 0) + 1
            )
        if not predictions:
            return
        preds_sorted = sorted(predictions, key=lambda t: t.score, reverse=True)
        for k in self.top_ks:
            matched, per_pred_hits = _match_triplets(
                preds_sorted[:k], ground_truth, self.iou_thresh
            )
            self._hits[k] += matched
            for predicate, count in per_pred_hits.items():
                bucket = self._predicate_hits[k]
                bucket[predicate] = bucket.get(predicate, 0) + count

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

def _infer_relations(detections: Sequence[_AGDetection]) -> List[AGTriplet]:
    triplets: List[AGTriplet] = []
    n = len(detections)
    for i in range(n):
        subj = detections[i]
        for j in range(n):
            if i == j:
                continue
            obj = detections[j]
            base_score = subj.confidence * obj.confidence
            preds = _relation_rules(subj, obj)
            for predicate, bonus in preds:
                triplets.append(
                    AGTriplet(
                        subject_bbox=subj.bbox,
                        object_bbox=obj.bbox,
                        subject_class=subj.label,
                        object_class=obj.label,
                        predicate=predicate,
                        score=max(1e-6, base_score * bonus),
                    )
                )
    return triplets


def _relation_rules(subj: _AGDetection, obj: _AGDetection) -> List[Tuple[str, float]]:
    preds: List[Tuple[str, float]] = []
    iou_val = _bbox_iou(subj.bbox, obj.bbox)
    subj_center = _bbox_center(subj.bbox)
    obj_center = _bbox_center(obj.bbox)
    overlap = _horiz_overlap(subj.bbox, obj.bbox)
    vert_gap = obj.bbox[1] - subj.bbox[3]

    if subj.label == "person" and obj.label != "person":
        if iou_val >= 0.05 or _point_in_box(obj_center, subj.bbox):
            preds.append(("holding", max(iou_val, 0.1)))
        support_classes = {"chair", "sofacouch", "bed"}
        if obj.label in support_classes and subj.bbox[3] >= obj.bbox[1] and overlap >= 0.3:
            preds.append(("sittingon", 0.3 + 0.7 * overlap))

    if iou_val >= 0.1:
        preds.append(("touching", iou_val))

    if subj.bbox[3] + 5 <= obj.bbox[1]:
        preds.append(("above", 0.5))
    if obj.bbox[3] + 5 <= subj.bbox[1]:
        preds.append(("beneath", 0.5))

    if _box_inside(subj.bbox, obj.bbox, threshold=0.85):
        preds.append(("in", 0.7))

    return preds


def _match_triplets(
    predictions: Sequence[AGTriplet],
    ground_truth: Sequence[AGTriplet],
    iou_thresh: float,
) -> Tuple[int, Dict[str, int]]:
    used = set()
    hits_by_predicate: Dict[str, int] = {}
    for pred in predictions:
        for idx, gt in enumerate(ground_truth):
            if idx in used:
                continue
            if pred.predicate != gt.predicate:
                continue
            if _bbox_iou(pred.subject_bbox, gt.subject_bbox) < iou_thresh:
                continue
            if _bbox_iou(pred.object_bbox, gt.object_bbox) < iou_thresh:
                continue
            used.add(idx)
            hits_by_predicate[gt.predicate] = hits_by_predicate.get(gt.predicate, 0) + 1
            break
    return len(used), hits_by_predicate


def _mean_recall(
    hits: Mapping[str, int],
    totals: Mapping[str, int],
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


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _point_in_box(point: Tuple[float, float], bbox: Sequence[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _box_inside(
    inner: Sequence[float],
    outer: Sequence[float],
    *,
    threshold: float,
) -> bool:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter_x1, inter_y1 = max(ix1, ox1), max(iy1, oy1)
    inter_x2, inter_y2 = min(ix2, ox2), min(iy2, oy2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    if iw <= 0 or ih <= 0:
        return False
    inter_area = iw * ih
    inner_area = max(1e-8, (ix2 - ix1) * (iy2 - iy1))
    return (inter_area / inner_area) >= threshold


def _horiz_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    denom = max(1e-6, min(ax2 - ax1, bx2 - bx1))
    return float(overlap / denom)
