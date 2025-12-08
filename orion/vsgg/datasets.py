"""Dataset helpers for Video Scene Graph experiments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from orion.data.action_genome import ActionGenomeFrameDataset, ActionGenomeSample

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Per-frame annotation bundle used by VidSGG modules."""

    frame_index: int
    boxes: np.ndarray  # [N, 4] xyxy
    labels: np.ndarray  # [N]
    relations: np.ndarray  # [M, 3] (subj_idx, obj_idx, predicate_id)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class VideoSceneGraphSample:
    """Video-level container (one or more frames)."""

    video_id: str
    frames: List[FrameData]
    metadata: Dict[str, object] = field(default_factory=dict)


class SceneGraphDataset:
    """Abstract base dataset for video scene graphs."""

    object_classes: Sequence[str]
    predicate_classes: Sequence[str]

    def __len__(self) -> int:  # pragma: no cover - satisfied in subclasses
        return len(self.samples)  # type: ignore[attr-defined]

    def __getitem__(self, idx: int) -> VideoSceneGraphSample:  # pragma: no cover - satisfied
        return self.samples[idx]  # type: ignore[attr-defined]

    @property
    def num_predicates(self) -> int:
        return len(self.predicate_classes)

    @property
    def num_objects(self) -> int:
        return len(self.object_classes)


class ActionGenomeSceneGraphDataset(SceneGraphDataset):
    """Wraps :class:`ActionGenomeFrameDataset` into VideoSceneGraphSample objects."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "test",
        max_samples: Optional[int] = None,
        group_by_video: bool = False,
        min_frames_per_video: int = 1,
        max_frames_per_video: Optional[int] = None,
        max_videos: Optional[int] = None,
    ) -> None:
        self._dataset = ActionGenomeFrameDataset(
            root=root,
            split=split,
            max_samples=max_samples,
            only_visible=True,
        )
        self.object_classes = list(self._dataset.object_classes)
        self.predicate_classes = list(self._dataset.relationship_classes)

        frame_records: List[Tuple[str, FrameData, Dict[str, object]]] = []
        for idx, sample in enumerate(self._dataset.samples):
            record = self._convert_sample(sample, idx)
            if record is None:
                continue
            frame_records.append(record)

        if group_by_video:
            self.samples = self._group_frames_by_video(
                frame_records,
                min_frames_per_video=min_frames_per_video,
                max_frames_per_video=max_frames_per_video,
            )
        else:
            self.samples = [
                VideoSceneGraphSample(video_id=vid, frames=[frame], metadata=metadata)
                for vid, frame, metadata in frame_records
            ]
        if max_videos is not None:
            self.samples = self.samples[:max_videos]

    # ------------------------------------------------------------------
    def _convert_sample(
        self,
        sample: ActionGenomeSample,
        ordinal: int,
    ) -> Optional[Tuple[str, FrameData, Dict[str, object]]]:
        boxes = sample.boxes.detach().cpu().numpy()
        labels = sample.labels.detach().cpu().numpy()
        relations = (
            sample.relations.detach().cpu().numpy()
            if sample.relations.numel()
            else np.empty((0, 3), dtype=np.int64)
        )
        frame_idx = _infer_frame_index(sample.tag, ordinal)
        frame = FrameData(
            frame_index=frame_idx,
            boxes=boxes,
            labels=labels,
            relations=relations,
            metadata={"image_path": str(sample.image_path)},
        )
        video_id = _infer_video_id(sample.tag)
        return video_id, frame, sample.metadata or {}

    @staticmethod
    def _group_frames_by_video(
        frame_records: Sequence[Tuple[str, FrameData, Dict[str, object]]],
        *,
        min_frames_per_video: int,
        max_frames_per_video: Optional[int],
    ) -> List[VideoSceneGraphSample]:
        grouped: Dict[str, Dict[str, object]] = {}
        for video_id, frame, metadata in frame_records:
            blob = grouped.setdefault(video_id, {"frames": [], "metadata": metadata})
            blob["frames"].append(frame)  # type: ignore[index]
        samples: List[VideoSceneGraphSample] = []
        for video_id, payload in grouped.items():
            frames: List[FrameData] = payload["frames"]  # type: ignore[assignment]
            frames.sort(key=lambda f: f.frame_index)
            if max_frames_per_video is not None:
                frames = frames[:max_frames_per_video]
            if len(frames) < max(1, min_frames_per_video):
                continue
            samples.append(
                VideoSceneGraphSample(
                    video_id=video_id,
                    frames=frames,
                    metadata=payload.get("metadata", {}),
                )
            )
        return samples


class SyntheticSceneGraphDataset(SceneGraphDataset):
    """Small procedurally-generated dataset for smoke tests."""

    def __init__(
        self,
        *,
        num_videos: int = 4,
        frames_per_video: int = 3,
        max_objects: int = 5,
        num_predicates: int = 6,
        seed: int = 7,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.object_classes = [f"obj_{i}" for i in range(max_objects)]
        self.predicate_classes = [f"pred_{i}" for i in range(num_predicates)]
        samples: List[VideoSceneGraphSample] = []
        for vid in range(num_videos):
            frames: List[FrameData] = []
            n_objects = rng.integers(2, max_objects + 1)
            for frame_idx in range(frames_per_video):
                boxes = rng.random((n_objects, 4)).astype(np.float32)
                boxes[:, 2:] = boxes[:, :2] + np.abs(boxes[:, 2:])
                labels = rng.integers(0, max_objects, size=(n_objects,), dtype=np.int64)
                rels = _random_relations(rng, n_objects, max(1, num_predicates))
                frames.append(
                    FrameData(
                        frame_index=frame_idx,
                        boxes=boxes,
                        labels=labels,
                        relations=rels,
                        metadata={"synthetic": True},
                    )
                )
            samples.append(
                VideoSceneGraphSample(
                    video_id=f"synthetic_{vid}",
                    frames=frames,
                    metadata={"synthetic": True},
                )
            )
        self.samples = samples


class _JSONSceneGraphDataset(SceneGraphDataset):
    """Shared loader for PVSG/VSGR-style JSON annotations."""

    def __init__(
        self,
        *,
        annotation_path: Path,
        object_classes: Sequence[str],
        predicate_classes: Sequence[str],
        max_videos: Optional[int] = None,
        min_frames_per_video: int = 1,
        max_frames_per_video: Optional[int] = None,
    ) -> None:
        self.object_classes = list(object_classes)
        self.predicate_classes = list(predicate_classes)
        entries = _load_json_annotations(annotation_path)
        samples: List[VideoSceneGraphSample] = []
        for entry in entries:
            video_id = str(entry.get("video_id") or entry.get("id") or entry.get("name"))
            frames_data = entry.get("frames") or []
            frames: List[FrameData] = []
            for idx, frame_entry in enumerate(frames_data):
                frame_index = int(
                    frame_entry.get("frame_index")
                    or frame_entry.get("frame_id")
                    or frame_entry.get("frame")
                    or idx
                )
                boxes = _parse_boxes(frame_entry)
                labels = _parse_labels(frame_entry, len(boxes))
                relations = _parse_relations(frame_entry)
                frames.append(
                    FrameData(
                        frame_index=frame_index,
                        boxes=boxes,
                        labels=labels,
                        relations=relations,
                        metadata=frame_entry.get("metadata", {}),
                    )
                )
            if not frames:
                continue
            frames.sort(key=lambda f: f.frame_index)
            if max_frames_per_video is not None:
                frames = frames[:max_frames_per_video]
            if len(frames) < max(1, min_frames_per_video):
                continue
            samples.append(
                VideoSceneGraphSample(
                    video_id=video_id,
                    frames=frames,
                    metadata=entry.get("metadata", {}),
                )
            )
            if max_videos is not None and len(samples) >= max_videos:
                break
        if not samples:
            raise RuntimeError(f"No samples were parsed from {annotation_path}")
        self.samples = samples


class PVSGSceneGraphDataset(_JSONSceneGraphDataset):
    """Loader for PVSG annotations stored under ``data/pvsg`` style directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "val",
        max_videos: Optional[int] = None,
        min_frames_per_video: int = 1,
        max_frames_per_video: Optional[int] = None,
    ) -> None:
        root = Path(root)
        annotation_path = root / "annotations" / f"{split}.json"
        object_classes = _load_class_list(root / "metadata" / "object_classes.txt")
        predicate_classes = _load_class_list(root / "metadata" / "predicate_classes.txt")
        if not object_classes or not predicate_classes:
            raise FileNotFoundError("PVSG metadata missing object/predicate classes text files")
        super().__init__(
            annotation_path=annotation_path,
            object_classes=object_classes,
            predicate_classes=predicate_classes,
            max_videos=max_videos,
            min_frames_per_video=min_frames_per_video,
            max_frames_per_video=max_frames_per_video,
        )


class VSGRSceneGraphDataset(_JSONSceneGraphDataset):
    """Loader for VSGR annotations (mirrors PVSG layout)."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "val",
        max_videos: Optional[int] = None,
        min_frames_per_video: int = 1,
        max_frames_per_video: Optional[int] = None,
    ) -> None:
        root = Path(root)
        annotation_path = root / "annotations" / f"{split}.json"
        object_classes = _load_class_list(root / "metadata" / "object_classes.txt")
        predicate_classes = _load_class_list(root / "metadata" / "predicate_classes.txt")
        if not object_classes or not predicate_classes:
            raise FileNotFoundError("VSGR metadata missing object/predicate classes text files")
        super().__init__(
            annotation_path=annotation_path,
            object_classes=object_classes,
            predicate_classes=predicate_classes,
            max_videos=max_videos,
            min_frames_per_video=min_frames_per_video,
            max_frames_per_video=max_frames_per_video,
        )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _infer_video_id(tag: str) -> str:
    if not tag:
        return "unknown"
    normalized = tag.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0]


def _infer_frame_index(tag: str, fallback: int) -> int:
    digits = "".join(ch for ch in str(tag) if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            return fallback
    return fallback


def _random_relations(rng: np.random.Generator, n_objects: int, num_predicates: int) -> np.ndarray:
    rels: List[Tuple[int, int, int]] = []
    for _ in range(max(1, n_objects - 1)):
        s = int(rng.integers(0, n_objects))
        o = int(rng.integers(0, n_objects))
        if s == o:
            o = (o + 1) % n_objects
        p = int(rng.integers(0, num_predicates))
        rels.append((s, o, p))
    if not rels:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(rels, dtype=np.int64)


def _load_class_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _load_json_annotations(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing annotation file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "videos" in data and isinstance(data["videos"], list):
            return data["videos"]
        return [dict(video_id=key, **value) for key, value in data.items()]
    raise ValueError(f"Unsupported annotation structure in {path}")


def _parse_boxes(frame_entry: Dict[str, object]) -> np.ndarray:
    boxes = frame_entry.get("boxes") or frame_entry.get("bbox") or []
    boxes_arr = np.asarray(boxes, dtype=np.float32)
    if boxes_arr.ndim != 2 or boxes_arr.shape[1] != 4:
        return np.empty((0, 4), dtype=np.float32)
    return boxes_arr


def _parse_labels(frame_entry: Dict[str, object], num_boxes: int) -> np.ndarray:
    labels = frame_entry.get("labels") or frame_entry.get("objects") or []
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.ndim != 1 or labels_arr.shape[0] != num_boxes:
        if num_boxes == 0:
            return np.empty((0,), dtype=np.int64)
        labels_arr = np.zeros((num_boxes,), dtype=np.int64)
    return labels_arr


def _parse_relations(frame_entry: Dict[str, object]) -> np.ndarray:
    relations = frame_entry.get("relations") or frame_entry.get("triplets") or []
    parsed: List[Tuple[int, int, int]] = []
    for rel in relations:
        subj = obj = predicate = None
        if isinstance(rel, dict):
            subj = rel.get("subject") or rel.get("subj") or rel.get("subject_idx")
            obj = rel.get("object") or rel.get("obj") or rel.get("object_idx")
            predicate = rel.get("predicate") or rel.get("predicate_id")
        elif isinstance(rel, (list, tuple)) and len(rel) >= 3:
            subj, obj, predicate = rel[:3]
        if subj is None or obj is None or predicate is None:
            continue
        parsed.append((int(subj), int(obj), int(predicate)))
    if not parsed:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(parsed, dtype=np.int64)


__all__ = [
    "FrameData",
    "VideoSceneGraphSample",
    "SceneGraphDataset",
    "ActionGenomeSceneGraphDataset",
    "SyntheticSceneGraphDataset",
    "PVSGSceneGraphDataset",
    "VSGRSceneGraphDataset",
]
