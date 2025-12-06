"""Action Genome dataset utilities.

Provides a lightweight frame-level dataset loader that maps the official
`object_bbox_and_relationship.pkl` annotations into PyTorch-friendly tensors.

The loader is intentionally defensive: annotations released over time have
slightly different structures, so we attempt to normalize the common variants
described in the official README. Each sample returns bounding boxes, object
labels, and subject/object/predicate triplets suitable for evaluation.
"""

from __future__ import annotations

import logging
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class ActionGenomeSample:
    """Represents a single annotated frame from Action Genome."""

    tag: str
    split: str
    image_path: Path
    boxes: torch.Tensor  # [N, 4] in xyxy format
    labels: torch.Tensor  # [N]
    relations: torch.Tensor  # [M, 3] -> (subject_idx, object_idx, predicate_id)
    metadata: Dict[str, Any]


class ActionGenomeFrameDataset(torch.utils.data.Dataset):
    """Frame-level dataset for Action Genome (AG).

    Parameters
    ----------
    root:
        Path to the ``dataset/ag`` directory containing ``frames`` and
        ``annotations``.
    split:
        Desired split. The annotations store the split in ``metadata['set']``;
        fall back to ``frame_list.txt`` ordering if unavailable.
    only_visible:
        If True, drop objects whose ``visible`` flag is False.
    min_box_area:
        Optional lower bound (in pixels^2) for object boxes. Boxes with area
        smaller than this threshold are filtered.
    include_person_boxes:
        Whether to merge boxes from ``person_bbox.pkl`` into the annotation.
    ``max_samples`` can be provided for quick smoke tests.
    auto_extract_frames:
        If True, missing frame PNGs will trigger an ffmpeg extraction pass for the
        corresponding video under ``root/videos``.
    ffmpeg_binary:
        Binary to invoke when extracting frames (default: ``ffmpeg``).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "test",
        only_visible: bool = True,
        min_box_area: float = 0.0,
        include_person_boxes: bool = False,
        max_samples: Optional[int] = None,
        auto_extract_frames: bool = False,
        ffmpeg_binary: str = "ffmpeg",
    ) -> None:
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        self.ann_dir = self.root / "annotations"

        self.only_visible = only_visible
        self.min_box_area = float(min_box_area)
        self.include_person_boxes = include_person_boxes
        self.auto_extract_frames = bool(auto_extract_frames)
        self.ffmpeg_binary = ffmpeg_binary
        self._extracted_videos: Set[str] = set()

        self.object_annotations = self._load_pickle(
            self.ann_dir / "object_bbox_and_relationship.pkl"
        )
        self.person_annotations = self._load_pickle(
            self.ann_dir / "person_bbox.pkl", required=False
        )

        self.frame_list = self._load_frame_list(self.ann_dir / "frame_list.txt")
        self.object_classes = self._load_class_list(self.ann_dir / "object_classes.txt")
        self.relationship_classes = self._load_class_list(
            self.ann_dir / "relationship_classes.txt"
        )

        self.obj_to_id = {name.lower(): idx for idx, name in enumerate(self.object_classes)}
        self.rel_to_id = {name.lower(): idx for idx, name in enumerate(self.relationship_classes)}

        split = split.lower()
        self.split = split

        # Flatten annotations into deterministic order
        samples: List[ActionGenomeSample] = []
        for tag in self.frame_list:
            if tag not in self.object_annotations:
                continue
            entry = self.object_annotations[tag]
            sample_split = self._infer_split(entry) or "unknown"
            if split != "all" and sample_split.lower() != split:
                continue
            try:
                sample = self._build_sample(tag, entry, sample_split)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Skipping frame %s (%s)", tag, exc)
                continue
            if sample is None:
                continue
            samples.append(sample)
            if max_samples is not None and len(samples) >= max_samples:
                break

        if not samples:
            raise RuntimeError(
                "No Action Genome samples were loaded. Ensure annotations exist "
                "and the requested split contains frames."
            )

        self.samples = samples

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ActionGenomeSample:
        return self.samples[idx]

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_pickle(path: Path, *, required: bool = True) -> Dict[str, Any]:
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing annotation file: {path}")
            logger.warning("Optional annotation file missing: %s", path)
            return {}
        with path.open("rb") as fh:
            return pickle.load(fh)

    @staticmethod
    def _load_frame_list(path: Path) -> List[str]:
        if not path.exists():
            raise FileNotFoundError(f"Missing frame list: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip()]

    @staticmethod
    def _load_class_list(path: Path) -> List[str]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip()]

    @staticmethod
    def _infer_split(entry: Sequence[Dict[str, Any]]) -> Optional[str]:
        for obj in entry:
            metadata = obj.get("metadata")
            if isinstance(metadata, dict):
                split = metadata.get("set")
                if split:
                    return split
        return None

    def _build_sample(
        self,
        tag: str,
        entry: Sequence[Dict[str, Any]],
        sample_split: str,
    ) -> Optional[ActionGenomeSample]:
        boxes: List[List[float]] = []
        labels: List[int] = []
        metadata = {"set": sample_split}

        index_map: Dict[int, int] = {}
        for raw_idx, obj in enumerate(entry):
            if self.only_visible and not bool(obj.get("visible", True)):
                continue
            bbox = obj.get("bbox")
            cls_name = str(obj.get("class", ""))
            if not bbox or len(bbox) != 4:
                continue
            cls_id = self._lookup_object_class(cls_name)
            if cls_id is None:
                continue
            x, y, w, h = map(float, bbox)
            if self.min_box_area > 0 and w * h < self.min_box_area:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(cls_id)
            index_map[raw_idx] = len(boxes) - 1

        if not boxes:
            return None

        relations = self._extract_relations(entry, index_map)

        image_rel = tag if isinstance(tag, str) else str(tag)
        if not image_rel.lower().endswith(".png"):
            image_rel = f"{image_rel}.png"
        image_path = self._resolve_image_path(Path(image_rel))
        if image_path is None or not image_path.exists():
            logger.warning("Missing frame %s; skipping sample", image_rel)
            return None

        return ActionGenomeSample(
            tag=tag,
            split=sample_split,
            image_path=image_path,
            boxes=torch.tensor(boxes, dtype=torch.float32),
            labels=torch.tensor(labels, dtype=torch.long),
            relations=torch.tensor(relations, dtype=torch.long) if relations else torch.empty((0, 3), dtype=torch.long),
            metadata=metadata,
        )

    def _lookup_object_class(self, class_name: str) -> Optional[int]:
        normalized = class_name.strip().lower()
        if normalized in self.obj_to_id:
            return self.obj_to_id[normalized]
        # Basic fallback for pluralization
        if normalized.endswith("s") and normalized[:-1] in self.obj_to_id:
            return self.obj_to_id[normalized[:-1]]
        return None

    def _extract_relations(
        self,
        entry: Sequence[Dict[str, Any]],
        index_map: Dict[int, int],
    ) -> List[Tuple[int, int, int]]:
        relations: List[Tuple[int, int, int]] = []
        for subj_idx, obj in enumerate(entry):
            subj_remapped = index_map.get(subj_idx)
            if subj_remapped is None:
                continue
            for rel_type in (
                "attention_relationship",
                "spatial_relationship",
                "contacting_relationship",
            ):
                rel_entries = obj.get(rel_type, [])
                if not rel_entries:
                    continue
                normalized_type = rel_type.replace("_relationship", "")
                for rel_obj_idx, predicate in self._normalize_relation_entries(rel_entries):
                    if rel_obj_idx is None or predicate is None:
                        continue
                    obj_remapped = index_map.get(rel_obj_idx)
                    if obj_remapped is None:
                        continue
                    pred_name = str(predicate).strip().lower()
                    predicate_id = self.rel_to_id.get(pred_name)
                    if predicate_id is None:
                        # Some releases omit the relationship namespace, so as a
                        # fallback register "attention:looking_at" style labels on-the-fly.
                        compound = f"{normalized_type}:{pred_name}"
                        if compound not in self.rel_to_id:
                            continue
                        predicate_id = self.rel_to_id[compound]
                    relations.append((subj_remapped, obj_remapped, predicate_id))
        return relations

    @staticmethod
    def _normalize_relation_entries(rel_entries: Iterable[Any]) -> Iterable[Tuple[Optional[int], Optional[str]]]:
        for rel in rel_entries:
            target_idx: Optional[int] = None
            predicate: Optional[str] = None
            if isinstance(rel, dict):
                target_idx = ActionGenomeFrameDataset._safe_int(
                    rel.get("object")
                    or rel.get("object_id")
                    or rel.get("object_index")
                    or rel.get("object_idx")
                )
                predicate = rel.get("predicate") or rel.get("name") or rel.get("class")
            elif isinstance(rel, (list, tuple)):
                if len(rel) >= 2:
                    predicate = str(rel[0])
                    target_idx = ActionGenomeFrameDataset._safe_int(rel[1])
            elif isinstance(rel, str):
                predicate = rel
                target_idx = None
            else:
                continue
            yield target_idx, predicate

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _resolve_image_path(self, rel_path: Path) -> Optional[Path]:
        image_path = self.frames_dir / rel_path
        if image_path.exists():
            return image_path
        if not self.auto_extract_frames:
            return image_path if image_path.exists() else None
        if not rel_path.parts:
            return None
        video_name = rel_path.parts[0]
        if not video_name:
            return None
        video_path = self.root / "videos" / video_name
        if not video_path.exists():
            logger.warning("Video file missing for %s", video_path)
            return None
        if video_name not in self._extracted_videos:
            self._extracted_videos.add(video_name)
            output_dir = self.frames_dir / video_name
            output_dir.mkdir(parents=True, exist_ok=True)
            self._extract_video_frames(video_path, output_dir)
        return image_path if image_path.exists() else None

    def _extract_video_frames(self, video_path: Path, output_dir: Path) -> None:
        pattern = str(output_dir / "%06d.png")
        cmd = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vsync",
            "0",
            pattern,
        ]
        try:
            subprocess.run(cmd, check=True)
            logger.info("Extracted frames for %s", video_path.name)
        except FileNotFoundError:
            logger.warning("ffmpeg not found while extracting %s", video_path)
        except subprocess.CalledProcessError as exc:
            logger.warning("ffmpeg failed for %s: %s", video_path, exc)


__all__ = ["ActionGenomeFrameDataset", "ActionGenomeSample"]
