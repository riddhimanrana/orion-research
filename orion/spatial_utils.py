"""Shared spatial reasoning heuristics for Orion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple, Union


@dataclass
class SpatialZone:
    """Spatial zone classification with coarse horizontal/vertical context."""

    zone_type: str
    confidence: float
    x_position: str
    y_position: str
    reasoning: List[str] = field(default_factory=list)


BBoxLike = Union[Sequence[float], Dict[str, float]]


def _extract_bbox(bbox: BBoxLike) -> Tuple[float, float, float, float]:
    if isinstance(bbox, dict):
        return (
            float(bbox.get("x1", 0.0)),
            float(bbox.get("y1", 0.0)),
            float(bbox.get("x2", 0.0)),
            float(bbox.get("y2", 0.0)),
        )
    if len(bbox) < 4:
        return 0.0, 0.0, 0.0, 0.0
    return tuple(float(v) for v in bbox[:4])  # type: ignore[return-value]


def _resolve_frame_dims(
    bbox: BBoxLike, frame_width: float | None, frame_height: float | None
) -> Tuple[float, float]:
    if frame_width is None or frame_height is None:
        if isinstance(bbox, dict):
            frame_width = float(bbox.get("frame_width", 1920.0))
            frame_height = float(bbox.get("frame_height", 1080.0))
        else:
            frame_width = frame_width or 1920.0
            frame_height = frame_height or 1080.0
    return frame_width or 1920.0, frame_height or 1080.0


def calculate_spatial_zone_from_bbox(
    bbox: BBoxLike,
    frame_width: float | None = None,
    frame_height: float | None = None,
) -> SpatialZone:
    """Classify spatial zone using the contextual engine heuristics."""

    x1, y1, x2, y2 = _extract_bbox(bbox)
    frame_width, frame_height = _resolve_frame_dims(bbox, frame_width, frame_height)

    if frame_width <= 0 or frame_height <= 0:
        return SpatialZone("unknown", 0.0, "unknown", "unknown", ["Invalid frame size"])

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Normalize to [0, 1]
    nx = cx / frame_width
    ny = cy / frame_height

    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))

    horizontal = "left" if nx < 0.33 else ("right" if nx > 0.66 else "center")

    if ny < 0.15:
        return SpatialZone(
            "ceiling",
            0.9,
            horizontal,
            "top",
            [f"Top {ny * 100:.1f}% of frame"],
        )
    if ny < 0.35:
        return SpatialZone(
            "wall_upper",
            0.85,
            horizontal,
            "top",
            [f"Upper wall ({ny * 100:.1f}% from top)"],
        )
    if ny < 0.65:
        return SpatialZone(
            "wall_middle",
            0.9,
            horizontal,
            "middle",
            [f"Mid-height ({ny * 100:.1f}%) - typical for hardware"],
        )
    if ny < 0.75:
        return SpatialZone(
            "wall_lower",
            0.85,
            horizontal,
            "middle",
            [f"Lower wall ({ny * 100:.1f}%)"],
        )
    return SpatialZone(
        "floor",
        0.9,
        horizontal,
        "bottom",
        [f"Near floor ({ny * 100:.1f}%)"],
    )


def infer_scene_type(object_classes: Iterable[str]) -> str:
    """Infer scene type using contextual engine heuristics."""

    class_counter: Dict[str, int] = {}
    for cls in object_classes:
        if not cls:
            continue
        key = cls.lower()
        class_counter[key] = class_counter.get(key, 0) + 1

    if not class_counter:
        return "general"

    def _score(required: Sequence[str]) -> int:
        return sum(1 for name in required if class_counter.get(name, 0) > 0)

    if _score(["oven", "microwave", "refrigerator", "stove", "sink"]) >= 2:
        return "kitchen"
    if _score(["bed", "nightstand", "dresser"]) >= 1:
        return "bedroom"
    if _score(["laptop", "keyboard", "mouse", "monitor", "desk"]) >= 2:
        return "office"
    if _score(["couch", "tv", "sofa"]) >= 1:
        return "living_room"
    return "general"