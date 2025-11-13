"""TapNet / BootsTAPIR integration stub for Orion.

This module wraps a causal (online) TapNet/TAPIR style point tracker.
Because network access / checkpoint download may be blocked, we require the
user to place the PyTorch checkpoint locally and pass `tapnet_checkpoint_path`
via `PerceptionConfig`.

Initial scope:
- Sample query points from detection centroids (optionally more per bbox)
- Maintain per-point trajectories (x, y per frame, visibility flag)
- Promote stable point clusters into entity-level track candidates

Future extensions:
- Rainbow visualization (camera-motion compensated trajectories)
- Foreground/background segmentation using motion consistency
- 3D lift using depth map (if available)

NOTE: This is a lightweight scaffold; real TapNet inference requires model
architecture & weight loading (ResNet18 + correlation refinement). We abstract
that behind `_model_forward` so we can later drop in official code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

@dataclass
class PointTrack:
    track_id: int
    query_xy: Tuple[float, float]  # initial coordinates (frame 0 in raster pixels)
    frames: List[int]
    x: List[float]
    y: List[float]
    visibility: List[bool]
    last_confidence: float

    def append(self, frame_number: int, x: float, y: float, visible: bool, confidence: float):
        self.frames.append(frame_number)
        self.x.append(x)
        self.y.append(y)
        self.visibility.append(visible)
        self.last_confidence = confidence

    @property
    def length(self) -> int:
        return len(self.frames)

    def as_array(self) -> np.ndarray:
        return np.stack([np.array(self.x), np.array(self.y)], axis=1)  # (L,2)

class TapNetTracker:
    """Causal TapNet-style point tracker wrapper.

    Minimal interface expected by `PerceptionEngine`:
        - initialize(detections, frame) -> seeds point tracks
        - update(frame_number, frame) -> advances all tracks one frame
        - get_active_tracks() -> returns list[PointTrack]
    """
    def __init__(
        self,
        checkpoint_path: str,
        max_points: int = 256,
        resolution: int = 256,
        device: Optional[str] = None,
        online_mode: bool = True,
        min_track_length: int = 3,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.max_points = max_points
        self.resolution = resolution
        self.device = self._select_device(device)
        self.online_mode = online_mode
        self.min_track_length = min_track_length

        self._model = None  # placeholder for loaded TapNet/TAPIR model
        self._tracks: List[PointTrack] = []
        self._next_track_id = 0
        self._initialized = False

        self._load_checkpoint()

    def _select_device(self, explicit: Optional[str]) -> str:
        if explicit:
            return explicit
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path:
            logger.warning("TapNet checkpoint path not provided; tracker disabled.")
            return
        try:
            if not torch.cuda.is_available():
                map_location = "cpu"
            else:
                map_location = None
            # NOTE: Real implementation would parse numpy / torch weight file
            # For now we just verify file existence.
            import os
            if not os.path.exists(self.checkpoint_path):
                logger.warning(
                    f"TapNet checkpoint not found at {self.checkpoint_path}. Tracking will be a no-op."\
                )
                return
            # Placeholder: self._model = load_actual_tapnet(self.checkpoint_path, device=self.device)
            self._model = object()  # sentinel
            logger.info(f"✓ TapNet checkpoint registered (stub): {self.checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load TapNet checkpoint: {e}. Tracker will be disabled.")
            self._model = None

    def initialize(self, detections: List[Dict], frame: np.ndarray) -> None:
        """Seed initial point tracks from detection centroids.
        Each detection contributes one query point (centroid). We cap by max_points.
        """
        if self._initialized:
            return
        if self._model is None:
            return
        points: List[Tuple[float, float]] = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            points.append((cx, cy))
            if len(points) >= self.max_points:
                break
        for (cx, cy) in points:
            track = PointTrack(
                track_id=self._next_track_id,
                query_xy=(cx, cy),
                frames=[],
                x=[],
                y=[],
                visibility=[],
                last_confidence=1.0,
            )
            self._tracks.append(track)
            self._next_track_id += 1
        self._initialized = True
        # Initial forward pass (stub)
        self._advance_tracks(frame_number=0, frame=frame)

    def update(self, frame_number: int, frame: np.ndarray) -> None:
        if not self._initialized or self._model is None:
            return
        self._advance_tracks(frame_number, frame)

    def _advance_tracks(self, frame_number: int, frame: np.ndarray) -> None:
        """Advance all tracks one frame. Stub version performs identity motion.
        Real TapNet would run correlation & refinement to localize points.
        """
        height, width = frame.shape[:2]
        for track in self._tracks:
            # Identity motion assumption (placeholder): keep last position
            if track.length == 0:
                x, y = track.query_xy
            else:
                x, y = track.x[-1], track.y[-1]
            visible = True  # Stub: always visible
            # Clamp to frame bounds
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            track.append(frame_number, x, y, visible, confidence=1.0)

    def get_active_tracks(self) -> List[PointTrack]:
        return self._tracks

    def get_promoted_entities(self) -> List[PointTrack]:
        """Return tracks long enough to be considered stable entities."""
        return [t for t in self._tracks if t.length >= self.min_track_length]

__all__ = ["TapNetTracker", "PointTrack"]
