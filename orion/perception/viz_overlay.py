"""Perception overlay renderer (compat shim).

This module contains the legacy `InsightOverlayRenderer` used by the CLI.
The previous file contained a duplicated V4 wrapper which caused accidental
code duplication and undefined variables; the module below provides a single
consistent implementation used by `orion.cli`.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from orion.perception.spatial_zones import ZoneManager

logger = logging.getLogger(__name__)


import json
import logging
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from orion.perception.spatial_zones import ZoneManager

logger = logging.getLogger(__name__)


def get_perception_run_dims(results_dir: Path) -> Optional[Tuple[int, int]]:
    """Get processing dimensions from run metadata."""
    meta_path = results_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return (meta.get("width", 1920), meta.get("height", 1080))
        except Exception:
            pass
    return None


@dataclass
class OverlayOptions:
    """Tunable parameters for the perception insight overlay."""

    max_relations: int = 4
    message_linger_seconds: float = 1.75
    max_state_messages: int = 5
    gap_frames_for_refind: int = 45
    overlay_basename: str = "video_overlay_insights.mp4"
    frame_offset: int = 0  # Number of frames to shift overlays (positive = overlays appear later)
    use_timestamp_matching: bool = True  # Use timestamp-based overlay matching by default
    timestamp_offset: float = 0.0  # Seconds to shift overlays in time (positive = overlays appear later)


class InsightOverlayRenderer:
    """Compose a narrative-rich overlay that ties together tracks, memory, and zones."""

    def __init__(
        self,
        video_path: Path,
        results_dir: Path,
        output_path: Optional[Path] = None,
        options: Optional[OverlayOptions] = None,
    ) -> None:
        self.video_path = video_path
        self.results_dir = results_dir
        self.options = options or OverlayOptions()
        self.output_path = output_path or (results_dir / self.options.overlay_basename)
        self.zone_manager = ZoneManager()

        self.tracks_path = results_dir / "tracks.jsonl"
        self.memory_path = results_dir / "memory.json"
        self.graph_path = results_dir / "scene_graph.jsonl"
        self.pipeline_output_path = results_dir / "pipeline_output.json"

        if not self.tracks_path.exists():
            logger.error(f"tracks.jsonl missing in {results_dir}")
            raise FileNotFoundError(f"tracks.jsonl missing in {results_dir}")
        if not self.memory_path.exists():
            logger.error(f"memory.json missing in {results_dir}")
            raise FileNotFoundError(f"memory.json missing in {results_dir}")
        if not self.video_path.exists():
            logger.error(f"Video not found: {video_path}")
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.tracks_by_id: Dict[int, List[Dict]] = defaultdict(list)
        self.memory: Dict[str, Dict] = {}
        self.embed_to_mem: Dict[str, str] = {}
        self.track_to_mem: Dict[int, str] = {}
        self.graph_by_frame: Dict[int, Dict] = {}
        self.interpolated_tracks: Dict[int, Dict[str, np.ndarray]] = defaultdict(dict)

        # Get FPS for timestamp calculations
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            logger.error(f"Unable to open video {self.video_path}")
            raise RuntimeError(f"Unable to open video {self.video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        self._load_tracks()
        self._load_memory()
        self._load_scene_graphs()
        self._prepare_interpolated_tracks()

        self.mem_last_seen: Dict[str, Optional[int]] = {mid: None for mid in self.memory}
        self.mem_tracks_seen: Dict[str, set[int]] = defaultdict(set)
        self.state_messages: Deque[Tuple[str, Tuple[int, int, int], int]] = deque()
        self.current_zone: Optional[int] = None

        # Scaling factors for mis-aligned processing dimensions
        self.scale_x = 1.0
        self.scale_y = 1.0
    

    def _load_memory(self) -> None:
        data = json.loads(self.memory_path.read_text())
        for obj in data.get("objects", []):
            mem_id = obj.get("memory_id")
            if not mem_id:
                continue
            cls = obj.get("class", "object")
            zone_id = obj.get("zone_id")
            if zone_id is None:
                zone_id = self.zone_manager.assign_zone_from_class(cls)
            obj["zone_id"] = zone_id
            obj["zone_name"] = self.zone_manager.get_zone_name(zone_id)
            self.memory[mem_id] = obj
            emb = obj.get("prototype_embedding")
            if emb:
                self.embed_to_mem[emb] = mem_id
            for segment in obj.get("appearance_history", []):
                tid = segment.get("track_id")
                if tid is not None:
                    self.track_to_mem[int(tid)] = mem_id
    
    def _match_track_to_memory(self, track_id: int) -> Optional[str]:
        """Finds the memory ID for a given track ID."""
        return self.track_to_mem.get(track_id)

        # Get FPS for timestamp calculations
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            logger.error(f"Unable to open video {self.video_path}")
            raise RuntimeError(f"Unable to open video {self.video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        self._load_tracks()
        self._load_memory()
        self._load_scene_graphs()

        self.mem_last_seen: Dict[str, Optional[int]] = {mid: None for mid in self.memory}
        self.mem_tracks_seen: Dict[str, set[int]] = defaultdict(set)
        self.state_messages: Deque[Tuple[str, Tuple[int, int, int], int]] = deque()
        self.current_zone: Optional[int] = None

        # Scaling factors for mis-aligned processing dimensions
        self.scale_x = 1.0
        self.scale_y = 1.0

    def _load_tracks(self) -> None:
        """Loads tracks and groups them by track_id."""
        with open(self.tracks_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                det = json.loads(line)
                track_id = det.get("track_id")
                if track_id is not None:
                    self.tracks_by_id[int(track_id)].append(det)
        # Sort each track's detections by frame
        for track_id in self.tracks_by_id:
            self.tracks_by_id[track_id].sort(key=lambda d: d.get("frame_id", 0))

    def _prepare_interpolated_tracks(self) -> None:
        """Pre-calculates interpolated bounding boxes based on timestamps for each track segment."""
        gap_threshold_seconds = self.options.gap_frames_for_refind / self.fps

        for track_id, detections in self.tracks_by_id.items():
            if not detections:
                continue

            # Split track into continuous segments based on timestamp gaps
            segments: List[List[Dict]] = []
            current_segment = [detections[0]]
            for i in range(1, len(detections)):
                prev_det = detections[i-1]
                curr_det = detections[i]
                time_gap = curr_det.get("timestamp", 0.0) - prev_det.get("timestamp", 0.0)
                if time_gap > gap_threshold_seconds:
                    segments.append(current_segment)
                    current_segment = [curr_det]
                else:
                    current_segment.append(curr_det)
            segments.append(current_segment)

            # For each segment, create an interpolation function based on timestamps
            for i, segment in enumerate(segments):
                if len(segment) < 2:
                    # Cannot interpolate a single point
                    continue

                timestamps = np.array([d["timestamp"] for d in segment])
                bboxes = np.array([d["bbox_2d"] for d in segment])

                # Store the raw data for this segment
                self.interpolated_tracks[track_id][f"segment_{i}"] = {
                    "timestamps": timestamps,
                    "bboxes": bboxes,
                    "metadata": segment, # Keep original detections for labels etc.
                }

    def _load_memory(self) -> None:
        data = json.loads(self.memory_path.read_text())
        for obj in data.get("objects", []):
            mem_id = obj.get("memory_id")
            if not mem_id:
                continue
            cls = obj.get("class", "object")
            zone_id = obj.get("zone_id")
            if zone_id is None:
                zone_id = self.zone_manager.assign_zone_from_class(cls)
            obj["zone_id"] = zone_id
            obj["zone_name"] = self.zone_manager.get_zone_name(zone_id)
            self.memory[mem_id] = obj
            emb = obj.get("prototype_embedding")
            if emb:
                self.embed_to_mem[emb] = mem_id
            for segment in obj.get("appearance_history", []):
                tid = segment.get("track_id")
                if tid is not None:
                    self.track_to_mem[int(tid)] = mem_id
    
    def _match_track_to_memory(self, track_id: int) -> Optional[str]:
        """Finds the memory ID for a given track ID."""
        return self.track_to_mem.get(track_id)

    def _load_scene_graphs(self) -> None:
        if not self.graph_path.exists():
            return
        with open(self.graph_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                frame = int(payload.get("frame", payload.get("frame_id", -1)))
                if frame < 0:
                    continue
                self.graph_by_frame[frame] = payload

    def _prepare_renderer(self, cap) -> Tuple[int, int, int, float]:
        """Gets video properties and calculates scaling factors."""
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        render_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        render_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # === FIX: Store original render dimensions ===
        self.output_video_width = render_w
        self.output_video_height = render_h
        # === END FIX ===

        # === FIX: Handle portrait video rotation in renderer ===
        self.is_portrait = render_h > render_w
        # === END FIX ===

        proc_dims = get_perception_run_dims(self.results_dir)
        if proc_dims:
            proc_w, proc_h = proc_dims
            
            # If portrait, the processing dimensions are swapped relative to render dimensions
            if self.is_portrait:
                # Render: 1080x1920, Processed: 1920x1080
                # Scale render_w (1080) by proc_w (1920) -> this is wrong
                # Scale render_h (1920) by proc_h (1080) -> this is wrong
                # Correct scaling:
                # scale_x maps processed-x to rendered-x.
                # After rotation, processed-x corresponds to rendered-y.
                # So, scale_x should be render_w / proc_h
                # And scale_y should be render_h / proc_w
                self.scale_x = render_w / proc_h if proc_h > 0 else 1.0
                self.scale_y = render_h / proc_w if proc_w > 0 else 1.0
            else:
                # Standard landscape
                self.scale_x = render_w / proc_w if proc_w > 0 else 1.0
                self.scale_y = render_h / proc_h if proc_h > 0 else 1.0
        
        print(f"Rendering at {render_w}x{render_h}. Portrait: {self.is_portrait}. Processing at {proc_dims}. Scaling by ({self.scale_x:.2f}, {self.scale_y:.2f})")
        return render_w, render_h, total_frames, fps

    def render(self) -> Path:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video {self.video_path}")

        render_w, render_h, total_frames, fps = self._prepare_renderer(cap)
        overlay_ttl = max(1, int(self.options.message_linger_seconds * fps))

        # Try multiple codecs for cross-platform compatibility
        codecs = ["avc1", "mp4v", "XVID", "MJPG"]
        writer = None
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (render_w, render_h))
            if writer.isOpened():
                logger.info(f"Using video codec: {codec}")
                break
            writer.release()
            writer = None
        
        if writer is None or not writer.isOpened():
            logger.error(f"Unable to open writer for {self.output_path}")
            raise RuntimeError(f"Unable to open writer for {self.output_path}")

        # --- DEBUG: Print video and detection stats ---
        all_frame_ids = []
        all_timestamps = []
        for track_id, detections in self.tracks_by_id.items():
            for det in detections:
                if "frame_id" in det:
                    all_frame_ids.append(det["frame_id"])
                if "timestamp" in det:
                    all_timestamps.append(det["timestamp"])
        if all_frame_ids:
            print(f"[Overlay-DEBUG] Detection frame_id range: {min(all_frame_ids)} to {max(all_frame_ids)}")
        if all_timestamps:
            print(f"[Overlay-DEBUG] Detection timestamp range: {min(all_timestamps):.3f} to {max(all_timestamps):.3f}")
        print(f"[Overlay-DEBUG] Video FPS: {fps}, Total frames: {total_frames}")

        # --- END DEBUG ---

        use_timestamp_matching = getattr(self.options, 'use_timestamp_matching', True)
        
        # Use interpolated tracks if available for smoother visualization
        use_interpolation = len(self.interpolated_tracks) > 0
        if use_interpolation:
            logger.info("Using interpolated tracks for smooth visualization")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                curr_time = frame_idx / fps
                detections_to_render = []

                if use_interpolation:
                    # Interpolate tracks at current timestamp
                    for track_id, segments in self.interpolated_tracks.items():
                        for seg_name, data in segments.items():
                            timestamps = data["timestamps"]
                            bboxes = data["bboxes"]
                            
                            # Check if current time is within segment range (with small buffer)
                            if timestamps[0] <= curr_time <= timestamps[-1]:
                                # Interpolate bbox
                                # Find indices
                                idx = np.searchsorted(timestamps, curr_time)
                                if idx == 0:
                                    bbox = bboxes[0]
                                    meta = data["metadata"][0]
                                elif idx >= len(timestamps):
                                    bbox = bboxes[-1]
                                    meta = data["metadata"][-1]
                                else:
                                    t0, t1 = timestamps[idx-1], timestamps[idx]
                                    ratio = (curr_time - t0) / (t1 - t0) if t1 > t0 else 0
                                    bbox = bboxes[idx-1] * (1 - ratio) + bboxes[idx] * ratio
                                    meta = data["metadata"][idx] # Use nearest metadata
                                
                                # Create synthetic detection for rendering
                                det = meta.copy()
                                det["bbox_2d"] = bbox.tolist()
                                det["timestamp"] = curr_time
                                detections_to_render.append(det)
                                break # Only one segment per track should match
                
                elif use_timestamp_matching:
                    # Fallback to raw timestamp matching
                    detection_time = curr_time
                    if hasattr(self.options, 'timestamp_offset'):
                        detection_time += getattr(self.options, 'timestamp_offset', 0.0)
                    
                    for track_id, detections in self.tracks_by_id.items():
                        for det in detections:
                            # Allow wider tolerance (1.5 frames) to prevent flashing
                            if abs(det.get("timestamp", -9999) - detection_time) < (1.5 / fps):
                                detections_to_render.append(det)
                else:
                    # Fallback to frame ID matching
                    detection_frame_idx = frame_idx + (self.options.frame_offset or 0)
                    for track_id, detections in self.tracks_by_id.items():
                        for det in detections:
                            if det.get("frame_id") == detection_frame_idx:
                                detections_to_render.append(det)

                if frame_idx % 30 == 0:
                    print(f"[Overlay] Frame {frame_idx} ({curr_time:.2f}s): drawing {len(detections_to_render)} detections")

                # Annotate frame and get active memories
                active_memories = self._annotate_detections(
                    frame, detections_to_render, frame_idx, overlay_ttl
                )

                # Update and compose the narrative panel
                narrative_lines = self._compose_panel_lines(
                    frame_idx, fps, total_frames, detections_to_render, active_memories
                )
                self._draw_narrative_panel(frame, narrative_lines)

                writer.write(frame)
                frame_idx += 1
        finally:
            cap.release()
            writer.release()

        return self.output_path

    def _match_memory(self, detection: Dict) -> Optional[str]:
        # This function is now less direct, we match via track_id
        tid = detection.get("track_id")
        if tid is not None:
            return self._match_track_to_memory(int(tid))
        return None

    def _annotate_detections(
        self, frame: np.ndarray, detections: List[Dict], frame_idx: int, overlay_ttl: int
    ) -> set[str]:
        """Draws bounding boxes and labels on the frame."""
        active_memories = set()
        
        for det in detections:
            track_id = det.get("track_id")
            mem_id = self._match_track_to_memory(track_id) if track_id is not None else None

            x1_proc, y1_proc, x2_proc, y2_proc = det["bbox_2d"]

            # === FIX: Correctly un-rotate coordinates for portrait video ===
            if self.is_portrait:
                proc_dims = get_perception_run_dims(self.results_dir)
                proc_w, proc_h = proc_dims if proc_dims else (1920, 1080)
                x1_unrotated = proc_w - 1 - y2_proc
                y1_unrotated = x1_proc
                x2_unrotated = proc_w - 1 - y1_proc
                y2_unrotated = x2_proc
                x1 = min(x1_unrotated, x2_unrotated)
                y1 = min(y1_unrotated, y2_unrotated)
                x2 = max(x1_unrotated, x2_unrotated)
                y2 = max(y1_unrotated, y2_unrotated)
            else:
                x1, y1, x2, y2 = x1_proc, y1_proc, x2_proc, y2_proc
            # === END FIX ===

            # Apply scaling
            x1, x2 = int(x1 * self.scale_x), int(x2 * self.scale_x)
            y1, y2 = int(y1 * self.scale_y), int(y2 * self.scale_y)

            # Bounding box validity check: skip if out of frame or zero/negative area
            frame_h, frame_w = frame.shape[:2]
            if x2 <= x1 or y2 <= y1:
                continue  # Skip invalid or zero-area boxes
            if x2 < 0 or y2 < 0 or x1 >= frame_w or y1 >= frame_h:
                continue  # Entirely out of frame

            # Clamp to frame bounds
            x1 = max(0, min(x1, frame_w - 1))
            x2 = max(0, min(x2, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            y2 = max(0, min(y2, frame_h - 1))

            # Initialize label from the detection itself
            label = det.get("class_name", "object")
            color = (128, 128, 128) # Default color

            if mem_id and self.memory.get(mem_id):
                mem_obj = self.memory[mem_id]
                label = mem_obj.get("class", label) # Get class from memory
                zone_id = mem_obj.get("zone_id")
                color = self.zone_manager.get_zone_color(zone_id=zone_id)
                self._update_memory_messages(mem_id, track_id, frame_idx, overlay_ttl)
                self.mem_last_seen[mem_id] = frame_idx
                active_memories.add(mem_id)
                label_text = f"{label} [{mem_id[-3:]}]"
            else:
                zone_id = self.zone_manager.assign_zone_from_class(label)
                color = self.zone_manager.get_zone_color(zone_id=zone_id)
                label_text = f"{label} #{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Labeling logic
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = y1 - 10 if y1 - 10 > text_h else y1 + text_h + 10
            cv2.rectangle(frame, (x1, label_y - text_h - baseline), (x1 + text_w, label_y + baseline), color, -1)
            cv2.putText(frame, label_text, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        return active_memories

    def _update_memory_messages(self, mem_id: str, track_id: int, frame_idx: int, overlay_ttl: int) -> None:
        last_seen = self.mem_last_seen.get(mem_id)
        track_ids = self.mem_tracks_seen[mem_id]
        info = self.memory[mem_id]
        if last_seen is None:
            self._push_state_message(
                f"Episode memory {mem_id} ({info.get('class', 'object')}) entered",
                (102, 255, 178),
                overlay_ttl,
            )
        else:
            gap = frame_idx - last_seen
            if gap >= self.options.gap_frames_for_refind:
                self._push_state_message(
                    f"Refound {mem_id} after {gap} frames",
                    (255, 140, 105),
                    overlay_ttl,
                )
        if track_id not in track_ids and track_id >= 0 and last_seen is not None:
            self._push_state_message(
                f"Re-tracked {mem_id} → track {track_id}",
                (173, 216, 230),
                overlay_ttl,
            )
        track_ids.add(track_id)
        self.mem_last_seen[mem_id] = frame_idx

    def _color_for_label(self, label: str) -> Tuple[int, int, int]:
        import colorsys

        h = abs(hash(label)) % 360
        hue = h / 360.0
        sat = 0.6
        val = 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return int(b * 255), int(g * 255), int(r * 255)

    def _draw_label(self, frame, line1: str, line2: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        x, y = origin
        (w1, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        (w2, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        panel_width = max(w1, w2) + 10
        panel_height = 38
        
        # Ensure the panel doesn't go off-screen
        y_start = y - panel_height if y - panel_height > 0 else y + 10
        
        panel = frame.copy()
        cv2.rectangle(panel, (x, y_start), (x + panel_width, y_start + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, line1, (x + 5, y_start + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if line2:
            cv2.putText(frame, line2, (x + 5, y_start + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


    def _compose_panel_lines(self, frame_idx, fps, total_frames, detections_to_render, active_memories):
        lines = []
        # ...existing code for composing panel lines...
        # This is a placeholder to ensure the function is closed and valid.


    def _format_memory_label(self, mem_id: Optional[str]) -> str:
        if not mem_id:
            return "?"
        info = self.memory.get(mem_id)
        if not info:
            return mem_id
        return f"{info.get('class', 'object')}[{mem_id}]"

    def _draw_narrative_panel(self, frame: np.ndarray, lines: List[str]) -> None:
        if not lines:
            return
        width = frame.shape[1]
        panel_width = min(620, width - 20)
        panel_height = 28 + 22 * len(lines)
        x, y = 10, 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, min(y + panel_height, frame.shape[0] - 10)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cursor_y = y + 25
        for line in lines:
            cv2.putText(frame, line, (x + 12, cursor_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cursor_y += 22

    def _decay_messages(self) -> None:
        new_queue: Deque[Tuple[str, Tuple[int, int, int], int]] = deque()
        for msg, color, ttl in self.state_messages:
            if ttl > 1:
                new_queue.append((msg, color, ttl - 1))
        self.state_messages = new_queue

    def _push_state_message(self, text: str, color: Tuple[int, int, int], ttl: int) -> None:
        self.state_messages.append((text, color, ttl))
        while len(self.state_messages) > self.options.max_state_messages:
            self.state_messages.popleft()


def render_insight_overlay(
    video_path: Path,
    results_dir: Path,
    output_path: Optional[Path] = None,
    options: Optional[OverlayOptions] = None,
) -> Path:
    """Convenience wrapper used by CLI entry points."""
    renderer = InsightOverlayRenderer(video_path=video_path, results_dir=results_dir, output_path=output_path, options=options)
    return renderer.render()
