from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from orion.perception.spatial_zones import ZoneManager


@dataclass
class OverlayOptions:
    """Tunable parameters for the perception insight overlay."""

    max_relations: int = 4
    message_linger_seconds: float = 1.75
    max_state_messages: int = 5
    gap_frames_for_refind: int = 45
    overlay_basename: str = "video_overlay_insights.mp4"


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

        if not self.tracks_path.exists():
            raise FileNotFoundError(f"tracks.jsonl missing in {results_dir}")
        if not self.memory_path.exists():
            raise FileNotFoundError(f"memory.json missing in {results_dir}")
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.tracks: List[Dict] = []
        self.tracks_by_frame: Dict[int, List[Dict]] = defaultdict(list)
        self.memory: Dict[str, Dict] = {}
        self.embed_to_mem: Dict[str, str] = {}
        self.track_to_mem: Dict[int, str] = {}
        self.graph_by_frame: Dict[int, Dict] = {}

        self._load_tracks()
        self._load_memory()
        self._load_scene_graphs()

        self.mem_last_seen: Dict[str, Optional[int]] = {mid: None for mid in self.memory}
        self.mem_tracks_seen: Dict[str, set[int]] = defaultdict(set)
        self.state_messages: Deque[Tuple[str, Tuple[int, int, int], int]] = deque()
        self.current_zone: Optional[int] = None

    def _load_tracks(self) -> None:
        with open(self.tracks_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                det = json.loads(line)
                self.tracks.append(det)
                frame_id = int(det.get("frame_id", 0))
                self.tracks_by_frame[frame_id].append(det)

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

    def render(self) -> Path:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        overlay_ttl = max(1, int(self.options.message_linger_seconds * fps))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open writer for {self.output_path}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = self.tracks_by_frame.get(frame_idx, [])
                active_memories = self._annotate_detections(frame, detections, frame_idx, overlay_ttl)
                narrative_lines = self._compose_panel_lines(frame_idx, fps, total_frames, detections, active_memories)
                self._draw_panel(frame, narrative_lines)
                writer.write(frame)
                frame_idx += 1
        finally:
            cap.release()
            writer.release()

        return self.output_path

    def _match_memory(self, detection: Dict) -> Optional[str]:
        emb = detection.get("embedding_id")
        if emb and emb in self.embed_to_mem:
            return self.embed_to_mem[emb]
        tid = detection.get("track_id")
        if tid is not None and int(tid) in self.track_to_mem:
            return self.track_to_mem[int(tid)]
        return None

    def _annotate_detections(
        self,
        frame,
        detections: List[Dict],
        frame_idx: int,
        overlay_ttl: int,
    ) -> List[str]:
        active_memories: List[str] = []
        zone_counts: Counter[int] = Counter()
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = map(int, bbox)
            category = det.get("category", "object")
            track_id = int(det.get("track_id", -1))
            mem_id = self._match_memory(det)
            color: Tuple[int, int, int]
            if mem_id and mem_id in self.memory:
                mem_info = self.memory[mem_id]
                zone_id = mem_info.get("zone_id", 0)
                zone_counts[zone_id] += 1
                active_memories.append(mem_id)
                label = f"{mem_info.get('class', category)} [{mem_id}]"
                sub_label = f"track {track_id} · {mem_info.get('zone_name', 'Unknown')}"
                color = self._color_for_label(mem_info.get("class", category))
                self._update_memory_messages(mem_id, track_id, frame_idx, overlay_ttl)
            else:
                label = f"{category} #{track_id}"
                sub_label = "detected"
                color = (255, 255, 255)
            segmentation = det.get("segmentation")
            if segmentation:
                self._draw_segmentation_mask(frame, (x1, y1, x2, y2), segmentation, color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            self._draw_label(frame, label, sub_label, (x1, max(y1 - 30, 20)), color)

        dominant_zone = None
        if zone_counts:
            dominant_zone = zone_counts.most_common(1)[0][0]
            if dominant_zone != self.current_zone:
                self.current_zone = dominant_zone
                zone_name = self.zone_manager.get_zone_name(dominant_zone)
                self._push_state_message(f"Moved focus to {zone_name}", (255, 215, 0), overlay_ttl)

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
        panel_width = max(w1, w2) + 40
        panel = frame.copy()
        cv2.rectangle(panel, (x - 8, y - 24), (x - 8 + panel_width, y + 34), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.35, frame, 0.65, 0, frame)
        cv2.putText(frame, line1, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(frame, line2, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def _compose_panel_lines(
        self,
        frame_idx: int,
        fps: float,
        total_frames: int,
        detections: List[Dict],
        active_memories: List[str],
    ) -> List[str]:
        self._decay_messages()
        lines = [
            f"Frame {frame_idx}/{total_frames}  Time {frame_idx / fps:6.2f}s",
            f"Active detections: {len(detections)}",
        ]
        if self.current_zone is not None:
            lines.append(f"Focus zone: {self.zone_manager.get_zone_name(self.current_zone)}")

        if self.state_messages:
            lines.append("Moments:")
            for msg, color, _ in list(self.state_messages)[: self.options.max_state_messages]:
                lines.append(f"  {msg}")

        if active_memories:
            ordered = list(dict.fromkeys(active_memories))
            lines.append("Episode memory highlight:")
            for mem_id in ordered[:3]:
                info = self.memory.get(mem_id, {})
                lines.append(
                    "  "
                    + f"{mem_id}: {info.get('class', 'object')} · obs={info.get('total_observations', '?')} · {info.get('zone_name', 'Unknown')}"
                )

        graph = self.graph_by_frame.get(frame_idx)
        if graph and graph.get("edges"):
            lines.append("Relations:")
            for edge in graph["edges"][: self.options.max_relations]:
                subj = edge.get("subject")
                obj = edge.get("object")
                rel = edge.get("relation")
                subj_label = self._format_memory_label(subj)
                obj_label = self._format_memory_label(obj)
                lines.append(f"  {subj_label} {rel} {obj_label}")
        return lines

    def _format_memory_label(self, mem_id: Optional[str]) -> str:
        if not mem_id:
            return "?"
        info = self.memory.get(mem_id)
        if not info:
            return mem_id
        return f"{info.get('class', 'object')}[{mem_id}]"

    def _draw_panel(self, frame, lines: List[str]) -> None:
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

    def _draw_segmentation_mask(
        self,
        frame,
        bbox: Tuple[int, int, int, int],
        segmentation: Dict,
        color: Tuple[int, int, int],
        alpha: float = 0.45,
    ) -> None:
        mask = self._decode_mask(segmentation)
        if mask is None:
            return
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return
        mask_h, mask_w = mask.shape[:2]
        target_h = y2 - y1
        target_w = x2 - x1
        if mask_h != target_h or mask_w != target_w:
            mask = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask = mask.astype(bool)
        if not mask.any():
            return
        color_layer = np.zeros_like(roi, dtype=np.uint8)
        color_layer[mask] = color
        cv2.addWeighted(color_layer, alpha, roi, 1 - alpha, 0, roi)

    def _decode_mask(self, segmentation: Dict) -> Optional[np.ndarray]:
        if not segmentation:
            return None
        if segmentation.get("encoding") == "rle" and segmentation.get("data"):
            return self._decode_rle(segmentation.get("data"))
        if "rle" in segmentation:
            return self._decode_rle(segmentation.get("rle"))
        return None

    @staticmethod
    def _decode_rle(rle: Optional[Dict]) -> Optional[np.ndarray]:
        if not rle:
            return None
        size = rle.get("size")
        counts = rle.get("counts")
        if not size or not counts:
            return None
        height, width = map(int, size)
        total = max(0, int(height) * int(width))
        if total == 0:
            return None
        start_value = int(rle.get("start", 0))
        flat = np.zeros(total, dtype=np.uint8)
        idx = 0
        value = start_value
        for raw_count in counts:
            count = int(raw_count)
            if count <= 0:
                value = 1 - value
                continue
            end = min(total, idx + count)
            flat[idx:end] = value
            idx = end
            value = 1 - value
            if idx >= total:
                break
        if idx < total:
            flat[idx:] = 0
        return flat.reshape((height, width))


def render_insight_overlay(
    video_path: Path,
    results_dir: Path,
    output_path: Optional[Path] = None,
    options: Optional[OverlayOptions] = None,
) -> Path:
    """Convenience wrapper used by CLI entry points."""
    renderer = InsightOverlayRenderer(video_path=video_path, results_dir=results_dir, output_path=output_path, options=options)
    return renderer.render()
