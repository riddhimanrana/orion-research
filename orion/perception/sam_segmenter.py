"""Segment Anything integration utilities for perception pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SegmentAnythingMaskGenerator:
    """Runs SAM to refine detector bounding boxes and produce instance masks."""

    def __init__(
        self,
        predictor: Any,
        *,
        mask_threshold: float = 0.5,
        stability_score_threshold: float = 0.85,
        min_mask_area: int = 400,
        batch_size: int = 16,
        refine_bounding_box: bool = True,
    ) -> None:
        self.predictor = predictor
        self.mask_threshold = float(mask_threshold)
        self.stability_score_threshold = float(stability_score_threshold)
        self.min_mask_area = int(min_mask_area)
        self.batch_size = int(max(1, batch_size))
        self.refine_bounding_box = bool(refine_bounding_box)
        # Derive device from underlying SAM model
        try:
            self.device = next(self.predictor.model.parameters()).device
        except Exception:  # pragma: no cover - defensive fallback
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def refine_detections(self, frame_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Annotate detections with SAM masks and optional bbox refinements."""
        if frame_bgr is None or not detections:
            return detections

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]

        boxes, mapping = self._extract_boxes(detections)
        if not boxes:
            return detections

        try:
            self.predictor.set_image(frame_rgb)
        except Exception as exc:  # pragma: no cover - SAM internal errors
            logger.warning("SAM failed to ingest frame: %s", exc)
            return detections

        for start in range(0, len(boxes), self.batch_size):
            end = min(start + self.batch_size, len(boxes))
            batch_mapping = mapping[start:end]
            batch_boxes = torch.as_tensor(boxes[start:end], device=self.device, dtype=torch.float32)
            transformed = self.predictor.transform.apply_boxes_torch(batch_boxes, (height, width))
            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed,
                    multimask_output=False,
                )

            masks = masks.squeeze(1).detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()

            for local_idx, det_idx in enumerate(batch_mapping):
                quality = float(scores[local_idx].item()) if np.ndim(scores[local_idx]) else float(scores[local_idx])
                if quality < self.stability_score_threshold:
                    continue
                binary_mask = masks[local_idx] > self.mask_threshold
                mask_area = int(binary_mask.sum())
                if mask_area < self.min_mask_area:
                    continue

                y_indices, x_indices = np.where(binary_mask)
                if y_indices.size == 0 or x_indices.size == 0:
                    continue
                x_min = int(np.clip(x_indices.min(), 0, width - 1))
                x_max = int(np.clip(x_indices.max() + 1, 0, width))
                y_min = int(np.clip(y_indices.min(), 0, height - 1))
                y_max = int(np.clip(y_indices.max() + 1, 0, height))
                mask_crop = binary_mask[y_min:y_max, x_min:x_max].astype(np.uint8)

                detection = detections[det_idx]
                detection["sam_mask"] = mask_crop
                detection["sam_bbox"] = [float(x_min), float(y_min), float(x_max), float(y_max)]
                detection["sam_score"] = quality
                detection["sam_area"] = mask_area
                if self.refine_bounding_box:
                    detection["bbox"] = detection["sam_bbox"]

        return detections

    @staticmethod
    def _extract_boxes(detections: Sequence[Dict[str, Any]]) -> Tuple[List[List[float]], List[int]]:
        boxes: List[List[float]] = []
        mapping: List[int] = []
        for idx, det in enumerate(detections):
            bbox = det.get("bbox")
            if bbox is None:
                continue
            if len(bbox) != 4:
                continue
            boxes.append([float(b) for b in bbox])
            mapping.append(idx)
        return boxes, mapping


class Sam3MaskGenerator:
    """Runs SAM3 (Segment Anything 3) to refine detections with mask output."""

    def __init__(
        self,
        processor: Any,
        *,
        min_mask_area: int = 400,
        refine_bounding_box: bool = True,
    ) -> None:
        self.processor = processor
        self.min_mask_area = int(min_mask_area)
        self.refine_bounding_box = bool(refine_bounding_box)
        self.device = getattr(processor, "device", "cuda" if torch.cuda.is_available() else "cpu")

    def refine_detections(self, frame_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if frame_bgr is None or not detections:
            return detections

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            pil_image = Image.fromarray(frame_rgb)
        except Exception:  # pragma: no cover - unexpected conversion failure
            logger.warning("SAM3 could not convert frame to PIL format; skipping refinement")
            return detections

        try:
            state = self.processor.set_image(pil_image, state=None)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("SAM3 failed to ingest frame: %s", exc)
            return detections

        height, width = frame_rgb.shape[:2]

        for detection in detections:
            bbox = detection.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue

            prompt_box = self._to_normalized_cxcywh(bbox, width, height)
            if prompt_box is None:
                continue

            try:
                state = self.processor.add_geometric_prompt(prompt_box, True, state)
            except Exception as exc:  # pragma: no cover - robustness
                logger.debug("SAM3 prompt failed: %s", exc)
                self.processor.reset_all_prompts(state)
                continue

            masks = state.get("masks")
            boxes = state.get("boxes")
            scores = state.get("scores")
            if masks is None or boxes is None or scores is None or masks.numel() == 0:
                self.processor.reset_all_prompts(state)
                continue

            best_idx = int(torch.argmax(scores).item())
            mask_tensor = masks[best_idx].squeeze(0)
            mask_binary = mask_tensor.detach().cpu().numpy().astype(np.uint8)
            mask_area = int(mask_binary.sum())
            if mask_area < self.min_mask_area:
                self.processor.reset_all_prompts(state)
                continue

            x1, y1, x2, y2 = boxes[best_idx].detach().cpu().numpy().tolist()
            x1_i = int(np.clip(round(x1), 0, max(1, width - 1)))
            y1_i = int(np.clip(round(y1), 0, max(1, height - 1)))
            x2_i = int(np.clip(round(x2), x1_i + 1, width))
            y2_i = int(np.clip(round(y2), y1_i + 1, height))
            mask_patch = mask_binary[y1_i:y2_i, x1_i:x2_i]

            detection["sam_mask"] = mask_patch
            detection["sam_bbox"] = [float(x1_i), float(y1_i), float(x2_i), float(y2_i)]
            detection["sam_score"] = float(scores[best_idx].detach().cpu().item())
            detection["sam_area"] = mask_area
            if self.refine_bounding_box:
                detection["bbox"] = detection["sam_bbox"]

            self.processor.reset_all_prompts(state)

        return detections

    @staticmethod
    def _to_normalized_cxcywh(bbox: Sequence[float], width: int, height: int) -> Optional[List[float]]:
        if width <= 0 or height <= 0:
            return None
        x1, y1, x2, y2 = map(float, bbox)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0 / float(width)
        cy = (y1 + y2) / 2.0 / float(height)
        norm_w = w / float(width)
        norm_h = h / float(height)
        cx = float(np.clip(cx, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))
        norm_w = float(np.clip(norm_w, 1e-3, 1.0))
        norm_h = float(np.clip(norm_h, 1e-3, 1.0))
        return [cx, cy, norm_w, norm_h]