"""Class label correction utilities using CLIP similarities."""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

import numpy as np

from orion.perception.config import ClassCorrectionConfig
from orion.perception.types import PerceptionEntity

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the",
    "and",
    "with",
    "that",
    "this",
    "from",
    "into",
    "onto",
    "over",
    "under",
    "about",
    "around",
    "into",
    "onto",
    "through",
    "across",
    "between",
    "without",
    "after",
    "before",
    "color",
    "colors",
    "texture",
    "light",
    "dark",
    "object",
    "image",
    "photo",
    "picture",
    "view",
    "scene",
    "close",
    "close-up",
}

_FALLBACK_LABELS = ["object", "pattern", "device", "fabric", "surface"]


class ClassCorrector:
    """Applies CLIP-based relabeling to perception entities."""

    def __init__(
        self,
        clip_model,
        config: ClassCorrectionConfig,
        detector_vocabulary: Optional[Sequence[str]] = None,
    ):
        self.clip_model = clip_model
        self.config = config
        self.detector_vocabulary = [label for label in (detector_vocabulary or []) if label]
        self._text_cache: Dict[str, np.ndarray] = {}
        self._label_lookup: Dict[str, str] = {}
        self._allowed_labels: Set[str] = set()

        # Register detector and configured labels as the only valid correction targets
        for label in self.detector_vocabulary:
            self._register_label(label)
        for label in getattr(self.config, "extra_labels", []) or []:
            self._register_label(label)

    def apply(self, entities: Iterable[PerceptionEntity]) -> int:
        """Run class correction across entities.

        Returns number of entities whose label was updated.
        """
        corrections = 0
        for entity in entities:
            visual_embedding = entity.average_embedding
            if visual_embedding is None:
                try:
                    visual_embedding = entity.compute_average_embedding()
                except Exception:
                    visual_embedding = None
            if visual_embedding is None:
                continue

            candidates = self._build_candidates(entity)
            if not candidates:
                continue

            label, score = self._select_label(visual_embedding, candidates)
            if label is None or score < self.config.min_similarity:
                continue

            if entity.corrected_class != label:
                entity.corrected_class = label
                entity.correction_confidence = score
                corrections += 1
        if corrections:
            logger.info("Class correction updated %d entities", corrections)
        return corrections

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _build_candidates(self, entity: PerceptionEntity) -> List[str]:
        description = entity.description or ""
        tokens = self._extract_keywords(description)
        candidates: List[str] = []

        for token in tokens:
            normalized = self._normalize_label(token)
            if not normalized or normalized not in self._allowed_labels:
                continue
            candidates.append(self._label_lookup.get(normalized, token))
            if len(candidates) >= self.config.max_description_tokens:
                break

        if self.config.include_detector_label:
            raw_label = (
                entity.corrected_class
                or (entity.object_class.value if hasattr(entity.object_class, "value") else str(entity.object_class))
            )
            self._append_registered_label(candidates, raw_label)

        if not candidates:
            for label in self.detector_vocabulary[:3]:
                self._append_registered_label(candidates, label)
                if len(candidates) >= 3:
                    break

        if not candidates:
            for label in _FALLBACK_LABELS:
                self._append_registered_label(candidates, label)

        # Deduplicate while preserving order
        seen = set()
        unique: List[str] = []
        for label in candidates:
            if not label or label in seen:
                continue
            seen.add(label)
            unique.append(label)
        return unique

    def _extract_keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
        keywords: List[str] = []
        for token in tokens:
            token = token.strip("- ")
            if not token or token in _STOPWORDS:
                continue
            if token.endswith("ing") and len(token) <= 5:
                continue
            keywords.append(token)
        return keywords

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def _select_label(
        self,
        visual_embedding: np.ndarray,
        candidates: Sequence[str],
    ) -> Tuple[Optional[str], float]:
        best_label: Optional[str] = None
        best_score: float = -1.0
        for label in candidates:
            text_embedding = self._encode_label(label)
            if text_embedding is None:
                continue
            score = float(np.dot(visual_embedding, text_embedding))
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, best_score

    def _encode_label(self, label: str) -> Optional[np.ndarray]:
        label = label.strip()
        if not label:
            return None
        if label in self._text_cache:
            return self._text_cache[label]

        prompt = self.config.prompt_template.format(label=label)
        try:
            embedding = self.clip_model.encode_text(prompt, normalize=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("CLIP encode_text failed for '%s': %s", label, exc)
            return None

        self._text_cache[label] = embedding
        return embedding

    def _normalize_label(self, label: Optional[str]) -> Optional[str]:
        if not label:
            return None
        return label.strip().lower()

    def _register_label(self, label: Optional[str]) -> Optional[str]:
        normalized = self._normalize_label(label)
        if not normalized:
            return None
        cleaned = label.strip() if isinstance(label, str) else normalized
        if normalized not in self._label_lookup:
            self._label_lookup[normalized] = cleaned
        self._allowed_labels.add(normalized)
        return normalized

    def _append_registered_label(self, bucket: List[str], label: Optional[str]) -> None:
        normalized = self._register_label(label)
        if not normalized:
            return
        canonical = self._label_lookup.get(normalized)
        if not canonical:
            return
        bucket.append(canonical)
