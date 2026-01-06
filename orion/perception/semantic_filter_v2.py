"""
Semantic Filter v2 for Orion v2

Advanced scene-based filtering using:
1. SentenceTransformer (all-mpnet-base-v2) for semantic label-to-scene matching
2. Scene change detection using frame embedding similarity
3. Scene type classification with per-type label blacklists
4. VLM verification for suspicious high-frequency detections

This replaces the CLIP-based scene_filter.py with a more robust approach
that addresses the refrigerator false positive issue.

Author: Orion Research Team
Date: January 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded models
_sentence_model = None


def get_sentence_transformer():
    """Lazy-load SentenceTransformer model (all-mpnet-base-v2).
    
    This model provides 768-dim embeddings optimized for semantic similarity.
    Better than CLIP for text-to-text matching.
    """
    global _sentence_model
    
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading all-mpnet-base-v2 for semantic filtering...")
            _sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            logger.info("✓ SentenceTransformer loaded (768-dim)")
        except ImportError as e:
            logger.warning(f"SentenceTransformer not available: {e}")
            return None
    
    return _sentence_model


# Scene type classification
SCENE_TYPES = {
    "bedroom": {
        "keywords": ["bed", "bedroom", "pillow", "mattress", "blanket", "nightstand", "wardrobe", "closet", "sleep"],
        "blacklist": ["refrigerator", "toilet", "sink", "oven", "microwave", "bathtub", "shower", "stove"],
        "expected": ["bed", "chair", "laptop", "tv", "lamp", "clock", "book", "cell phone"],
    },
    "office": {
        "keywords": ["desk", "office", "computer", "monitor", "work", "keyboard", "mouse", "chair", "workspace"],
        "blacklist": ["refrigerator", "toilet", "bed", "oven", "microwave", "bathtub", "shower", "stove", "sink"],
        "expected": ["laptop", "keyboard", "mouse", "monitor", "tv", "chair", "book", "cell phone", "clock", "bottle", "cup"],
    },
    "kitchen": {
        "keywords": ["kitchen", "stove", "oven", "refrigerator", "fridge", "cook", "cooking", "cabinet", "counter", "sink"],
        "blacklist": ["bed", "toilet", "bathtub", "shower"],
        "expected": ["refrigerator", "oven", "microwave", "sink", "bottle", "cup", "bowl", "knife", "spoon", "fork", "dining table", "chair"],
    },
    "bathroom": {
        "keywords": ["bathroom", "toilet", "sink", "shower", "bathtub", "mirror", "towel", "bath"],
        "blacklist": ["refrigerator", "bed", "oven", "microwave", "laptop", "keyboard", "dining table", "couch"],
        "expected": ["toilet", "sink", "toothbrush", "bottle", "cup"],
    },
    "living_room": {
        "keywords": ["living room", "couch", "sofa", "tv", "television", "coffee table", "living", "lounge"],
        "blacklist": ["toilet", "bathtub", "shower", "oven", "stove"],
        "expected": ["couch", "chair", "tv", "remote", "book", "clock", "vase", "potted plant", "bottle"],
    },
    "hallway": {
        "keywords": ["hallway", "corridor", "door", "entrance", "hall", "doorway", "passage"],
        "blacklist": ["refrigerator", "toilet", "bathtub", "oven", "microwave", "bed", "couch", "dining table", "sink"],
        "expected": ["door", "person", "clock", "vase", "potted plant"],
    },
    "dining": {
        "keywords": ["dining", "table", "eat", "meal", "dinner", "lunch", "breakfast"],
        "blacklist": ["toilet", "bathtub", "shower", "bed"],
        "expected": ["dining table", "chair", "bottle", "cup", "bowl", "wine glass", "fork", "knife", "spoon", "vase"],
    },
}

# Labels that are commonly confused/false positives
SUSPICIOUS_LABELS = {
    "refrigerator": {
        "min_scene_similarity": 0.60,  # Higher threshold than default
        "requires_vlm_verification": True,
        "vlm_check_keywords": ["refrigerator", "fridge", "appliance", "kitchen"],
        "common_confusions": ["door", "closet", "cabinet", "wardrobe"],
    },
    "toilet": {
        "min_scene_similarity": 0.55,
        "requires_vlm_verification": True,
        "vlm_check_keywords": ["toilet", "bathroom"],
        "common_confusions": ["chair", "vase", "bowl"],
    },
    "airplane": {
        "min_scene_similarity": 0.70,  # Very unlikely indoors
        "requires_vlm_verification": True,
        "vlm_check_keywords": ["airplane", "plane", "aircraft", "toy plane", "model"],
        "common_confusions": ["ceiling fan", "light fixture"],
    },
    "boat": {
        "min_scene_similarity": 0.70,
        "requires_vlm_verification": True,
        "vlm_check_keywords": ["boat", "ship", "vessel", "toy boat", "model"],
        "common_confusions": ["bowl", "container"],
    },
    "sports ball": {
        "min_scene_similarity": 0.55,
        "requires_vlm_verification": False,
        "vlm_check_keywords": ["ball", "sports ball", "soccer", "basketball", "football"],
        "common_confusions": ["round object", "globe", "vase"],
    },
}

# Semantic aliases for better matching
SEMANTIC_ALIASES = {
    "tv": ["monitor", "screen", "display", "television"],
    "laptop": ["computer", "notebook", "macbook"],
    "dining table": ["desk", "table", "work surface", "counter"],
    "couch": ["sofa", "settee", "loveseat"],
    "bed": ["mattress", "bedding"],
    "keyboard": ["computer keyboard", "typing device"],
    "mouse": ["computer mouse", "input device"],
    "cell phone": ["phone", "smartphone", "mobile", "iphone", "android"],
    "remote": ["remote control", "tv remote"],
    "potted plant": ["plant", "houseplant", "flower"],
}


@dataclass
class SemanticFilterV2Config:
    """Configuration for enhanced semantic filtering."""
    
    # Base similarity thresholds
    min_similarity: float = 0.30
    """Base minimum similarity for keeping detections (SentenceTransformer scale)."""
    
    high_confidence_threshold: float = 0.45
    """High-confidence threshold (definitely keep)."""
    
    # Scene change detection
    scene_change_threshold: float = 0.75
    """Cosine similarity below this triggers a scene change."""
    
    min_frames_between_scene_changes: int = 15
    """Minimum frames between scene updates (prevents flicker)."""
    
    # VLM verification
    enable_vlm_verification: bool = True
    """Enable VLM verification for suspicious labels."""
    
    vlm_verification_sample_rate: float = 0.2
    """Sample rate for VLM verification (fraction of suspicious detections)."""
    
    max_vlm_verifications_per_frame: int = 2
    """Maximum VLM verifications per frame."""
    
    # Label frequency thresholds
    suspicious_label_count_threshold: int = 20
    """If a label appears this many times, flag for extra scrutiny."""
    
    # Caching
    cache_embeddings: bool = True
    """Cache label embeddings."""
    
    # Device
    device: str = "cuda"
    """Device for model inference."""


@dataclass
class SceneState:
    """Current scene state for multi-room tracking."""
    caption: str
    scene_type: Optional[str]
    embedding: Optional[np.ndarray]
    frame_idx: int
    blacklist: Set[str] = field(default_factory=set)
    expected: Set[str] = field(default_factory=set)


class SemanticFilterV2:
    """
    Enhanced semantic filter with scene change detection and VLM verification.
    
    Key improvements over v1 (CLIP-based):
    1. Uses SentenceTransformer for better text similarity
    2. Detects scene changes and updates context dynamically
    3. Classifies scene type (bedroom/office/kitchen) for smart blacklists
    4. Verifies suspicious high-frequency detections with VLM
    
    Usage:
        filter = SemanticFilterV2()
        
        # Set initial scene
        filter.set_scene("A desk with computer monitor and keyboard", frame_idx=0)
        
        # As video progresses, check for scene changes
        if filter.should_update_scene(new_caption, frame_idx=30):
            filter.set_scene(new_caption, frame_idx=30)
        
        # Filter detections
        for det in detections:
            result = filter.check_detection(det["label"], det["confidence"], det)
            if not result.is_valid:
                det["filtered"] = True
    """
    
    def __init__(self, config: Optional[SemanticFilterV2Config] = None, vlm=None):
        self.config = config or SemanticFilterV2Config()
        self._sentence_model = None
        self._vlm = vlm
        self._current_scene: Optional[SceneState] = None
        self._scene_history: List[SceneState] = []
        self._label_cache: Dict[str, np.ndarray] = {}
        self._label_counts: Counter = Counter()
        self._vlm_verification_cache: Dict[str, Tuple[bool, str]] = {}  # label@bbox -> (is_valid, reason)
        self._last_scene_change_frame: int = -999
        
    def _ensure_sentence_model(self) -> bool:
        """Ensure sentence transformer is loaded."""
        if self._sentence_model is None:
            self._sentence_model = get_sentence_transformer()
        return self._sentence_model is not None
    
    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Embed text using SentenceTransformer."""
        if not self._ensure_sentence_model():
            return None
        
        try:
            embedding = self._sentence_model.encode(text, convert_to_numpy=True)
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding
        except Exception as e:
            logger.warning(f"Text embedding failed: {e}")
            return None
    
    def _get_label_embedding(self, label: str) -> Optional[np.ndarray]:
        """Get embedding for a label (cached)."""
        if self.config.cache_embeddings and label in self._label_cache:
            return self._label_cache[label]
        
        # Expand label for better context
        text = f"A {label} object in a room or scene"
        
        emb = self._embed_text(text)
        
        if self.config.cache_embeddings and emb is not None:
            self._label_cache[label] = emb
        
        return emb
    
    def classify_scene_type(self, caption: str) -> Optional[str]:
        """Classify scene into a type (bedroom/office/kitchen/etc).
        
        Args:
            caption: Scene caption from VLM.
            
        Returns:
            Scene type string or None if unknown.
        """
        caption_lower = caption.lower()
        
        best_type = None
        best_score = 0
        
        for scene_type, info in SCENE_TYPES.items():
            score = sum(1 for kw in info["keywords"] if kw in caption_lower)
            if score > best_score:
                best_score = score
                best_type = scene_type
        
        # Require at least 1 keyword match
        return best_type if best_score >= 1 else None
    
    def set_scene(
        self, 
        caption: str, 
        frame_idx: int = 0,
        force: bool = False,
    ) -> SceneState:
        """Set the current scene context.
        
        Args:
            caption: Scene caption from VLM.
            frame_idx: Current frame index.
            force: Force update even if within min_frames threshold.
            
        Returns:
            New SceneState.
        """
        if not caption or not caption.strip():
            if self._current_scene:
                return self._current_scene
            caption = "Unknown scene"
        
        # Check minimum frames between changes
        if not force and (frame_idx - self._last_scene_change_frame) < self.config.min_frames_between_scene_changes:
            if self._current_scene:
                return self._current_scene
        
        # Embed caption
        embedding = self._embed_text(caption)
        
        # Classify scene type
        scene_type = self.classify_scene_type(caption)
        
        # Build blacklist and expected sets
        blacklist: Set[str] = set()
        expected: Set[str] = set()
        
        if scene_type and scene_type in SCENE_TYPES:
            info = SCENE_TYPES[scene_type]
            blacklist = set(info.get("blacklist", []))
            expected = set(info.get("expected", []))
        
        # Create scene state
        state = SceneState(
            caption=caption.strip(),
            scene_type=scene_type,
            embedding=embedding,
            frame_idx=frame_idx,
            blacklist=blacklist,
            expected=expected,
        )
        
        self._current_scene = state
        self._scene_history.append(state)
        self._last_scene_change_frame = frame_idx
        
        # Clear VLM cache on scene change
        self._vlm_verification_cache.clear()
        
        logger.info(f"Scene set: type={scene_type}, caption='{caption[:60]}...'")
        if blacklist:
            logger.info(f"  Blacklist: {sorted(blacklist)}")
        
        return state
    
    def should_update_scene(self, new_caption: str, frame_idx: int) -> bool:
        """Check if scene has changed significantly.
        
        Args:
            new_caption: New scene caption to compare.
            frame_idx: Current frame index.
            
        Returns:
            True if scene should be updated.
        """
        if self._current_scene is None:
            return True
        
        # Check minimum frames
        if (frame_idx - self._last_scene_change_frame) < self.config.min_frames_between_scene_changes:
            return False
        
        if self._current_scene.embedding is None:
            return True
        
        new_embedding = self._embed_text(new_caption)
        if new_embedding is None:
            return False
        
        # Compute similarity
        similarity = float(np.dot(self._current_scene.embedding, new_embedding))
        
        # If similarity is low, scene has changed
        return similarity < self.config.scene_change_threshold
    
    def compute_similarity(self, label: str) -> float:
        """Compute similarity between label and current scene.
        
        Uses semantic aliases and expanded matching.
        
        Args:
            label: Object label.
            
        Returns:
            Similarity score (0-1).
        """
        if self._current_scene is None or self._current_scene.embedding is None:
            return 0.5  # Neutral
        
        # Get base similarity
        label_emb = self._get_label_embedding(label)
        if label_emb is None:
            return 0.5
        
        best_sim = float(np.dot(self._current_scene.embedding, label_emb))
        
        # Try semantic aliases
        if label in SEMANTIC_ALIASES:
            for alias in SEMANTIC_ALIASES[label]:
                alias_emb = self._get_label_embedding(alias)
                if alias_emb is not None:
                    alias_sim = float(np.dot(self._current_scene.embedding, alias_emb))
                    if alias_sim > best_sim:
                        best_sim = alias_sim
        
        return best_sim
    
    def _vlm_verify(self, label: str, crop: Any) -> Tuple[bool, str]:
        """Verify a detection using VLM.
        
        Args:
            label: Detected label to verify.
            crop: PIL Image crop of the detection.
            
        Returns:
            Tuple of (is_valid, reason).
        """
        if self._vlm is None:
            return True, "vlm_unavailable"
        
        if label not in SUSPICIOUS_LABELS:
            return True, "not_suspicious"
        
        try:
            from PIL import Image
            
            # Generate description
            prompt = f"What is this object? Is it a {label}? Answer briefly."
            description = self._vlm.generate_description(crop, prompt, max_tokens=50)
            description_lower = description.lower()
            
            # Check for target keywords
            keywords = SUSPICIOUS_LABELS[label].get("vlm_check_keywords", [label])
            has_keyword = any(kw.lower() in description_lower for kw in keywords)
            
            # Check for confusion keywords
            confusions = SUSPICIOUS_LABELS[label].get("common_confusions", [])
            has_confusion = any(conf.lower() in description_lower for conf in confusions)
            
            if has_keyword and not has_confusion:
                return True, f"vlm_confirmed: {description[:50]}"
            elif has_confusion:
                return False, f"vlm_confused: {description[:50]}"
            else:
                # Ambiguous - use stricter threshold
                return False, f"vlm_ambiguous: {description[:50]}"
                
        except Exception as e:
            logger.warning(f"VLM verification failed: {e}")
            return True, f"vlm_error: {e}"
    
    def check_detection(
        self,
        label: str,
        confidence: float = 0.5,
        detection: Optional[Dict[str, Any]] = None,
        crop: Optional[Any] = None,
    ) -> Tuple[bool, float, str]:
        """Check if a detection should be kept.
        
        This is the main entry point for filtering.
        
        Args:
            label: Detection label.
            confidence: Detection confidence.
            detection: Full detection dict (for metadata).
            crop: Optional PIL Image for VLM verification.
            
        Returns:
            Tuple of (is_valid, similarity, reason).
        """
        # Track label counts
        self._label_counts[label] += 1
        
        # Check blacklist first (hard rejection)
        if self._current_scene and label in self._current_scene.blacklist:
            return False, 0.0, f"blacklisted_for_{self._current_scene.scene_type}"
        
        # Check if expected (easy pass)
        if self._current_scene and label in self._current_scene.expected:
            return True, 1.0, "expected_for_scene"
        
        # Compute semantic similarity
        similarity = self.compute_similarity(label)
        
        # Get threshold (use stricter for suspicious labels)
        min_threshold = self.config.min_similarity
        if label in SUSPICIOUS_LABELS:
            min_threshold = max(min_threshold, SUSPICIOUS_LABELS[label].get("min_scene_similarity", min_threshold))
        
        # High confidence = definitely keep
        if similarity >= self.config.high_confidence_threshold:
            return True, similarity, "high_similarity"
        
        # Pass similarity threshold
        if similarity >= min_threshold:
            # Additional VLM check for suspicious labels
            if (
                label in SUSPICIOUS_LABELS 
                and SUSPICIOUS_LABELS[label].get("requires_vlm_verification", False)
                and self.config.enable_vlm_verification
                and crop is not None
            ):
                is_valid, reason = self._vlm_verify(label, crop)
                if not is_valid:
                    return False, similarity, reason
            
            return True, similarity, "passes_threshold"
        
        # Below threshold = filter out
        return False, similarity, "below_threshold"
    
    def filter_detections(
        self,
        detections: List[Dict[str, Any]],
        label_key: str = "object_class",
        confidence_key: str = "confidence",
        in_place: bool = True,
    ) -> List[Dict[str, Any]]:
        """Filter a list of detections.
        
        Args:
            detections: List of detection dicts.
            label_key: Key for object label.
            confidence_key: Key for confidence.
            in_place: Modify dicts in place.
            
        Returns:
            Filtered list (only valid detections).
        """
        if self._current_scene is None:
            return detections
        
        filtered = []
        removed_by_reason: Counter = Counter()
        
        for det in detections:
            label = str(det.get(label_key, "unknown"))
            confidence = float(det.get(confidence_key, 0.5) or 0.5)
            
            is_valid, similarity, reason = self.check_detection(label, confidence, det)
            
            if in_place:
                det["scene_similarity_v2"] = similarity
                det["scene_filter_reason_v2"] = reason
            
            if is_valid:
                filtered.append(det)
            else:
                removed_by_reason[reason] += 1
                if in_place:
                    det["scene_filtered_v2"] = True
        
        if removed_by_reason:
            logger.info(f"SemanticFilterV2: Removed {sum(removed_by_reason.values())}/{len(detections)}")
            for reason, count in removed_by_reason.most_common(5):
                logger.info(f"  - {reason}: {count}")
        
        return filtered
    
    def get_label_stats(self) -> Dict[str, int]:
        """Get label frequency statistics."""
        return dict(self._label_counts)
    
    def get_suspicious_labels(self) -> List[str]:
        """Get labels that appear suspiciously often."""
        threshold = self.config.suspicious_label_count_threshold
        return [label for label, count in self._label_counts.items() if count >= threshold]
    
    def reset(self):
        """Reset filter state."""
        self._current_scene = None
        self._scene_history = []
        self._label_counts.clear()
        self._vlm_verification_cache.clear()
        self._last_scene_change_frame = -999
        logger.debug("SemanticFilterV2 reset")


def create_semantic_filter_v2(
    device: str = "cuda",
    vlm=None,
    **kwargs,
) -> SemanticFilterV2:
    """Factory function to create SemanticFilterV2.
    
    Args:
        device: Device for model inference.
        vlm: Optional VLM for verification (FastVLM instance).
        **kwargs: Additional config options.
        
    Returns:
        Configured SemanticFilterV2 instance.
    """
    config = SemanticFilterV2Config(device=device, **kwargs)
    return SemanticFilterV2(config=config, vlm=vlm)
