"""
Class Correction System
=======================

Fast, deterministic correction of YOLO misclassifications with optional LLM.

Pipeline:
1) CLIP/threshold heuristics (trust high confidence)
2) FastVLM description rules + synonyms → canonical_label (e.g., 'knob')
3) Optional LLM refinement (off by default)

We keep both a canonical_label and a mapped COCO class for compatibility.

Author: Orion Research Team
Date: October 2025
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from .config import OrionConfig

logger = logging.getLogger('orion.class_correction')


class ClassCorrector:
    """Corrects YOLO misclassifications using FastVLM descriptions"""
    
    # Common YOLO misclassifications and their typical actual classes
    COMMON_CORRECTIONS = {
        'hair drier': ['knob', 'handle', 'dial', 'button', 'switch'],
        'potted plant': ['star', 'decoration', 'ornament', 'light'],
        'cell phone': ['remote', 'card', 'label', 'tag'],
        'refrigerator': ['cabinet', 'door', 'wall', 'panel'],
        'suitcase': ['tire', 'wheel', 'cushion', 'bag'],
        'backpack': ['bag', 'case', 'container'],
        'bottle': ['cylinder', 'tube', 'container'],
        'cat': ['dog', 'animal', 'pet'],
        'dog': ['cat', 'animal', 'pet'],
    }
    
    # Valid COCO classes (80 classes from YOLO)
    VALID_CLASSES = {
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    }
    
    def __init__(self, config: Optional[OrionConfig] = None, model_manager: Optional[object] = None, llm_model: str = "gemma3:4b"):
        self.config = config or OrionConfig()
        self.llm_model = llm_model
        self.model_manager = model_manager  # Unused for now, reserved for CLIP verify
    
    def should_correct(
        self,
        yolo_class: str,
        description: str,
        confidence: float,
        clip_verified: bool = True
    ) -> bool:
        """
        Determine if a class should be corrected
        
        Args:
            yolo_class: YOLO's predicted class
            description: FastVLM's description
            confidence: YOLO's confidence score
            clip_verified: Whether CLIP verified the class
            
        Returns:
            True if correction is needed
        """
        # If CLIP verified and high confidence, trust YOLO
        if clip_verified and confidence > 0.7:
            return False
        
        # If CLIP failed verification, definitely correct
        if not clip_verified:
            return True
        
        # Check if YOLO class is mentioned in description
        desc_lower = description.lower()
        yolo_lower = yolo_class.lower()
        
        # Direct mention - probably correct
        if yolo_lower in desc_lower:
            return False
        
        # Check for synonyms
        synonyms = {
            'tv': ['monitor', 'screen', 'display'],
            'laptop': ['computer', 'notebook'],
            'cell phone': ['phone', 'smartphone'],
        }
        
        if yolo_class in synonyms:
            if any(syn in desc_lower for syn in synonyms[yolo_class]):
                return False  # Synonym found, don't correct
        
        # Check for common misclassifications
        if yolo_class in self.COMMON_CORRECTIONS:
            # Check if description mentions what it actually is
            actual_classes = self.COMMON_CORRECTIONS[yolo_class]
            if any(actual in desc_lower for actual in actual_classes):
                return True
        
        # Low confidence and no mention = correct
        if confidence < 0.4:  # Raised threshold to be more conservative
            return True
        
        return False
    
    def extract_corrected_class(
        self,
        yolo_class: str,
        description: str,
        use_llm: Optional[bool] = None
    ) -> Tuple[Optional[str], float]:
        """
        Extract the corrected class from description
        
        Args:
            yolo_class: Original YOLO class
            description: FastVLM description
            use_llm: Whether to use LLM for extraction
            
        Returns:
            (corrected_class, confidence)
        """
        # First try simple keyword matching
        corrected, conf = self._keyword_extraction(description, yolo_class)
        
        if corrected and conf > 0.7:
            return corrected, conf
        
        # If keyword matching failed and LLM available, try LLM
        if use_llm is None:
            use_llm = bool(self.config.correction.use_llm)
        if use_llm:
            try:
                llm_corrected, llm_conf = self._llm_extraction(description, yolo_class)
                if llm_corrected and llm_conf > conf:
                    return llm_corrected, llm_conf
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")
        
        # Return keyword result or None
        return corrected, conf
    
    def _keyword_extraction(
        self,
        description: str,
        yolo_class: str
    ) -> Tuple[Optional[str], float]:
        """Extract class using keyword matching"""
        desc_lower = description.lower()
        
        # Look for "appears to be a/an X" patterns
        patterns = [
            r'appears to be (?:a|an) ([a-z\s]+)',
            r'looks like (?:a|an) ([a-z\s]+)',
            r'seems to be (?:a|an) ([a-z\s]+)',
            r'(?:is|are) (?:a|an) ([a-z\s]+)',
            r'depicts (?:a|an) ([a-z\s]+)',
            r'shows (?:a|an) ([a-z\s]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, desc_lower)
            for match in matches:
                # Clean up the match
                candidate = match.strip()
                
                # Try to map to COCO class
                mapped = self._map_to_coco_class(candidate)
                if mapped and mapped != yolo_class:
                    logger.info(f"Keyword extraction: '{yolo_class}' → '{mapped}' (from: '{candidate}')")
                    return mapped, 0.8
        
        # Look for common misclassification corrections
        if yolo_class in self.COMMON_CORRECTIONS:
            for actual_class in self.COMMON_CORRECTIONS[yolo_class]:
                if actual_class in desc_lower:
                    mapped = self._map_to_coco_class(actual_class)
                    if mapped:
                        logger.info(f"Common correction: '{yolo_class}' → '{mapped}'")
                        return mapped, 0.7
        
        return None, 0.0
    
    def _map_to_coco_class(self, text: str) -> Optional[str]:
        """Map a text or canonical label to nearest COCO class"""
        text = text.strip().lower()
        
        # Direct match
        if text in self.VALID_CLASSES:
            return text
        
        # Plural handling
        if text.endswith('s') and text[:-1] in self.VALID_CLASSES:
            return text[:-1]
        
        # Common mappings
        mappings = {
            'knob': 'bottle',  # Represent as small cylindrical for compatibility
            'handle': 'bottle',
            'dial': 'bottle',
            'button': 'bottle',
            'remote control': 'remote',
            'monitor': 'tv',
            'screen': 'tv',
            'display': 'tv',
            'notebook': 'book',
            'computer': 'laptop',
            'phone': 'cell phone',
            'smartphone': 'cell phone',
            'tire': 'car',  # Part of car
            'wheel': 'bicycle',  # Part of bicycle
            'cushion': 'couch',  # Part of couch
            'pillow': 'couch',
            'bag': 'backpack',
            'cabinet': 'refrigerator',
            'decoration': 'vase',
            'ornament': 'vase',
        }
        
        if text in mappings:
            return mappings[text]
        
        # Word-based matching
        for word in text.split():
            if word in self.VALID_CLASSES:
                return word
            if word in mappings:
                return mappings[word]
        
        return None

    def _canonical_from_description(self, description: str, yolo_class: str) -> Optional[str]:
        """Extract a fine-grained canonical label from the description (non-COCO allowed)."""
        text = description.lower()
        # High-precision patterns for parts and small objects
        candidates = [
            'knob', 'door knob', 'handle', 'door handle', 'dial', 'button', 'switch',
            'hinge', 'latch', 'doorknob', 'lever',
        ]
        for c in candidates:
            if c in text:
                # Normalize variants
                if c in ('door knob', 'doorknob'):
                    return 'knob'
                if c in ('door handle',):
                    return 'handle'
                return c
        # Fallback from common corrections
        if yolo_class in self.COMMON_CORRECTIONS:
            for alt in self.COMMON_CORRECTIONS[yolo_class]:
                if alt in text:
                    return alt
        return None
    
    def _llm_extraction(
        self,
        description: str,
        yolo_class: str
    ) -> Tuple[Optional[str], float]:
        """Extract class using LLM"""
        prompt = f"""Given this object description, what is the most accurate object class?

Description: {description[:300]}

YOLO detector classified this as: {yolo_class}

Choose the BEST matching class from this list:
{', '.join(sorted(self.VALID_CLASSES))}

Respond with ONLY the class name, nothing else. If the description matches the YOLO class, respond with the YOLO class. If not, choose the best alternative.

Best class:"""
        
        try:
            # Lazy import to avoid hard dependency
            import ollama  # type: ignore
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1}  # Low temp for consistency
            )
            
            answer = response["message"]["content"].strip().lower()
            
            # Extract just the class name (remove any extra words)
            words = answer.split()
            for word in words:
                if word in self.VALID_CLASSES:
                    if word != yolo_class:
                        logger.info(f"LLM extraction: '{yolo_class}' → '{word}'")
                        return word, 0.9
                    else:
                        return None, 0.0  # LLM agrees with YOLO
            
            # Try multi-word classes
            if answer in self.VALID_CLASSES:
                if answer != yolo_class:
                    logger.info(f"LLM extraction: '{yolo_class}' → '{answer}'")
                    return answer, 0.9
            
        except Exception as e:
            logger.warning(f"LLM query failed: {e}")
        
        return None, 0.0
    
    def apply_corrections(
        self,
        entities: List[Dict],
        use_llm: Optional[bool] = None
    ) -> Tuple[List[Dict], Dict[str, Tuple[str, str, float]]]:
        """
        Apply corrections to all entities
        
        Args:
            entities: List of entity dictionaries
            use_llm: Whether to use LLM for extraction
            
        Returns:
            (corrected_entities, correction_map)
        """
        correction_map: Dict[str, Tuple[str, str, float]] = {}
        corrected_count = 0
        
        for entity in entities:
            original_class = entity['class']
            description = entity.get('description', '')
            confidence = entity.get('described_from_confidence', 0.5)
            
            # Skip low-confidence detections that were already flagged
            if 'Low confidence detection' in description:
                continue
            
            # Check if correction needed (assume clip_verified unknown=False path)
            if self.should_correct(original_class, description, confidence, clip_verified=False):
                # Try to extract corrected class
                corrected_class, corr_confidence = self.extract_corrected_class(
                    original_class,
                    description,
                    use_llm=use_llm
                )
                # Determine canonical label if enabled
                canonical_label = None
                if self.config.correction.enable_canonical_labels:
                    canonical_label = self._canonical_from_description(description, original_class)
                
                if corrected_class and corr_confidence > 0.6:
                    logger.info(f"Correcting {entity['id']}: '{original_class}' → '{corrected_class}'")
                    entity['class'] = corrected_class
                    entity['original_yolo_class'] = original_class
                    entity['correction_confidence'] = corr_confidence
                    if canonical_label:
                        entity['canonical_label'] = canonical_label
                    correction_map[entity['id']] = (original_class, corrected_class, corr_confidence)
                    corrected_count += 1
                else:
                    # If no mapped class but we have a canonical, keep YOLO class and add canonical
                    if canonical_label:
                        entity['canonical_label'] = canonical_label
            else:
                # Even if not correcting, we may set canonical label for finer QA
                if self.config.correction.enable_canonical_labels:
                    canon = self._canonical_from_description(description, original_class)
                    if canon:
                        entity['canonical_label'] = canon
        
        logger.info(f"Applied {corrected_count} class corrections")
        return entities, correction_map


# ============================================================================
# INTEGRATION WITH TRACKING RESULTS
# ============================================================================

def correct_tracking_results(
    tracking_results: Dict,
    use_llm: bool = True,
    save_corrected: bool = True
) -> Dict:
    """
    Correct class labels in tracking results
    
    Args:
        tracking_results: Output from tracking engine
        use_llm: Whether to use LLM for correction
        save_corrected: Whether to save corrected version
        
    Returns:
        Corrected tracking results
    """
    logger.info("="*80)
    logger.info("CLASS CORRECTION")
    logger.info("="*80)
    
    corrector = ClassCorrector()
    
    entities = tracking_results.get('entities', [])
    corrected_entities, correction_map = corrector.apply_corrections(entities, use_llm=use_llm)
    
    tracking_results['entities'] = corrected_entities
    tracking_results['class_corrections'] = correction_map
    tracking_results['num_corrections'] = len(correction_map)
    
    logger.info(f"Total corrections: {len(correction_map)}")
    for entity_id, correction in correction_map.items():
        logger.info(f"  {entity_id}: {correction}")
    
    logger.info("="*80 + "\n")
    
    # Optionally save corrected results
    if save_corrected:
        import json
        from pathlib import Path
        
        output_path = Path('data/testing') / "tracking_results_corrected.json"
        
        with open(output_path, 'w') as f:
            json.dump(tracking_results, f, indent=2)
        
        logger.info(f"Saved corrected results to: {output_path}")
    
    return tracking_results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import json
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    
    # Load tracking results
    results_path = Path("data/testing/tracking_results_save1.json")
    
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        exit(1)
    
    with open(results_path) as f:
        tracking_results = json.load(f)
    
    logger.info(f"Loaded {len(tracking_results['entities'])} entities")
    
    # Apply corrections
    corrected_results = correct_tracking_results(tracking_results, use_llm=True)
    
    logger.info(f"\n✓ Corrected {corrected_results['num_corrections']} misclassifications")
