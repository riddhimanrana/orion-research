"""
Class Correction System
=======================

Fast, deterministic correction of YOLO misclassifications with optional LLM and CLIP.

Pipeline:
1) CLIP/threshold heuristics (trust high confidence)
2) FastVLM description rules + synonyms → canonical_label (e.g., 'knob')
3) Optional CLIP semantic matching for verification
4) Optional LLM refinement (off by default)

We keep both a canonical_label and a mapped COCO class for compatibility.

Author: Orion Research Team
Date: October 2025
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import OrionConfig

logger = logging.getLogger('orion.class_correction')


class ClassCorrector:
    """Corrects YOLO misclassifications using FastVLM descriptions"""
    
    # REMOVED: Rule-based corrections - we use pure semantic validation now
    
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
        self.model_manager = model_manager
        self._clip_model = None  # Lazy load for verification
        self._sentence_model = None  # Lazy load sentence transformer
        self._class_embeddings_cache = None  # Precomputed class embeddings
        self._semantic_validator = None  # Lazy load for description-class validation
    
    def _get_clip_model(self):
        """Lazy load CLIP model for semantic verification"""
        if self._clip_model is None:
            try:
                from .embedding_model import EmbeddingModel
                self._clip_model = EmbeddingModel(backend="auto")
                logger.info("CLIP model loaded for semantic verification")
            except Exception as e:
                logger.warning(f"Could not load CLIP model: {e}")
                self._clip_model = False  # Mark as failed
        return self._clip_model if self._clip_model is not False else None
    
    def _get_sentence_model(self):
        """Lazy load Sentence Transformer for semantic matching"""
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✓ Sentence Transformer loaded for semantic class correction")
            except Exception as e:
                logger.warning(f"Could not load Sentence Transformer: {e}")
                self._sentence_model = False
        return self._sentence_model if self._sentence_model is not False else None
    
    def _get_class_embeddings(self):
        """Precompute embeddings for all valid COCO classes"""
        if self._class_embeddings_cache is not None:
            return self._class_embeddings_cache
        
        model = self._get_sentence_model()
        if not model:
            return None
        
        try:
            # Create class descriptions for better matching
            class_texts = [f"a {cls}" for cls in sorted(self.VALID_CLASSES)]
            embeddings = model.encode(class_texts, show_progress_bar=False)
            self._class_embeddings_cache = {
                cls: emb for cls, emb in zip(sorted(self.VALID_CLASSES), embeddings)
            }
            logger.debug(f"Precomputed embeddings for {len(self._class_embeddings_cache)} classes")
            return self._class_embeddings_cache
        except Exception as e:
            logger.warning(f"Could not precompute class embeddings: {e}")
            return None
    
    def validate_correction_with_description(
        self,
        description: str,
        original_class: str,
        proposed_class: str,
        threshold: float = 0.25
    ) -> Tuple[bool, float]:
        """
        Validate that proposed class makes sense given the description.
        Uses semantic similarity to prevent bad corrections like "tire" → "car".
        
        Args:
            description: Rich description from VLM
            original_class: Original YOLO classification
            proposed_class: Proposed corrected class
            threshold: Minimum similarity threshold for validation (lowered from 0.5 to 0.25)
            
        Returns:
            (is_valid, similarity_score)
        """
        model = self._get_sentence_model()
        if not model:
            # No validator available, allow correction
            return True, 1.0
        
        try:
            # Encode description and proposed class
            desc_embedding = model.encode(description, show_progress_bar=False)
            
            # Check for part-of relationships FIRST
            # e.g., "car tire" means subject is "tire", not "car"
            desc_lower = description.lower()
            part_indicators = [
                f"{proposed_class} tire", f"{proposed_class} wheel", f"{proposed_class} door",
                f"{proposed_class} handle", f"{proposed_class} knob", f"{proposed_class} button",
                f"part of a {proposed_class}", f"attached to a {proposed_class}",
                f"on the {proposed_class}", f"of the {proposed_class}", f"of a {proposed_class}"
            ]
            
            # If description mentions proposed_class as a modifier/container, REJECT
            if any(indicator in desc_lower for indicator in part_indicators):
                logger.warning(
                    f"⚠ Rejected correction '{original_class}' → '{proposed_class}' "
                    f"(detected as part-of relationship)"
                )
                return False, 0.0
            
            proposed_text = f"This is a {proposed_class}"
            original_text = f"This is a {original_class}"
            
            # Encode both classes
            class_embeddings = model.encode([proposed_text, original_text], show_progress_bar=False)
            proposed_emb = class_embeddings[0]
            original_emb = class_embeddings[1]
            
            # Compute cosine similarity
            def cosine_sim(a, b):
                return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            
            proposed_sim = cosine_sim(desc_embedding, proposed_emb)
            original_sim = cosine_sim(desc_embedding, original_emb)
            
            # Validation logic:
            # 1. Proposed class should be MORE similar to description than original
            # 2. Allow moderate semantic drift (descriptions don't perfectly match class names)
            is_valid = (
                proposed_sim > (original_sim + 0.05)  # Must be better than original by at least 5%
            )
            
            if not is_valid:
                logger.debug(
                    f"⚠ Rejected correction '{original_class}' → '{proposed_class}' "
                    f"(proposed: {proposed_sim:.3f} vs original: {original_sim:.3f}, diff: {proposed_sim - original_sim:.3f})"
                )
            else:
                logger.info(
                    f"✓ Validated correction '{original_class}' → '{proposed_class}' "
                    f"(proposed: {proposed_sim:.3f} vs original: {original_sim:.3f}, diff: {proposed_sim - original_sim:.3f})"
                )
            
            return is_valid, proposed_sim
            
        except Exception as e:
            logger.debug(f"Validation failed: {e}, allowing correction")
            return True, 1.0
    
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
        
        # Check for part-of relationships that shouldn't be corrected
        # e.g., "tire" in a car context, "handle" on a suitcase
        part_of_indicators = {
            'tire': ['car', 'vehicle', 'wheel', 'tread', 'rim'],
            'wheel': ['bicycle', 'car', 'vehicle', 'spoke', 'rim'],
            'handle': ['door', 'suitcase', 'bag', 'drawer'],
            'knob': ['door', 'cabinet', 'drawer', 'oven'],
        }
        
        if yolo_class in part_of_indicators:
            # Check if description suggests this is actually a part of something
            indicators = part_of_indicators[yolo_class]
            if any(ind in desc_lower for ind in indicators):
                # This might be a part, not a standalone object
                # Flag for potential non-correction or LLM review
                logger.debug(f"Detected '{yolo_class}' in part-of context: {indicators}")
                # Return False to avoid bad mapping (e.g., tire → car)
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
    
    def verify_with_clip(
        self,
        description: str,
        yolo_class: str,
        candidate_classes: List[str]
    ) -> Tuple[Optional[str], float]:
        """
        Use CLIP text embeddings to find best matching class
        
        Args:
            description: Object description from VLM
            yolo_class: Original YOLO classification
            candidate_classes: List of candidate classes to check
            
        Returns:
            (best_class, confidence_score)
        """
        clip_model = self._get_clip_model()
        if not clip_model:
            return None, 0.0
        
        try:
            # Generate embeddings for description and candidate classes
            texts = [description[:200]] + [f"a {cls}" for cls in candidate_classes]  # Limit desc length
            embeddings = clip_model.encode(texts)
            
            if embeddings is None or len(embeddings) < 2:
                return None, 0.0
            
            # Compute cosine similarity (embeddings are already normalized)
            desc_emb = embeddings[0]
            candidate_embs = embeddings[1:]
            
            similarities = []
            for i, cand_emb in enumerate(candidate_embs):
                # Cosine similarity (already normalized embeddings)
                sim = float(np.dot(desc_emb, cand_emb))
                similarities.append((candidate_classes[i], sim))
            
            # Find best match
            best_class, best_sim = max(similarities, key=lambda x: x[1])
            
            # Only return if significantly better than YOLO class
            yolo_sim = next((sim for cls, sim in similarities if cls == yolo_class), 0.0)
            
            if best_sim > yolo_sim + 0.1 and best_sim > 0.6:  # Threshold for confidence
                logger.info(f"CLIP verification: '{yolo_class}' → '{best_class}' (sim: {best_sim:.3f} vs {yolo_sim:.3f})")
                return best_class, best_sim
            
        except Exception as e:
            logger.debug(f"CLIP verification failed: {e}")
        
        return None, 0.0
    
    def semantic_class_match(
        self,
        description: str,
        yolo_class: str,
        top_k: int = 5,
        threshold: float = 0.2  # Lowered - trust the embeddings more
    ) -> Tuple[Optional[str], float]:
        """
        Use Sentence Transformers to find best matching class based on description.
        Pure semantic approach - NO RULES.
        
        Args:
            description: Object description from VLM
            yolo_class: Original YOLO classification
            top_k: Number of top matches to consider
            threshold: Minimum similarity threshold
            
        Returns:
            (best_class, confidence_score)
        """
        model = self._get_sentence_model()
        class_embeddings = self._get_class_embeddings()
        
        if not model or not class_embeddings:
            return None, 0.0
        
        try:
            # First, check if description contains likely object nouns
            desc_words = description.lower().split()
            desc_lower = description.lower()
            
            # Check for direct class mentions - BUT verify it's the main subject, not a part
            for cls in self.VALID_CLASSES:
                cls_words = cls.lower().split()
                for cls_word in cls_words:
                    if cls_word in desc_words:
                        # Check if this is describing a part vs the whole thing
                        # e.g., "car tire" means tire is the subject, not car
                        part_indicators = [
                            f"{cls_word} tire", f"{cls_word} wheel", f"{cls_word} door",
                            f"{cls_word} handle", f"{cls_word} knob", f"{cls_word} button",
                            f"part of", f"attached to", f"on the {cls_word}", f"of the {cls_word}"
                        ]
                        
                        # If any part indicator is present, don't treat as direct match
                        if any(indicator in desc_lower for indicator in part_indicators):
                            logger.debug(f"Skipping direct match for '{cls}' - detected as part-of")
                            continue
                        
                        # Direct match found and it's the main subject
                        logger.info(f"✓ Direct match: found '{cls}' in description")
                        return cls, 0.95
            
            # Check for contextual keywords that override similarity
            # e.g., "computer monitor" should be "laptop" not "tv"
            contextual_overrides = {
                'laptop': ['computer', 'notebook computer'],
                'cell phone': ['smartphone', 'phone'],
                'remote': ['remote control'],
            }
            
            for cls, keywords in contextual_overrides.items():
                for kw in keywords:
                    if kw in desc_lower and cls != yolo_class:
                        logger.info(f"✓ Contextual override: '{yolo_class}' → '{cls}' (keyword: '{kw}')")
                        return cls, 0.85
            
            # Encode description and compare with all COCO classes
            desc_embedding = model.encode(description, show_progress_bar=False)
            
            # Compute similarities with all classes
            similarities = []
            for cls, cls_emb in class_embeddings.items():
                # Cosine similarity
                sim = float(np.dot(desc_embedding, cls_emb) / 
                           (np.linalg.norm(desc_embedding) * np.linalg.norm(cls_emb)))
                similarities.append((cls, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_matches = similarities[:top_k]
            
            # Get best match
            best_class, best_sim = top_matches[0]
            
            # Get YOLO class similarity for comparison
            yolo_sim = next((sim for cls, sim in similarities if cls == yolo_class), 0.0)
            
            # Accept if best class is different AND better than YOLO (even slightly)
            if best_class != yolo_class and best_sim > (yolo_sim + 0.03):  # Just 3% better
                logger.info(
                    f"✓ Semantic match: '{yolo_class}' → '{best_class}' "
                    f"(sim: {best_sim:.3f}, YOLO sim: {yolo_sim:.3f})"
                )
                logger.debug(f"Top matches: {[(c, f'{s:.3f}') for c, s in top_matches]}")
                
                return best_class, best_sim
            
            # YOLO class is best match
            return None, 0.0
            
        except Exception as e:
            logger.debug(f"Semantic class matching failed: {e}")
            return None, 0.0
    
    def extract_corrected_class(
        self,
        yolo_class: str,
        description: str,
        use_llm: Optional[bool] = None,
        use_clip: bool = True,
        validate_with_description: bool = True
    ) -> Tuple[Optional[str], float]:
        """
        Pure semantic correction using VLM description and embedding similarity.
        
        NO RULES - just find the best COCO class that matches the rich description.
        
        Args:
            yolo_class: Original YOLO class
            description: Rich VLM description
            use_llm: Ignored (kept for API compatibility)
            use_clip: Ignored (kept for API compatibility)
            validate_with_description: Always True
            
        Returns:
            (corrected_class, confidence)
        """
        # Primary: Semantic matching with all COCO classes
        semantic_corrected, semantic_conf = self.semantic_class_match(description, yolo_class)
        
        if semantic_corrected and semantic_conf > 0.3:  # Lower threshold - trust embeddings
            logger.info(
                f"✓ Semantic uplift: '{yolo_class}' → '{semantic_corrected}' "
                f"(confidence: {semantic_conf:.3f})"
            )
            return semantic_corrected, semantic_conf
        
        # Fallback: Keyword extraction from description
        keyword_class, keyword_conf = self._keyword_extraction(description, yolo_class)
        
        if keyword_class and keyword_conf > 0.5:
            logger.info(
                f"✓ Keyword uplift: '{yolo_class}' → '{keyword_class}' "
                f"(confidence: {keyword_conf:.3f})"
            )
            return keyword_class, keyword_conf
        
        # No correction - keep original
        logger.debug(f"No semantic correction for '{yolo_class}'")
        return yolo_class, 0.0
    
    def _keyword_extraction(
        self,
        description: str,
        yolo_class: str
    ) -> Tuple[Optional[str], float]:
        """Extract class using keyword matching with improved pattern matching"""
        desc_lower = description.lower()
        
        # Enhanced patterns for object identification
        patterns = [
            r'appears to be (?:a|an) ([a-z\s]+)',
            r'looks like (?:a|an) ([a-z\s]+)',
            r'seems to be (?:a|an) ([a-z\s]+)',
            r'(?:is|are) (?:a|an) ([a-z\s]+)',
            r'depicts (?:a|an) ([a-z\s]+)',
            r'shows (?:a|an) ([a-z\s]+)',
            r'which (?:appears to be|looks like|seems to be) (?:a|an) ([a-z\s]+)',
            r'could be (?:a|an) ([a-z\s]+)',
            r'represents (?:a|an) ([a-z\s]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, desc_lower)
            for match in matches:
                # Clean up the match
                candidate = match.strip()
                
                # Skip overly generic matches
                if candidate in ['object', 'item', 'thing', 'piece', 'part']:
                    continue
                
                # Try to map to COCO class
                mapped = self._map_to_coco_class(candidate)
                if mapped and mapped != yolo_class:
                    logger.info(f"Keyword extraction: '{yolo_class}' → '{mapped}' (from: '{candidate}')")
                    return mapped, 0.8
        
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
        
        # Common mappings - IMPROVED to avoid bad mappings
        mappings = {
            # Hardware parts -> reasonable approximations
            'knob': 'remote',  # Small handheld object
            'handle': 'remote',  # Better than mapping to bottle
            'dial': 'remote',
            'button': 'remote',
            'switch': 'remote',
            
            # Electronics
            'remote control': 'remote',
            'monitor': 'tv',
            'screen': 'tv',
            'display': 'tv',
            'notebook': 'book',
            'computer': 'laptop',
            'phone': 'cell phone',
            'smartphone': 'cell phone',
            
            # Vehicle parts -> DON'T map to full vehicle (causes issues)
            # 'tire': None,  # Don't auto-correct, needs context
            # 'wheel': None,  # Don't auto-correct, needs context
            
            # Furniture parts
            'cushion': 'couch',
            'pillow': 'couch',
            
            # Containers
            'bag': 'backpack',
            'container': 'bottle',
            
            # Home items
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
        
        # High-precision patterns for parts and small objects (ordered by priority)
        # First check for complete object descriptions
        if yolo_class == 'cell phone':
            if any(term in text for term in ['remote control', 'remote']):
                return 'remote'
        
        if yolo_class == 'potted plant':
            for term in ['decoration', 'ornament', 'star']:
                if term in text:
                    return 'decoration' if 'decorat' in text else term
        
        # Hardware and door components (high priority for hair drier misclassifications)
        hardware_parts = [
            ('door knob', 'knob'), ('doorknob', 'knob'), ('knob', 'knob'),
            ('door handle', 'handle'), ('handle', 'handle'),
            ('dial', 'dial'), ('switch', 'switch'),
            ('hinge', 'hinge'), ('latch', 'latch'), ('lever', 'lever'),
        ]
        
        for pattern, canonical in hardware_parts:
            if pattern in text:
                return canonical
        
        # Fallback: check common corrections
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
