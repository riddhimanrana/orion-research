"""
Strategic FastVLM Captioning for Spatial Memory

Implements intelligent entity selection and batch captioning to maximize
semantic richness while staying within time budget (~22s overhead for 60s video).

Strategy:
1. Score entities by importance (detection confidence, tracking duration, visibility)
2. Select top 15 entities for captioning
3. Batch caption to minimize model loading overhead
4. Parse captions for structured attributes (color, material, state, etc.)
5. Store in spatial memory for LLM querying

Time Budget: 15 captions × 1.5s = 22.5s overhead (acceptable for <90s target)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class EntityScore:
    """Importance score for deciding which entities to caption"""
    entity_id: int
    score: float
    reason: str  # Why this entity is important


@dataclass
class ParsedCaption:
    """Structured caption with extracted attributes"""
    raw_caption: str
    
    # Extracted attributes
    colors: List[str]
    materials: List[str]
    states: List[str]  # e.g., "open", "closed", "folded"
    relationships: List[str]  # e.g., "on table", "near window"
    descriptors: List[str]  # e.g., "large", "small", "modern"


class StrategicCaptioner:
    """
    Selects and captions the most important entities for spatial memory
    
    Selection criteria (weighted):
    - High confidence detections (avoid captioning noisy detections)
    - Long tracking duration (persistent objects more important)
    - Good visibility (clear, centered, large objects)
    - Class diversity (caption different object types)
    - Spatial coverage (caption objects from different zones)
    
    Usage:
        captioner = StrategicCaptioner(budget=15)
        
        # After processing video, select entities to caption
        selected = captioner.select_entities_to_caption(
            tracks=all_tracks,
            zones=zone_info
        )
        
        # Batch caption them
        captions = captioner.generate_captions(
            selected_entities=selected,
            frame_crops=stored_crops,
            fastvlm_model=model
        )
        
        # Parse for structured attributes
        parsed = [captioner.parse_caption(c) for c in captions]
    """
    
    def __init__(
        self,
        caption_budget: int = 15,
        min_confidence: float = 0.6,
        min_tracking_frames: int = 5,
        diversity_weight: float = 0.3
    ):
        """
        Initialize strategic captioner
        
        Args:
            caption_budget: Max number of captions to generate
            min_confidence: Minimum detection confidence to consider
            min_tracking_frames: Minimum frames tracked to be captionable
            diversity_weight: Weight for class diversity in scoring
        """
        self.caption_budget = caption_budget
        self.min_confidence = min_confidence
        self.min_tracking_frames = min_tracking_frames
        self.diversity_weight = diversity_weight
        
        # Color keywords for parsing
        self.color_keywords = {
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
            'black', 'white', 'gray', 'grey', 'brown', 'beige', 'tan',
            'silver', 'gold', 'navy', 'maroon', 'teal', 'cyan', 'magenta',
            'dark', 'light', 'bright', 'pale'
        }
        
        # Material keywords
        self.material_keywords = {
            'wooden', 'wood', 'metal', 'metallic', 'glass', 'plastic',
            'fabric', 'leather', 'cotton', 'wool', 'ceramic', 'porcelain',
            'stone', 'marble', 'concrete', 'rubber', 'paper', 'cardboard',
            'steel', 'aluminum', 'brass', 'copper'
        }
        
        # State keywords
        self.state_keywords = {
            'open', 'opened', 'closed', 'shut', 'folded', 'unfolded',
            'stacked', 'empty', 'full', 'clean', 'dirty', 'lit', 'unlit',
            'on', 'off', 'hanging', 'sitting', 'standing', 'lying'
        }
        
        logger.info(f"[StrategicCaptioner] Initialized (budget={caption_budget})")
    
    def score_entity_importance(
        self,
        entity: Dict,
        all_classes: List[str],
        captioned_classes: set,
        zone_distribution: Dict[int, int]
    ) -> EntityScore:
        """
        Score an entity's importance for captioning
        
        Args:
            entity: Entity info (id, class, confidence, frames_tracked, zone_id, bbox_size)
            all_classes: All unique classes in video
            captioned_classes: Classes already captioned
            zone_distribution: Count of entities per zone
            
        Returns:
            EntityScore with total score and reasoning
        """
        score = 0.0
        reasons = []
        
        # 1. Confidence score (0-1)
        confidence = entity.get('confidence', 0.5)
        if confidence > 0.8:
            score += 0.3
            reasons.append("high_confidence")
        elif confidence > 0.6:
            score += 0.15
        
        # 2. Tracking duration score (0-0.3)
        frames_tracked = entity.get('frames_tracked', 0)
        if frames_tracked > 50:
            score += 0.3
            reasons.append("long_tracked")
        elif frames_tracked > 20:
            score += 0.2
        elif frames_tracked > 10:
            score += 0.1
        
        # 3. Visibility score (0-0.2)
        bbox_size = entity.get('bbox_size', 0)  # Normalized 0-1
        if bbox_size > 0.1:  # >10% of frame
            score += 0.2
            reasons.append("highly_visible")
        elif bbox_size > 0.05:
            score += 0.1
        
        # 4. Class diversity bonus (0-0.2)
        entity_class = entity.get('class_name', 'unknown')
        if entity_class not in captioned_classes:
            score += 0.2 * self.diversity_weight
            reasons.append("new_class")
        
        # 5. Zone diversity bonus (0-0.1)
        zone_id = entity.get('zone_id', 0)
        if zone_distribution.get(zone_id, 0) < 2:  # Few entities from this zone
            score += 0.1
            reasons.append("zone_coverage")
        
        return EntityScore(
            entity_id=entity['entity_id'],
            score=score,
            reason=", ".join(reasons) if reasons else "baseline"
        )
    
    def select_entities_to_caption(
        self,
        tracks: List[Dict],
        zones: Optional[Dict] = None
    ) -> List[Tuple[int, float, str]]:
        """
        Select top N entities for captioning based on importance
        
        Args:
            tracks: List of entity track info
            zones: Optional zone information
            
        Returns:
            List of (entity_id, score, reason) tuples, sorted by score
        """
        # Filter by minimum requirements
        eligible = []
        for track in tracks:
            confidence = track.get('confidence', 0.0)
            frames_tracked = track.get('frames_tracked', 0)
            
            if confidence >= self.min_confidence and frames_tracked >= self.min_tracking_frames:
                eligible.append(track)
        
        if len(eligible) == 0:
            logger.warning("[StrategicCaptioner] No eligible entities for captioning")
            return []
        
        # Get class distribution
        all_classes = list(set(e.get('class_name', 'unknown') for e in tracks))
        captioned_classes = set()  # Will be updated as we select
        
        # Get zone distribution
        zone_distribution = {}
        for e in tracks:
            zone_id = e.get('zone_id', 0)
            zone_distribution[zone_id] = zone_distribution.get(zone_id, 0) + 1
        
        # Score all eligible entities
        scored = []
        for entity in eligible:
            score_obj = self.score_entity_importance(
                entity, all_classes, captioned_classes, zone_distribution
            )
            scored.append((score_obj.entity_id, score_obj.score, score_obj.reason))
            
            # Update captioned classes for diversity scoring
            if score_obj.score > 0.5:  # Likely to be selected
                captioned_classes.add(entity.get('class_name', 'unknown'))
        
        # Sort by score (descending) and take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[:self.caption_budget]
        
        logger.info(f"[StrategicCaptioner] Selected {len(selected)}/{len(eligible)} entities")
        for entity_id, score, reason in selected[:5]:  # Log top 5
            logger.info(f"  Entity {entity_id}: score={score:.2f} ({reason})")
        
        return selected
    
    def parse_caption(self, caption: str) -> ParsedCaption:
        """
        Parse caption to extract structured attributes
        
        Args:
            caption: Raw caption text
            
        Returns:
            ParsedCaption with extracted attributes
        """
        caption_lower = caption.lower()
        words = set(re.findall(r'\b\w+\b', caption_lower))
        
        # Extract colors
        colors = [c for c in self.color_keywords if c in words]
        
        # Extract materials
        materials = [m for m in self.material_keywords if m in words]
        
        # Extract states
        states = [s for s in self.state_keywords if s in words]
        
        # Extract relationships (simple patterns)
        relationships = []
        rel_patterns = [
            r'on (?:the |a )?(\w+)',
            r'near (?:the |a )?(\w+)',
            r'next to (?:the |a )?(\w+)',
            r'under (?:the |a )?(\w+)',
            r'above (?:the |a )?(\w+)',
            r'in (?:the |a )?(\w+)',
        ]
        for pattern in rel_patterns:
            matches = re.findall(pattern, caption_lower)
            relationships.extend(matches)
        
        # Extract descriptors (size, quality adjectives)
        size_descriptors = ['large', 'small', 'big', 'tiny', 'huge', 'medium']
        quality_descriptors = ['new', 'old', 'modern', 'vintage', 'antique', 'clean', 'worn']
        descriptors = [d for d in size_descriptors + quality_descriptors if d in words]
        
        return ParsedCaption(
            raw_caption=caption,
            colors=colors,
            materials=materials,
            states=states,
            relationships=relationships,
            descriptors=descriptors
        )
    
    def format_caption_for_memory(
        self,
        entity_id: int,
        class_name: str,
        parsed: ParsedCaption
    ) -> Dict:
        """
        Format parsed caption for spatial memory storage
        
        Args:
            entity_id: Entity identifier
            class_name: Object class
            parsed: Parsed caption
            
        Returns:
            Dictionary ready for memory storage
        """
        return {
            'entity_id': entity_id,
            'class_name': class_name,
            'caption': parsed.raw_caption,
            'attributes': {
                'colors': parsed.colors,
                'materials': parsed.materials,
                'states': parsed.states,
                'relationships': parsed.relationships,
                'descriptors': parsed.descriptors
            },
            'searchable_text': ' '.join([
                class_name,
                parsed.raw_caption,
                *parsed.colors,
                *parsed.materials,
                *parsed.descriptors
            ]).lower()
        }
    
    def estimate_captioning_time(self, num_captions: int) -> float:
        """
        Estimate time to generate captions
        
        Args:
            num_captions: Number of captions to generate
            
        Returns:
            Estimated time in seconds
        """
        # Based on FastVLM-0.5B benchmarks:
        # - Model loading: ~2s (one-time)
        # - Per-caption: ~1.5s average
        loading_time = 2.0
        per_caption_time = 1.5
        
        total_time = loading_time + (num_captions * per_caption_time)
        return total_time
    
    def get_statistics(self) -> Dict:
        """Get captioning statistics"""
        return {
            'caption_budget': self.caption_budget,
            'min_confidence': self.min_confidence,
            'min_tracking_frames': self.min_tracking_frames,
            'estimated_time_15_captions': self.estimate_captioning_time(15)
        }
