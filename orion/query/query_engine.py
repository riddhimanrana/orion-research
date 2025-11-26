"""
Query Engine - Natural language queries over processed video
=============================================================

Uses LLM to parse queries and retrieve relevant information.

Author: Orion Research Team
Date: November 2025
"""

import re
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .index import VideoIndex, EntityObservation


@dataclass
class QueryResult:
    """Result of a visual query"""
    question: str
    answer: str
    entities: List[EntityObservation]
    confidence: float


class QueryEngine:
    """
    Natural language query engine for processed videos.
    Supports queries like:
    - "What color was the book?"
    - "Where did I see the laptop?"
    - "What was on the desk?"
    """
    
    def __init__(self, index: VideoIndex, video_path: Path, fastvlm_model=None):
        self.index = index
        self.video_path = Path(video_path)
        self.fastvlm = fastvlm_model
        self.video_cap = None
        
        # Fetch available classes from index
        self.available_classes = self._get_available_classes()
        
    def _get_available_classes(self) -> List[str]:
        """Get list of all object classes in the index"""
        if not self.index.conn:
            return []
        cursor = self.index.conn.cursor()
        try:
            cursor.execute("SELECT DISTINCT class_name FROM observations")
            return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []

    def parse_query(self, query: str) -> Tuple[str, str]:
        """
        Parse natural language query to extract:
        - object class
        - question type (color, location, description)
        """
        query_lower = query.lower()
        
        # Extract object class (match against available classes)
        detected_class = None
        
        # Sort by length descending to match "potted plant" before "plant"
        sorted_classes = sorted(self.available_classes, key=len, reverse=True)
        
        for obj_class in sorted_classes:
            if obj_class in query_lower:
                detected_class = obj_class
                break
        
        # Fallback to hardcoded list if index is empty or no match found
        if not detected_class:
            object_classes = [
                'book', 'laptop', 'keyboard', 'mouse', 'phone', 'monitor', 'tv',
                'cup', 'bottle', 'person', 'chair', 'desk', 'backpack', 'bird', 'cat', 
                'traffic light', 'potted plant', 'sofa', 'refrigerator', 'bed', 'truck'
            ]
            for obj_class in object_classes:
                if obj_class in query_lower:
                    detected_class = obj_class
                    break
        
        # Determine question type
        if 'color' in query_lower or 'what color' in query_lower:
            question_type = 'color'
        elif 'where' in query_lower or 'location' in query_lower:
            question_type = 'location'
        elif 'what' in query_lower or 'describe' in query_lower:
            question_type = 'description'
        else:
            question_type = 'general'
        
        return detected_class, question_type
    
    def query(self, question: str) -> QueryResult:
        """
        Execute a visual query.
        
        Examples:
            "What color was the book?"
            "Where was the laptop?"
            "What was on the desk?"
        """
        # Parse query
        object_class, question_type = self.parse_query(question)
        
        if not object_class:
            return QueryResult(
                question=question,
                answer="I couldn't identify which object you're asking about. Try: 'What color was the book?'",
                entities=[],
                confidence=0.0
            )
        
        # Find observations of this object
        observations = self.index.query_by_class(object_class)
        
        if not observations:
            return QueryResult(
                question=question,
                answer=f"I didn't see any {object_class} in the video.",
                entities=[],
                confidence=0.0
            )
        
        # Check if we already have a caption
        captioned_obs = [obs for obs in observations if obs.caption]
        
        if not captioned_obs:
            # Generate caption on-demand
            print(f"\n🔍 Generating caption for {object_class}...")
            captioned_obs = self._caption_entities(observations)
            
        # Filter out inconsistent captions (hallucinations)
        consistent_obs = []
        for obs in captioned_obs:
            if self._verify_consistency(obs.class_name, obs.caption):
                consistent_obs.append(obs)
            else:
                print(f"  ⚠️  Filtered inconsistent observation for {obs.class_name} (ID {obs.entity_id})")
        
        if not consistent_obs:
             return QueryResult(
                question=question,
                answer=f"I detected a {object_class}, but upon closer inspection, it appears to be a false detection.",
                entities=[],
                confidence=0.1
            )
            
        captioned_obs = consistent_obs
        
        # Answer the query based on type
        if question_type == 'color':
            answer = self._extract_color(captioned_obs)
        elif question_type == 'location':
            answer = self._extract_location(captioned_obs)
        elif question_type == 'description':
            answer = self._extract_description(captioned_obs)
        else:
            answer = self._extract_description(captioned_obs)
        
        return QueryResult(
            question=question,
            answer=answer,
            entities=captioned_obs,
            confidence=0.9 if captioned_obs else 0.3
        )
    
    def _verify_consistency(self, class_name: str, caption: str) -> bool:
        """
        Check if the VLM caption supports the YOLO class detection.
        Returns True if consistent, False if likely a hallucination.
        """
        if not caption:
            return True # Benefit of the doubt
            
        caption_lower = caption.lower()
        class_lower = class_name.lower()
        
        # Direct match
        if class_lower in caption_lower:
            return True
            
        # Split class name (e.g. "potted plant" -> "plant")
        parts = class_lower.split()
        if any(part in caption_lower for part in parts if len(part) > 2):
            return True
            
        # Synonyms/Related words (basic list)
        synonyms = {
            'tv': ['monitor', 'screen', 'display', 'television'],
            'monitor': ['screen', 'display', 'tv'],
            'couch': ['sofa', 'furniture', 'seat'],
            'sofa': ['couch', 'furniture', 'seat'],
            'plant': ['flower', 'leaf', 'leaves', 'vegetation', 'tree', 'sprout'],
            'potted plant': ['flower', 'leaf', 'leaves', 'vegetation', 'pot', 'sprout'],
            'person': ['man', 'woman', 'human', 'hand', 'arm', 'leg', 'face', 'boy', 'girl'],
            'book': ['notebook', 'journal', 'paper', 'text', 'binder'],
            'notebook': ['book', 'journal', 'paper', 'text', 'binder'],
            'laptop': ['computer', 'screen', 'keyboard'],
            'keyboard': ['keys', 'typing'],
            'mouse': ['clicker'],
            'bird': ['animal', 'creature', 'wing', 'feather'],
            'cat': ['animal', 'creature', 'feline', 'kitten'],
            'dog': ['animal', 'creature', 'canine', 'puppy'],
            'truck': ['vehicle', 'car', 'auto'],
            'car': ['vehicle', 'truck', 'auto'],
            'bed': ['blanket', 'pillow', 'mattress', 'sleep'],
            'refrigerator': ['fridge', 'freezer', 'appliance', 'kitchen'],
            'chandelier': ['light', 'lamp', 'fixture', 'ceiling'],
        }
        
        for key, values in synonyms.items():
            if key in class_lower:
                if any(val in caption_lower for val in values):
                    return True
                    
        return False

    def _caption_entities(self, observations: List[EntityObservation]) -> List[EntityObservation]:
        """Generate captions for entities on-demand"""
        if not self.fastvlm:
            return observations
        
        # Open video
        if not self.video_cap:
            self.video_cap = cv2.VideoCapture(str(self.video_path))
        
        captioned = []
        seen_entities = set()
        
        for obs in observations:
            if obs.entity_id in seen_entities:
                captioned.append(obs)
                continue
            
            seen_entities.add(obs.entity_id)
            
            # Seek to frame
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, obs.frame_idx)
            ret, frame = self.video_cap.read()
            
            if not ret:
                captioned.append(obs)
                continue
            
            # Crop entity
            x1, y1, x2, y2 = [int(v) for v in obs.bbox]
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                captioned.append(obs)
                continue
            
            # Generate caption
            try:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                from PIL import Image
                crop_pil = Image.fromarray(crop_rgb)
                
                prompt = f"Describe this {obs.class_name} in detail, including its color, appearance, and any distinctive features:"
                caption = self.fastvlm.generate_description(
                    crop_pil,
                    prompt,
                    max_tokens=128,
                    temperature=0.3
                )
                
                # Verify consistency
                if not self._verify_consistency(obs.class_name, caption):
                    print(f"  ⚠️  Inconsistent caption for {obs.class_name} (ID {obs.entity_id}): {caption[:60]}...")
                    # We still save it, but maybe we flag it? 
                    # For now, let's prepend a warning to the caption so query logic can see it
                    # Or better, just log it. The query logic will filter it.
                
                # Update database
                obs.caption = caption
                self.index.update_caption(obs.entity_id, obs.frame_idx, caption)
                print(f"  ✓ Captioned entity {obs.entity_id}: {caption[:60]}...")
                
            except Exception as e:
                print(f"  ⚠️  Failed to caption entity {obs.entity_id}: {e}")
            
            captioned.append(obs)
        
        self.index.commit()
        return captioned
    
    def _extract_color(self, observations: List[EntityObservation]) -> str:
        """Extract color information from captions"""
        if not observations or not observations[0].caption:
            return "I don't have enough visual information to determine the color."
        
        caption = observations[0].caption.lower()
        obj_class = observations[0].class_name.lower()
        
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'grey',
                  'brown', 'orange', 'purple', 'pink', 'silver', 'gold', 'beige', 'tan', 'maroon']
        
        # Prioritize "is [color]" or "[color] [object]"
        strong_matches = []
        for color in colors:
            # Regex patterns for strong association
            patterns = [
                f"is {color}",
                f"{color} {obj_class}",
                f"{color} color",
                f"color of the .* is {color}"
            ]
            for pattern in patterns:
                if re.search(pattern, caption):
                    strong_matches.append(color)
                    break
        
        if strong_matches:
            return f"The {observations[0].class_name} is {strong_matches[0]}. {observations[0].caption}"
            
        # Fallback to simple presence
        found_colors = [color for color in colors if color in caption]
        if found_colors:
             return f"The {observations[0].class_name} appears to be {found_colors[0]}. {observations[0].caption}"
             
        return f"I couldn't determine the exact color, but here's what I saw: {observations[0].caption}"
    
    def _extract_location(self, observations: List[EntityObservation]) -> str:
        """Extract location information"""
        if not observations:
            return "I couldn't determine the location."
        
        obs = observations[0]
        
        # Get zone info
        if obs.zone_id is not None:
            zone_obs = self.index.query_by_zone(obs.zone_id)
            zone_classes = set(o.class_name for o in zone_obs)
            location_desc = f"in an area with {', '.join(list(zone_classes)[:3])}"
        else:
            location_desc = "in the scene"
        
        # Get temporal info
        timestamp = obs.timestamp
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        
        if obs.caption:
            return f"I saw the {obs.class_name} {location_desc} at {minutes}:{seconds:02d}. {obs.caption}"
        else:
            return f"I saw the {obs.class_name} {location_desc} at {minutes}:{seconds:02d}."
    
    def _extract_description(self, observations: List[EntityObservation]) -> str:
        """Extract general description"""
        if not observations or not observations[0].caption:
            obj = observations[0].class_name if observations else "object"
            return f"I saw the {obj} but don't have detailed visual information."
        
        return observations[0].caption
    
    def close(self):
        """Clean up resources"""
        if self.video_cap:
            self.video_cap.release()
