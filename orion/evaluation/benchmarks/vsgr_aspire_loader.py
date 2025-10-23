"""
VSGR ASPIRe Dataset Loader

Loads and parses the VSGR ASPIRe dataset (from HyperGLM paper) for CIS evaluation.
Dataset format: COCO-style with annotations containing 'reasoning' text
that describes object interactions and causal relationships.

Based on: HyperGLM - Video Scene Graph Generation (arXiv:2411.18042)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger("VSGRASpire")


@dataclass
class ASPIReTrack:
    """Track representing a single object across frames"""
    track_id: int
    category_id: int
    video_id: int
    annotations: List[Dict] = field(default_factory=list)
    

@dataclass
class ASPIReVideo:
    """Video metadata"""
    video_id: int
    name: str
    width: int
    height: int
    num_frames: int
    

@dataclass
class CausalPair:
    """Extracted causal relationship between two objects"""
    video_id: int
    frame_range: Tuple[int, int]
    agent_track_id: int
    patient_track_id: int
    agent_category: str
    patient_category: str
    interaction_type: str
    confidence: float
    reasoning_text: str
    

class VSGRASpireLoader:
    """Loader for VSGR ASPIRe dataset"""
    
    # Common causal/interaction verbs
    CAUSAL_VERBS = {
        'riding', 'holding', 'carrying', 'pushing', 'pulling', 'touching',
        'grabbing', 'throwing', 'catching', 'kicking', 'hitting', 'opening',
        'closing', 'moving', 'placing', 'putting', 'using', 'operating',
        'controlling', 'driving', 'steering', 'manipulating', 'wearing',
        'eating', 'drinking', 'cutting', 'washing', 'cleaning'
    }
    
    def __init__(self, aspire_root: Path):
        """
        Initialize VSGRASpire loader
        
        Args:
            aspire_root: Root directory containing ASPIRe annotations OR direct JSON file path
        """
        self.aspire_root = Path(aspire_root)
        
        # Check if it's a direct JSON file
        self.is_direct_file = self.aspire_root.suffix == '.json'
        
        # Category mapping
        self.categories = {}
        
    def load_split(self, split: str = "train") -> Dict:
        """
        Load annotations for a split
        
        Args:
            split: Dataset split (train/test)
            
        Returns:
            Dictionary with videos, tracks, images, annotations
        """
        # If direct file provided, use it
        if self.is_direct_file:
            ann_file = self.aspire_root
        else:
            ann_file = self.aspire_root / f"aspire_{split}.json"
            
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        
        logger.info(f"Loading VSGR ASPIRe data from {ann_file}")
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Build category mapping
        if 'categories' in data:
            self.categories = {cat['id']: cat['name'] for cat in data['categories']}
            logger.info(f"Loaded {len(self.categories)} categories")
        
        logger.info(f"Loaded {len(data.get('videos', []))} videos, "
                   f"{len(data.get('images', []))} images, "
                   f"{len(data.get('annotations', []))} annotations")
        
        return data
    
    def get_video_ids(self, split: str = "train") -> List[int]:
        """Get list of video IDs in split"""
        data = self.load_split(split)
        return [v['id'] for v in data.get('videos', [])]
    
    def get_videos(self, split: str = "train") -> List[ASPIReVideo]:
        """Get video metadata"""
        data = self.load_split(split)
        videos = []
        
        for v in data.get('videos', []):
            videos.append(ASPIReVideo(
                video_id=v['id'],
                name=v.get('name', f"video_{v['id']}"),
                width=v.get('width', 1920),
                height=v.get('height', 1080),
                num_frames=v.get('num_frames', len([img for img in data.get('images', []) if img['video_id'] == v['id']]))
            ))
        
        return videos
    
    def get_tracks(self, split: str = "train", video_id: Optional[int] = None) -> List[ASPIReTrack]:
        """Get object tracks with their annotations"""
        data = self.load_split(split)
        
        # Group annotations by track
        track_dict = defaultdict(lambda: {'annotations': [], 'category_id': None, 'video_id': None})
        
        for ann in data.get('annotations', []):
            if video_id is not None and ann.get('video_id') != video_id:
                continue
                
            track_id = ann.get('track_id')
            if track_id is None:
                continue
            
            track_dict[track_id]['annotations'].append(ann)
            track_dict[track_id]['category_id'] = ann['category_id']
            track_dict[track_id]['video_id'] = ann.get('video_id', 0)
        
        # Convert to ASPIReTrack objects
        tracks = []
        for track_id, track_data in track_dict.items():
            tracks.append(ASPIReTrack(
                track_id=track_id,
                category_id=track_data['category_id'],
                video_id=track_data['video_id'],
                annotations=sorted(track_data['annotations'], key=lambda x: x.get('image_id', 0))
            ))
        
        return tracks
    
    def extract_causal_relationships(self, split: str = "train", max_videos: Optional[int] = None) -> List[CausalPair]:
        """
        Extract causal relationships from reasoning text
        
        Args:
            split: Dataset split
            max_videos: Maximum number of videos to process
            
        Returns:
            List of CausalPair objects
        """
        data = self.load_split(split)
        causal_pairs = []
        
        video_ids = [v['id'] for v in data.get('videos', [])]
        if max_videos:
            video_ids = video_ids[:max_videos]
        
        logger.info(f"Extracting causal relationships from {len(video_ids)} videos...")
        
        for video_id in video_ids:
            pairs = self._extract_video_causality(data, video_id)
            causal_pairs.extend(pairs)
        
        logger.info(f"Extracted {len(causal_pairs)} causal pairs from {len(video_ids)} videos")
        return causal_pairs
    
    def _extract_video_causality(self, data: Dict, video_id: int) -> List[CausalPair]:
        """Extract causal pairs from a single video"""
        causal_pairs = []
        
        # Get all annotations for this video
        video_anns = [ann for ann in data.get('annotations', []) if ann.get('video_id') == video_id]
        
        # Group by image (frame)
        frame_anns = defaultdict(list)
        for ann in video_anns:
            frame_anns[ann['image_id']].append(ann)
        
        # Process each frame
        for image_id, anns in frame_anns.items():
            # Look for causal interactions in reasoning text
            for ann in anns:
                reasoning = ann.get('reasoning', {})
                reasoning_text = ""
                
                # Extract reasoning text
                if isinstance(reasoning, dict):
                    for key, value in reasoning.items():
                        if isinstance(value, str):
                            reasoning_text += value + " "
                elif isinstance(reasoning, str):
                    reasoning_text = reasoning
                
                reasoning_text = reasoning_text.lower()
                
                # Check for causal verbs
                causal_matches = self._find_causal_interactions(reasoning_text, ann, anns)
                causal_pairs.extend(causal_matches)
        
        return causal_pairs
    
    def _find_causal_interactions(self, text: str, agent_ann: Dict, all_anns: List[Dict]) -> List[CausalPair]:
        """Find causal interactions in text"""
        pairs = []
        
        # Get agent info
        agent_category = self.categories.get(agent_ann['category_id'], 'unknown')
        
        # Look for interactions with other objects
        for verb in self.CAUSAL_VERBS:
            if verb in text:
                # Find potential patient objects mentioned near the verb
                for patient_ann in all_anns:
                    if patient_ann['id'] == agent_ann['id']:
                        continue
                    
                    patient_category = self.categories.get(patient_ann['category_id'], 'unknown')
                    
                    # Check if patient is mentioned near the verb
                    if patient_category in text:
                        # Calculate confidence based on proximity in text
                        confidence = self._calculate_confidence(text, verb, agent_category, patient_category)
                        
                        if confidence > 0.3:  # Threshold
                            pairs.append(CausalPair(
                                video_id=agent_ann.get('video_id', 0),
                                frame_range=(agent_ann['image_id'], agent_ann['image_id']),
                                agent_track_id=agent_ann.get('track_id', -1),
                                patient_track_id=patient_ann.get('track_id', -1),
                                agent_category=agent_category,
                                patient_category=patient_category,
                                interaction_type=verb,
                                confidence=confidence,
                                reasoning_text=text[:200]  # First 200 chars
                            ))
        
        return pairs
    
    def _calculate_confidence(self, text: str, verb: str, agent: str, patient: str) -> float:
        """Calculate confidence score for a causal relationship"""
        # Simple heuristic: check word proximity
        words = text.split()
        
        try:
            verb_idx = next(i for i, w in enumerate(words) if verb in w)
            agent_idx = next((i for i, w in enumerate(words) if agent in w), -1)
            patient_idx = next((i for i, w in enumerate(words) if patient in w), -1)
            
            if agent_idx == -1 or patient_idx == -1:
                return 0.0
            
            # Closer words = higher confidence
            max_distance = 10
            agent_dist = abs(verb_idx - agent_idx)
            patient_dist = abs(verb_idx - patient_idx)
            
            if agent_dist < max_distance and patient_dist < max_distance:
                confidence = 1.0 - (agent_dist + patient_dist) / (2 * max_distance)
                return max(0.3, min(1.0, confidence))
        
        except StopIteration:
            pass
        
        return 0.3  # Default low confidence
    
    def export_to_cis_format(self, causal_pairs: List[CausalPair], output_path: Path):
        """
        Export causal pairs to format compatible with CIS optimizer
        
        Args:
            causal_pairs: List of extracted causal pairs
            output_path: Output JSON file path
        """
        ground_truth = []
        
        for pair in causal_pairs:
            ground_truth.append({
                "video_id": pair.video_id,
                "frame_range": list(pair.frame_range),
                "agent_id": f"track_{pair.agent_track_id}",
                "patient_id": f"track_{pair.patient_track_id}",
                "agent_class": pair.agent_category,
                "patient_class": pair.patient_category,
                "interaction_type": pair.interaction_type,
                "is_causal": True,  # All extracted pairs are positive examples
                "confidence": pair.confidence,
                "annotation_source": "vsgr_aspire_reasoning",
                "reasoning": pair.reasoning_text
            })
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                "dataset": "vsgr_aspire",
                "total_pairs": len(ground_truth),
                "causal_pairs": ground_truth
            }, f, indent=2)
        
        logger.info(f"Exported {len(ground_truth)} causal pairs to {output_path}")
