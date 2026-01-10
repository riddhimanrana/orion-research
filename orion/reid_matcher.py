"""
Re-ID + CLIP Integration for Cross-View Object Deduplication

Phase 5: Semantic matching to merge tracks representing the same object
from different viewpoints or poses.

Strategy:
1. Compute similarity between track embeddings using CLIP
2. Build bipartite graph of similar tracks
3. Merge highly similar tracks as single entity
4. Maintains track ID consistency for visualization
"""

import numpy as np
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class EmbeddingMatch:
    """Match between two tracks based on embedding similarity"""
    track_id_1: int
    track_id_2: int
    similarity: float  # Cosine similarity 0-1
    match_type: str  # 'perfect', 'high', 'medium'


class ReIDMatcher:
    """
    Matches tracks across viewpoints using semantic embeddings.
    
    Handles:
    - CLIP embeddings for appearance matching
    - Cross-view person re-identification
    - Object appearance consistency over time
    - Temporal gaps (different frames/videos)
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        merge_threshold: float = 0.75,
        min_embedding_age: int = 2,  # Need 2+ embedding observations
    ):
        """
        Initialize Re-ID matcher.
        
        Args:
            similarity_threshold: Min cosine similarity to consider a match
            merge_threshold: Higher threshold for actually merging tracks
            min_embedding_age: Require N embedding observations before matching
        """
        self.similarity_threshold = similarity_threshold
        self.merge_threshold = merge_threshold
        self.min_embedding_age = min_embedding_age
        
        self.matches: list[EmbeddingMatch] = []
        self.merged_tracks: dict[int, int] = {}  # Maps old_id → canonical_id
    
    def match_tracks(self, tracks: dict) -> list[EmbeddingMatch]:
        """
        Find matches between tracks using embeddings.
        
        Args:
            tracks: dict of {track_id: TrackedObject}
        
        Returns:
            list of EmbeddingMatch objects
        """
        self.matches = []
        track_ids = sorted(tracks.keys())
        
        # Compare all pairs
        for i, tid1 in enumerate(track_ids):
            track1 = tracks[tid1]
            
            # Skip if not enough embedding history
            if len(track1.embeddings) < self.min_embedding_age:
                continue
            
            for tid2 in track_ids[i+1:]:
                track2 = tracks[tid2]
                
                # Skip if not enough embedding history
                if len(track2.embeddings) < self.min_embedding_age:
                    continue
                
                # Different classes never match (unless generic)
                if track1.class_name != track2.class_name:
                    continue
                
                # Compute similarity
                similarity = self._compute_embedding_similarity(
                    track1.embeddings,
                    track2.embeddings
                )
                
                if similarity >= self.similarity_threshold:
                    # Classify match strength
                    if similarity >= self.merge_threshold:
                        match_type = 'perfect'
                    elif similarity >= (self.similarity_threshold + self.merge_threshold) / 2:
                        match_type = 'high'
                    else:
                        match_type = 'medium'
                    
                    match = EmbeddingMatch(
                        track_id_1=tid1,
                        track_id_2=tid2,
                        similarity=similarity,
                        match_type=match_type,
                    )
                    self.matches.append(match)
        
        # Sort by similarity (best first)
        self.matches.sort(key=lambda x: -x.similarity)
        return self.matches
    
    def _compute_embedding_similarity(
        self,
        embeddings1: list[np.ndarray],
        embeddings2: list[np.ndarray]
    ) -> float:
        """
        Compute cosine similarity between embedding sequences.
        
        Uses robust averaging: median of best-match similarities
        """
        if not embeddings1 or not embeddings2:
            return 0.0
        
        # Filter out None embeddings
        valid_emb1 = [e for e in embeddings1 if e is not None]
        valid_emb2 = [e for e in embeddings2 if e is not None]
        
        if not valid_emb1 or not valid_emb2:
            return 0.0
        
        # Use median embedding from each track
        median_emb1 = np.median(np.array(valid_emb1), axis=0)
        median_emb2 = np.median(np.array(valid_emb2), axis=0)
        
        # Cosine similarity
        norm1 = np.linalg.norm(median_emb1)
        norm2 = np.linalg.norm(median_emb2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        similarity = np.dot(median_emb1, median_emb2) / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))
    
    def merge_tracks_by_similarity(
        self,
        min_similarity: float = 0.75
    ) -> dict[int, list[int]]:
        """
        Merge tracks with high embedding similarity.
        
        Returns:
            dict mapping canonical_id → list of merged track_ids
        """
        # Build graph of high-similarity matches
        connections = defaultdict(set)
        
        for match in self.matches:
            if match.similarity >= min_similarity:
                connections[match.track_id_1].add(match.track_id_2)
                connections[match.track_id_2].add(match.track_id_1)
        
        # Find connected components (transitive closure)
        merged_groups = {}
        visited = set()
        
        for track_id in connections:
            if track_id not in visited:
                # BFS to find all connected tracks
                component = self._find_component(track_id, connections)
                visited.update(component)
                
                # Use smallest ID as canonical
                canonical_id = min(component)
                merged_groups[canonical_id] = sorted(list(component))
        
        return merged_groups
    
    def _find_component(self, start_id: int, connections: dict) -> Set[int]:
        """Find connected component in graph (BFS)"""
        component = set()
        queue = [start_id]
        
        while queue:
            current = queue.pop(0)
            if current in component:
                continue
            
            component.add(current)
            for neighbor in connections.get(current, []):
                if neighbor not in component:
                    queue.append(neighbor)
        
        return component
    
    def get_match_statistics(self) -> dict:
        """Get statistics about found matches"""
        if not self.matches:
            return {
                'total_matches': 0,
                'perfect_matches': 0,
                'high_matches': 0,
                'medium_matches': 0,
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
            }
        
        perfect = sum(1 for m in self.matches if m.match_type == 'perfect')
        high = sum(1 for m in self.matches if m.match_type == 'high')
        medium = sum(1 for m in self.matches if m.match_type == 'medium')
        similarities = [m.similarity for m in self.matches]
        
        return {
            'total_matches': len(self.matches),
            'perfect_matches': perfect,
            'high_matches': high,
            'medium_matches': medium,
            'avg_similarity': float(np.mean(similarities)),
            'max_similarity': float(np.max(similarities)),
            'min_similarity': float(np.min(similarities)),
        }


class CrossViewMerger:
    """
    Merges tracker results across multiple views/cameras.
    
    Integrates Re-ID matching to create unified object identities.
    """
    
    def __init__(self, reid_matcher: Optional[ReIDMatcher] = None):
        """Initialize merger with Re-ID matcher"""
        self.reid_matcher = reid_matcher or ReIDMatcher()
        self.merged_tracks = {}
    
    def merge_all_tracks(
        self,
        all_tracks: dict,  # dict of {track_id: TrackedObject}
    ) -> tuple[dict, dict]:
        """
        Merge similar tracks into unified identities.
        
        Args:
            all_tracks: dictionary of tracked objects
        
        Returns:
            tuple of:
            - merged_tracks: dict with deduplicated track_ids
            - merge_groups: dict mapping canonical_id → list of merged_ids
        """
        # Find matches
        matches = self.reid_matcher.match_tracks(all_tracks)
        
        # Merge by similarity
        merge_groups = self.reid_matcher.merge_tracks_by_similarity(
            min_similarity=0.75
        )
        
        # Create merged track representation
        merged_tracks = {}
        merged_mapping = {}  # old_id → canonical_id
        
        for canonical_id, member_ids in merge_groups.items():
            # Use track with most observations as canonical
            canonical_track = max(
                [all_tracks[tid] for tid in member_ids],
                key=lambda t: t.age
            )
            merged_tracks[canonical_id] = canonical_track
            
            for member_id in member_ids:
                merged_mapping[member_id] = canonical_id
        
        # Add non-merged tracks
        for track_id, track in all_tracks.items():
            if track_id not in merged_mapping:
                merged_tracks[track_id] = track
        
        return merged_tracks, merge_groups
    
    def get_statistics(self) -> dict:
        """Get merger statistics"""
        stats = self.reid_matcher.get_match_statistics()
        stats['merged_groups'] = len(self.merged_tracks)
        return stats
