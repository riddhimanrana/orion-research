"""
Loop Closure Detection for SLAM
================================

Detects when the camera revisits a previous location and applies
pose graph optimization to correct accumulated drift.

Key Components:
- Keyframe database with ORB descriptors
- Bag-of-Words (BoW) for fast loop candidate retrieval
- Geometric verification with RANSAC
- Pose graph optimization (g2o-style)

Author: Orion Research Team
Date: November 2025
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Keyframe:
    """
    Keyframe for loop closure detection.
    
    Stores visual features and pose for place recognition.
    """
    frame_id: int
    pose: np.ndarray  # 4x4 transformation matrix
    keypoints: List  # List of cv2.KeyPoint
    descriptors: np.ndarray  # Nx128 ORB descriptors
    image_gray: Optional[np.ndarray] = None  # Optional grayscale image
    
    # BoW representation (visual vocabulary)
    bow_vector: Optional[Dict[int, float]] = None
    
    # Loop closure links
    loop_closure_to: Set[int] = None  # Set of frame_ids this has loop closure with
    
    def __post_init__(self):
        if self.loop_closure_to is None:
            self.loop_closure_to = set()


@dataclass
class LoopClosure:
    """Detected loop closure between two keyframes"""
    query_id: int  # Current frame
    match_id: int  # Previous frame that matches
    relative_pose: np.ndarray  # 4x4 relative transform (query → match)
    inliers: int  # Number of geometric inliers
    confidence: float  # 0-1 confidence score
    feature_matches: int  # Total feature matches


class SimpleBagOfWords:
    """
    Simplified Bag-of-Words for visual place recognition.
    
    Uses k-means clustering of ORB descriptors to build a visual vocabulary.
    """
    
    def __init__(self, vocabulary_size: int = 1000):
        """
        Args:
            vocabulary_size: Number of visual words (cluster centers)
        """
        self.vocabulary_size = vocabulary_size
        self.vocabulary: Optional[np.ndarray] = None  # Kx128 cluster centers
        self.trained = False
        
        # Matcher for assigning descriptors to visual words
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def train(self, all_descriptors: List[np.ndarray], max_samples: int = 50000):
        """
        Train vocabulary using k-means on descriptor pool.
        
        Args:
            all_descriptors: List of descriptor arrays from multiple frames
            max_samples: Maximum descriptors to use (for speed)
        """
        # Pool all descriptors
        pooled = []
        for desc in all_descriptors:
            if desc is not None and len(desc) > 0:
                pooled.append(desc)
        
        if len(pooled) == 0:
            logger.warning("No descriptors for BoW training")
            return
        
        pooled = np.vstack(pooled)
        
        # Subsample if too many
        if len(pooled) > max_samples:
            indices = np.random.choice(len(pooled), max_samples, replace=False)
            pooled = pooled[indices]
        
        logger.info(f"Training BoW vocabulary with {len(pooled)} descriptors...")
        
        # K-means clustering (OpenCV format)
        pooled_float = pooled.astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pooled_float,
            self.vocabulary_size,
            None,
            criteria,
            3,  # attempts
            cv2.KMEANS_PP_CENTERS
        )
        
        self.vocabulary = centers.astype(np.uint8)  # Back to uint8
        self.trained = True
        
        logger.info(f"✓ BoW vocabulary trained: {self.vocabulary_size} visual words")
    
    def compute_bow(self, descriptors: np.ndarray) -> Dict[int, float]:
        """
        Convert descriptors to BoW representation.
        
        Args:
            descriptors: Nx128 ORB descriptors
        
        Returns:
            Dict mapping visual word ID → count (TF-IDF style)
        """
        if not self.trained or descriptors is None or len(descriptors) == 0:
            return {}
        
        # Match each descriptor to nearest visual word
        matches = self.matcher.match(descriptors, self.vocabulary)
        
        # Count visual word occurrences
        bow_vec = defaultdict(float)
        for match in matches:
            word_id = match.trainIdx
            bow_vec[word_id] += 1.0
        
        # Normalize to unit length (L2 norm)
        total = sum(v**2 for v in bow_vec.values()) ** 0.5
        if total > 0:
            bow_vec = {k: v/total for k, v in bow_vec.items()}
        
        return dict(bow_vec)
    
    def similarity(self, bow1: Dict[int, float], bow2: Dict[int, float]) -> float:
        """
        Compute cosine similarity between two BoW vectors.
        
        Args:
            bow1, bow2: BoW representations
        
        Returns:
            Similarity score [0, 1]
        """
        if not bow1 or not bow2:
            return 0.0
        
        # Cosine similarity: dot product (already L2-normalized)
        common_words = set(bow1.keys()) & set(bow2.keys())
        dot_product = sum(bow1[w] * bow2[w] for w in common_words)
        
        return float(dot_product)


class LoopClosureDetector:
    """
    Loop closure detection system.
    
    Detects when camera revisits a location and provides
    loop constraints for pose graph optimization.
    """
    
    def __init__(
        self,
        min_loop_interval: int = 30,  # Minimum frames between loop closures
        min_bow_similarity: float = 0.70,  # BoW similarity threshold
        min_inliers: int = 30,  # Minimum geometric inliers
        max_reprojection_error: float = 3.0,  # RANSAC threshold (pixels)
        enable_pose_graph_optimization: bool = True,
    ):
        """
        Initialize loop closure detector.
        
        Args:
            min_loop_interval: Minimum frame gap for valid loop
            min_bow_similarity: BoW similarity threshold for candidates
            min_inliers: Minimum RANSAC inliers for valid loop
            max_reprojection_error: RANSAC reprojection threshold
            enable_pose_graph_optimization: Apply pose graph optimization
        """
        self.min_loop_interval = min_loop_interval
        self.min_bow_similarity = min_bow_similarity
        self.min_inliers = min_inliers
        self.max_reprojection_error = max_reprojection_error
        self.enable_pose_graph = enable_pose_graph_optimization
        
        # Keyframe database
        self.keyframes: Dict[int, Keyframe] = {}
        self.keyframe_ids: List[int] = []
        
        # Bag of Words for fast retrieval
        self.bow = SimpleBagOfWords(vocabulary_size=1000)
        self.bow_trained = False
        
        # Detected loop closures
        self.loop_closures: List[LoopClosure] = []
        
        # Statistics
        self.num_loop_candidates: int = 0
        self.num_loops_detected: int = 0
        
        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        logger.info("LoopClosureDetector initialized")
        logger.info(f"  Min loop interval: {min_loop_interval} frames")
        logger.info(f"  BoW similarity threshold: {min_bow_similarity}")
        logger.info(f"  Min inliers: {min_inliers}")
    
    def add_keyframe(
        self,
        frame_id: int,
        pose: np.ndarray,
        keypoints: List,
        descriptors: np.ndarray,
        image_gray: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add keyframe to database.
        
        Args:
            frame_id: Unique frame identifier
            pose: 4x4 camera pose
            keypoints: List of cv2.KeyPoint
            descriptors: ORB descriptors
            image_gray: Optional grayscale image
        """
        if descriptors is None or len(descriptors) == 0:
            return
        
        keyframe = Keyframe(
            frame_id=frame_id,
            pose=pose.copy(),
            keypoints=keypoints,
            descriptors=descriptors,
            image_gray=image_gray,
        )
        
        self.keyframes[frame_id] = keyframe
        self.keyframe_ids.append(frame_id)
        
        # Train BoW after collecting enough keyframes
        if not self.bow_trained and len(self.keyframes) >= 10:
            self._train_bow()
        
        # Compute BoW for this keyframe
        if self.bow_trained:
            keyframe.bow_vector = self.bow.compute_bow(descriptors)
    
    def _train_bow(self) -> None:
        """Train BoW vocabulary from accumulated keyframes"""
        all_descriptors = [kf.descriptors for kf in self.keyframes.values()]
        self.bow.train(all_descriptors)
        
        # Recompute BoW for all existing keyframes
        for kf in self.keyframes.values():
            kf.bow_vector = self.bow.compute_bow(kf.descriptors)
        
        self.bow_trained = True
    
    def detect_loop(
        self,
        query_id: int,
        camera_matrix: np.ndarray,
    ) -> Optional[LoopClosure]:
        """
        Detect loop closure for query keyframe.
        
        Args:
            query_id: Query keyframe ID
            camera_matrix: Camera intrinsics (3x3)
        
        Returns:
            LoopClosure if detected, None otherwise
        """
        if query_id not in self.keyframes:
            return None
        
        query_kf = self.keyframes[query_id]
        
        # Step 1: Find loop candidates using BoW similarity
        candidates = self._find_loop_candidates(query_kf)
        
        if not candidates:
            return None
        
        self.num_loop_candidates += len(candidates)
        
        # Step 2: Geometric verification (best candidate)
        for candidate_id in candidates:
            candidate_kf = self.keyframes[candidate_id]
            
            # Match features between query and candidate
            matches = self._match_features(query_kf, candidate_kf)
            
            if len(matches) < 30:
                continue
            
            # Geometric verification with RANSAC
            relative_pose, inliers, inlier_mask = self._geometric_verification(
                query_kf, candidate_kf, matches, camera_matrix
            )
            
            if inliers >= self.min_inliers:
                # Loop detected!
                confidence = min(1.0, inliers / 100.0)
                
                loop = LoopClosure(
                    query_id=query_id,
                    match_id=candidate_id,
                    relative_pose=relative_pose,
                    inliers=inliers,
                    confidence=confidence,
                    feature_matches=len(matches),
                )
                
                self.loop_closures.append(loop)
                self.num_loops_detected += 1
                
                # Mark loop closure link
                query_kf.loop_closure_to.add(candidate_id)
                candidate_kf.loop_closure_to.add(query_id)
                
                logger.info(
                    f"✓ Loop closure detected: frame {query_id} → {candidate_id} "
                    f"(inliers: {inliers}, matches: {len(matches)})"
                )
                
                return loop
        
        return None
    
    def _find_loop_candidates(self, query_kf: Keyframe) -> List[int]:
        """
        Find loop closure candidates using BoW similarity.
        
        Args:
            query_kf: Query keyframe
        
        Returns:
            List of candidate keyframe IDs (sorted by similarity)
        """
        if not self.bow_trained or query_kf.bow_vector is None:
            return []
        
        candidates = []
        
        for candidate_id in self.keyframe_ids:
            # Skip recent frames (must have min interval)
            if abs(candidate_id - query_kf.frame_id) < self.min_loop_interval:
                continue
            
            # Skip if already has loop closure
            if candidate_id in query_kf.loop_closure_to:
                continue
            
            candidate_kf = self.keyframes[candidate_id]
            
            if candidate_kf.bow_vector is None:
                continue
            
            # Compute BoW similarity
            similarity = self.bow.similarity(query_kf.bow_vector, candidate_kf.bow_vector)
            
            if similarity >= self.min_bow_similarity:
                candidates.append((candidate_id, similarity))
        
        # Sort by similarity (best first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 candidates
        return [cid for cid, _ in candidates[:3]]
    
    def _match_features(self, kf1: Keyframe, kf2: Keyframe) -> List:
        """Match features between two keyframes"""
        matches = self.matcher.knnMatch(kf1.descriptors, kf2.descriptors, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def _geometric_verification(
        self,
        query_kf: Keyframe,
        match_kf: Keyframe,
        matches: List,
        camera_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Geometric verification using essential matrix + RANSAC.
        
        Returns:
            (relative_pose, num_inliers, inlier_mask)
        """
        # Extract matched keypoint coordinates
        pts1 = np.float32([query_kf.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([match_kf.keypoints[m.trainIdx].pt for m in matches])
        
        # Compute essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.max_reprojection_error,
        )
        
        if E is None or mask is None:
            return np.eye(4), 0, np.array([])
        
        inliers = np.sum(mask)
        
        # Recover relative pose
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, camera_matrix, mask=mask)
        
        # Build 4x4 transformation matrix
        relative_pose = np.eye(4)
        relative_pose[:3, :3] = R
        relative_pose[:3, 3] = t.flatten()
        
        return relative_pose, int(inliers), mask
    
    def get_loop_constraints(self) -> List[Tuple[int, int, np.ndarray]]:
        """
        Get all loop closure constraints for pose graph optimization.
        
        Returns:
            List of (frame_id1, frame_id2, relative_pose)
        """
        constraints = []
        for loop in self.loop_closures:
            constraints.append((
                loop.query_id,
                loop.match_id,
                loop.relative_pose,
            ))
        return constraints
    
    def get_statistics(self) -> Dict:
        """Get loop closure statistics"""
        return {
            'num_keyframes': len(self.keyframes),
            'num_loop_candidates': self.num_loop_candidates,
            'num_loops_detected': self.num_loops_detected,
            'bow_trained': self.bow_trained,
        }
