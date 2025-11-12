#!/usr/bin/env python3
"""
Real-Time Adaptive Video Understanding Pipeline
================================================

Dynamic frame skipping based on:
- System performance (1-30 FPS adaptive)
- Scene complexity (static vs high motion)
- Visual caching (skip redundant frames)
- Resource budgets per component

Optimized for mobile deployment with graceful degradation.

Author: Orion Research  
Date: November 11, 2025
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import cv2


@dataclass
class PerformanceMetrics:
    """Real-time performance tracking"""
    fps: float = 0.0
    frame_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    depth_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    slam_time_ms: float = 0.0
    
    # Scene complexity
    motion_score: float = 0.0  # 0-1, higher = more motion
    n_objects: int = 0
    
    # Adaptive state
    frame_skip: int = 1  # Process every Nth frame
    use_fast_mode: bool = False


class AdaptiveFrameSelector:
    """
    Intelligently selects which frames to process based on:
    - Scene motion (skip static frames)
    - System load (adaptive FPS)
    - Visual change (cache similar frames)
    
    Real-time: ~1-2ms overhead
    """
    
    def __init__(self,
                 target_fps: float = 10.0,
                 motion_threshold: float = 0.05,
                 similarity_threshold: float = 0.95):
        """
        Args:
            target_fps: Target processing rate
            motion_threshold: Min motion to process frame
            similarity_threshold: Skip frames above this similarity
        """
        self.target_fps = target_fps
        self.motion_threshold = motion_threshold
        self.similarity_threshold = similarity_threshold
        
        # State
        self.prev_frame_gray = None
        self.prev_frame_time = 0
        self.frame_times = deque(maxlen=30)
        self.motion_history = deque(maxlen=10)
        
        # Adaptive parameters
        self.current_fps = target_fps
        self.frame_skip = 1
        
    def should_process(self,
                      frame: np.ndarray,
                      force: bool = False) -> Tuple[bool, Dict]:
        """
        Decide if frame should be processed
        
        Args:
            frame: Current frame (H, W, 3)
            force: Force processing (keyframe)
        
        Returns:
            (should_process, stats)
        """
        current_time = time.time()
        
        if force:
            self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_frame_time = current_time
            return True, {'reason': 'forced', 'motion_score': 1.0}
        
        # Check time budget
        if self.prev_frame_time > 0:
            elapsed = current_time - self.prev_frame_time
            if elapsed < (1.0 / self.target_fps):
                return False, {'reason': 'time_budget', 'elapsed': elapsed}
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            self.prev_frame_time = current_time
            return True, {'reason': 'first_frame', 'motion_score': 0.0}
        
        # Compute motion score (fast optical flow approximation)
        motion_score = self._compute_motion(gray, self.prev_frame_gray)
        self.motion_history.append(motion_score)
        
        # Check if scene is static
        if motion_score < self.motion_threshold:
            # Static scene - check visual similarity
            similarity = self._compute_similarity(gray, self.prev_frame_gray)
            
            if similarity > self.similarity_threshold:
                return False, {
                    'reason': 'static_scene',
                    'motion_score': motion_score,
                    'similarity': similarity
                }
        
        # Process this frame
        self.prev_frame_gray = gray
        self.prev_frame_time = current_time
        
        return True, {
            'reason': 'process',
            'motion_score': motion_score,
            'avg_motion': np.mean(list(self.motion_history)) if self.motion_history else 0.0
        }
    
    def _compute_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Fast motion estimation using frame differencing
        
        Returns motion score in [0, 1]
        """
        # Downsample for speed
        h, w = frame1.shape
        frame1_small = cv2.resize(frame1, (w//4, h//4))
        frame2_small = cv2.resize(frame2, (w//4, h//4))
        
        # Compute absolute difference
        diff = cv2.absdiff(frame1_small, frame2_small)
        
        # Threshold and count changed pixels
        _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        motion_pixels = thresh.sum() / 255
        total_pixels = thresh.size
        
        motion_score = motion_pixels / total_pixels
        
        return min(1.0, motion_score * 10)  # Scale to [0, 1]
    
    def _compute_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Fast similarity using normalized cross-correlation
        
        Returns similarity in [0, 1]
        """
        # Downsample
        h, w = frame1.shape
        f1 = cv2.resize(frame1, (w//8, h//8)).astype(np.float32)
        f2 = cv2.resize(frame2, (w//8, h//8)).astype(np.float32)
        
        # Normalize
        f1 = (f1 - f1.mean()) / (f1.std() + 1e-8)
        f2 = (f2 - f2.mean()) / (f2.std() + 1e-8)
        
        # Correlation
        correlation = (f1 * f2).mean()
        
        return (correlation + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
    
    def update_performance(self, frame_time_ms: float):
        """Update FPS based on actual performance"""
        self.frame_times.append(frame_time_ms)
        
        if len(self.frame_times) >= 5:
            avg_time = np.mean(list(self.frame_times))
            self.current_fps = 1000.0 / avg_time
            
            # Adjust target FPS if falling behind
            if self.current_fps < self.target_fps * 0.7:
                # Slow down target
                self.target_fps = max(1.0, self.target_fps * 0.9)
            elif self.current_fps > self.target_fps * 1.2:
                # Speed up target (system can handle more)
                self.target_fps = min(30.0, self.target_fps * 1.1)


class DynamicResourceBudget:
    """
    Allocates time budget across pipeline components
    
    Priorities:
    1. Detection (always)
    2. Tracking (high priority)
    3. Depth (medium)
    4. SLAM (low - skip if needed)
    """
    
    def __init__(self, target_frame_time_ms: float = 100.0):
        """
        Args:
            target_frame_time_ms: Target time per frame (e.g., 100ms = 10 FPS)
        """
        self.target_time = target_frame_time_ms
        self.component_times = {}
        self.budget_allocation = {
            'detection': 0.35,  # 35% of budget
            'depth': 0.25,      # 25%
            'tracking': 0.20,   # 20%
            'slam': 0.15,       # 15%
            'other': 0.05       # 5% overhead
        }
        
    def should_run_component(self, 
                            component: str,
                            elapsed_ms: float) -> bool:
        """
        Check if component should run given time budget
        
        Args:
            component: Component name
            elapsed_ms: Time already spent this frame
        
        Returns:
            True if component should run
        """
        # Always run critical components
        if component in ['detection', 'tracking']:
            return True
        
        # Check remaining budget
        remaining = self.target_time - elapsed_ms
        component_budget = self.target_time * self.budget_allocation.get(component, 0.05)
        
        return remaining > component_budget
    
    def record_time(self, component: str, time_ms: float):
        """Record component execution time"""
        if component not in self.component_times:
            self.component_times[component] = deque(maxlen=30)
        
        self.component_times[component].append(time_ms)
    
    def get_avg_time(self, component: str) -> float:
        """Get average time for component"""
        if component not in self.component_times:
            return 0.0
        
        return np.mean(list(self.component_times[component]))
    
    def adjust_budget(self):
        """Dynamically adjust budget based on actual usage"""
        total_avg = sum(self.get_avg_time(c) for c in self.budget_allocation.keys())
        
        if total_avg > self.target_time * 1.2:
            # Over budget - reduce non-critical components
            self.budget_allocation['slam'] *= 0.8
            self.budget_allocation['depth'] *= 0.9
            self.budget_allocation['tracking'] = min(0.25, self.budget_allocation['tracking'] * 1.1)
        elif total_avg < self.target_time * 0.7:
            # Under budget - can increase quality
            self.budget_allocation['slam'] = min(0.20, self.budget_allocation['slam'] * 1.2)
            self.budget_allocation['depth'] = min(0.30, self.budget_allocation['depth'] * 1.1)


class VisualCache:
    """
    Caches visual features for static scenes
    Avoids re-computing on similar frames
    
    Real-time: ~0.5ms lookup
    """
    
    def __init__(self, cache_size: int = 100, similarity_threshold: float = 0.90):
        """
        Args:
            cache_size: Max cached items
            similarity_threshold: Min similarity to use cache
        """
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        
        # Cache: frame_hash -> cached_data
        self.cache = {}
        self.access_times = {}
        self.frame_hashes = deque(maxlen=cache_size)
        
    def get(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Get cached data for frame
        
        Args:
            frame: Current frame
        
        Returns:
            Cached data if available and similar enough
        """
        # Compute perceptual hash (fast)
        frame_hash = self._hash_frame(frame)
        
        if frame_hash in self.cache:
            # Update access time
            self.access_times[frame_hash] = time.time()
            return self.cache[frame_hash]
        
        # Check if similar to recent frames
        for cached_hash in reversed(list(self.frame_hashes)[-10:]):
            if self._hashes_similar(frame_hash, cached_hash):
                # Reuse cached data
                self.access_times[cached_hash] = time.time()
                return self.cache[cached_hash]
        
        return None
    
    def put(self, frame: np.ndarray, data: Dict):
        """
        Cache data for frame
        
        Args:
            frame: Frame to cache
            data: Associated data (detections, depth, etc.)
        """
        frame_hash = self._hash_frame(frame)
        
        # Evict oldest if full
        if len(self.cache) >= self.cache_size:
            oldest_hash = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_hash]
            del self.access_times[oldest_hash]
        
        self.cache[frame_hash] = data
        self.access_times[frame_hash] = time.time()
        self.frame_hashes.append(frame_hash)
    
    def _hash_frame(self, frame: np.ndarray) -> int:
        """Compute perceptual hash of frame"""
        # Downsample to 8x8
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        
        # DCT
        dct = cv2.dct(np.float32(small))
        
        # Use top-left 8x8 for hash
        dct_low = dct[:8, :8]
        
        # Hash
        median = np.median(dct_low)
        hash_bits = (dct_low > median).flatten()
        
        # Convert to int
        hash_int = int(''.join(['1' if b else '0' for b in hash_bits]), 2)
        
        return hash_int
    
    def _hashes_similar(self, hash1: int, hash2: int, threshold: int = 8) -> bool:
        """Check if two hashes are similar (Hamming distance)"""
        xor = hash1 ^ hash2
        hamming_distance = bin(xor).count('1')
        return hamming_distance < threshold


def get_performance_stats(metrics: PerformanceMetrics) -> str:
    """Format performance metrics for display"""
    return (
        f"FPS: {metrics.fps:.1f} | "
        f"Frame: {metrics.frame_time_ms:.0f}ms | "
        f"Detect: {metrics.detection_time_ms:.0f}ms | "
        f"Depth: {metrics.depth_time_ms:.0f}ms | "
        f"Track: {metrics.tracking_time_ms:.0f}ms | "
        f"Motion: {metrics.motion_score:.2f} | "
        f"Obj: {metrics.n_objects} | "
        f"Skip: {metrics.frame_skip}"
    )
