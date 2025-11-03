"""
Temporal Window Manager
=======================

Creates temporal windows for event composition.

Responsibilities:
- Group related state changes into temporal windows
- Apply adaptive windowing strategies
- Identify significant activity periods
- Prepare windows for event composition

Author: Orion Research Team
Date: October 2025
"""

import logging
from typing import List, Optional

from orion.semantic.types import StateChange, TemporalWindow
from orion.semantic.config import TemporalWindowConfig

logger = logging.getLogger(__name__)


class TemporalWindowManager:
    """
    Manages temporal windowing for event composition.
    
    Groups related state changes into windows that represent
    coherent events or activities.
    """
    
    def __init__(self, config: TemporalWindowConfig):
        """
        Initialize temporal window manager.
        
        Args:
            config: Temporal window configuration
        """
        self.config = config
        self.windows: List[TemporalWindow] = []
        
        logger.debug(
            f"TemporalWindowManager initialized: max_duration={config.max_duration_seconds}s, "
            f"max_gap={config.max_gap_between_changes}s"
        )
    
    def create_windows(
        self,
        state_changes: List[StateChange],
        *,
        total_duration: Optional[float] = None,
    ) -> List[TemporalWindow]:
        """
        Create temporal windows from state changes.
        
        Args:
            state_changes: List of detected state changes
            
        Returns:
            List of temporal windows
        """
        logger.info("="*80)
        logger.info("PHASE 2C: TEMPORAL WINDOWING")
        logger.info("="*80)
        
        if not state_changes:
            logger.warning("No state changes to window - creating time-based windows as fallback")
            return self._create_time_based_windows(total_duration)
        
        logger.info(f"Creating temporal windows from {len(state_changes)} state changes...")
        logger.info(f"  Max duration: {self.config.max_duration_seconds}s")
        logger.info(f"  Max gap: {self.config.max_gap_between_changes}s")
        
        # Sort by timestamp
        sorted_changes = sorted(state_changes, key=lambda c: c.timestamp_after)
        
        windows = []
        current_window = None
        
        for change in sorted_changes:
            if current_window is None:
                # Start new window
                current_window = TemporalWindow(
                    start_time=change.timestamp_before,
                    end_time=change.timestamp_after,
                )
                current_window.add_state_change(change)
            else:
                # Check if change fits in current window
                time_gap = change.timestamp_before - current_window.end_time
                window_duration = change.timestamp_after - current_window.start_time

                fits_window = (
                    time_gap <= self.config.max_gap_between_changes
                    and window_duration <= self.config.max_duration_seconds
                    and len(current_window.state_changes) < self.config.max_changes_per_window
                )

                # Allow fallback windows to stay open longer to gather context
                if not fits_window and current_window.fallback_generated and change.is_fallback:
                    fits_window = True

                if fits_window:
                    current_window.add_state_change(change)
                else:
                    if current_window.state_changes:
                        self._compute_window_significance(current_window)

                    if current_window.is_significant or current_window.fallback_generated:
                        if current_window.fallback_generated and current_window.significance_score < 0.5:
                            current_window.significance_score = max(current_window.significance_score, 0.6)
                        windows.append(current_window)
                    
                    current_window = TemporalWindow(
                        start_time=change.timestamp_before,
                        end_time=change.timestamp_after,
                    )
                    current_window.add_state_change(change)
        
        # Add final window
        if current_window:
            if current_window.state_changes:
                self._compute_window_significance(current_window)
            if current_window.is_significant or current_window.fallback_generated:
                if current_window.fallback_generated and current_window.significance_score < 0.5:
                    current_window.significance_score = max(current_window.significance_score, 0.6)
                windows.append(current_window)
        
        logger.info(f"\n✓ Created {len(windows)} temporal windows")
        
        if windows:
            logger.info("  Window statistics:")
            total_changes = sum(len(w.state_changes) for w in windows)
            avg_duration = sum(w.duration for w in windows) / len(windows)
            logger.info(f"    Total state changes in windows: {total_changes}")
            logger.info(f"    Average window duration: {avg_duration:.2f}s")
            logger.info(f"    Average changes per window: {total_changes / len(windows):.1f}")
        
        logger.info("="*80 + "\n")
        
        self.windows = windows
        return windows

    
    def _compute_window_significance(self, window: TemporalWindow) -> None:
        """
        Compute significance score for a temporal window.
        
        Significance is based on:
        - Number of state changes
        - Average change magnitude
        - Number of unique entities
        - Duration
        
        Args:
            window: Temporal window to score
        """
        if not window.state_changes:
            window.significance_score = 0.0
            return
        
        # Factor 1: Number of state changes (normalized, max at 10 changes)
        num_changes = len(window.state_changes)
        change_score = min(num_changes / 10.0, 1.0)
        
        # Factor 2: Average change magnitude
        avg_magnitude = sum(c.change_magnitude for c in window.state_changes) / num_changes
        magnitude_score = avg_magnitude  # Already 0-1
        
        # Factor 3: Entity diversity (more unique entities = more significant)
        num_entities = len(window.active_entities)
        entity_score = min(num_entities / 5.0, 1.0)  # Max at 5 entities
        
        # Factor 4: Duration (longer windows more significant, max at 10 seconds)
        duration_score = min(window.duration / 10.0, 1.0)
        
        # Weighted combination
        window.significance_score = (
            0.35 * change_score +      # Heavily weight number of changes
            0.35 * magnitude_score +   # Heavily weight magnitude
            0.20 * entity_score +      # Moderate weight on diversity
            0.10 * duration_score      # Light weight on duration
        )
        
        # Also compute average change magnitude for reference
        window.average_change_magnitude = avg_magnitude
    
    def get_window_at_time(self, timestamp: float) -> List[TemporalWindow]:
        """
        Get windows active at a given timestamp.
        
        Args:
            timestamp: Target timestamp
            
        Returns:
            List of windows overlapping timestamp
        """
        return [
            w for w in self.windows
            if w.start_time <= timestamp <= w.end_time
        ]
    
    def merge_overlapping_windows(self) -> List[TemporalWindow]:
        """
        Merge overlapping temporal windows.
        
        Returns:
            List of merged windows
        """
        if not self.windows:
            return []
        
        # Sort by start time
        sorted_windows = sorted(self.windows, key=lambda w: w.start_time)
        
        merged = []
        current = sorted_windows[0]
        
        for window in sorted_windows[1:]:
            if window.start_time <= current.end_time:
                # Overlapping - merge
                current.end_time = max(current.end_time, window.end_time)
                current.state_changes.extend(window.state_changes)
                current.active_entities.update(window.active_entities)
                current.causal_links.extend(window.causal_links)
            else:
                # No overlap - start new merged window
                merged.append(current)
                current = window
        
        # Add final window
        merged.append(current)
        
        return merged
    
    def _create_time_based_windows(
        self,
        total_duration: Optional[float] = None,
    ) -> List[TemporalWindow]:
        """
        Create temporal windows based on time alone (fallback when no state changes).
        
        This ensures the pipeline can continue even when state change detection fails.
        Windows are created at regular intervals across the video duration.
        
        Returns:
            List of time-based temporal windows
        """
        logger.info("  Creating time-based fallback windows...")

        duration = total_duration if total_duration and total_duration > 0 else self.config.max_duration_seconds
        if duration <= 0:
            logger.warning("  ↳ Unable to synthesize fallback window (no duration information)")
            self.windows = []
            return []

        window = TemporalWindow(
            start_time=0.0,
            end_time=duration,
        )
        window.fallback_generated = True
        window.significance_score = 0.6

        logger.info("  ✓ Created 1 fallback time-based window")
        logger.info("="*80 + "\n")

        self.windows = [window]
        return self.windows
