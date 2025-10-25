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
from typing import List

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
            logger.warning("No state changes to window")
            return []
        
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
                
                if (time_gap <= self.config.max_gap_between_changes and
                    window_duration <= self.config.max_duration_seconds and
                    len(current_window.state_changes) < self.config.max_changes_per_window):
                    # Add to current window
                    current_window.add_state_change(change)
                else:
                    # Close current window and start new one
                    if current_window.is_significant:
                        windows.append(current_window)
                    
                    current_window = TemporalWindow(
                        start_time=change.timestamp_before,
                        end_time=change.timestamp_after,
                    )
                    current_window.add_state_change(change)
        
        # Add final window
        if current_window and current_window.is_significant:
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
