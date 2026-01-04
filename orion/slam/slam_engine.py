"""
SLAM Engine
===========

Module for Simultaneous Localization and Mapping (SLAM) or Visual Odometry.
Responsible for estimating camera pose and trajectory.

This module now imports from visual_odometry.py which contains the full
ORB-based visual odometry implementation with depth integration.

Author: Orion Research Team
Date: November 2025 (Updated January 2026)
"""

# Re-export from new implementation for backward compatibility
from orion.slam.visual_odometry import (
    SLAMConfig,
    SLAMEngine,
    VisualOdometry,
    Keyframe,
    MapPoint,
)

__all__ = [
    "SLAMConfig",
    "SLAMEngine", 
    "VisualOdometry",
    "Keyframe",
    "MapPoint",
]
