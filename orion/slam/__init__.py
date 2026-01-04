"""
SLAM Module
===========

Visual odometry and SLAM for camera tracking and 3D mapping.

Components:
- SLAMEngine: Main interface for camera pose estimation
- VisualOdometry: ORB-based feature tracking and pose estimation  
- SLAMConfig: Configuration for SLAM parameters
- Keyframe: Stored keyframes with features and poses
- MapPoint: 3D map points with descriptors
"""

from .visual_odometry import (
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