"""
SLAM Package - Visual SLAM for Camera Pose Estimation

Provides visual odometry and SLAM for transforming observations
to consistent world coordinates.

Author: Orion Research Team
Date: November 2025
"""

from orion.slam.slam_engine import SLAMEngine, OpenCVSLAM
from orion.slam.world_coordinate_tracker import WorldCoordinateTracker

__all__ = [
    'SLAMEngine',
    'OpenCVSLAM',
    'WorldCoordinateTracker',
]
