#!/usr/bin/env python3
"""
3D Spatial Map Builder
======================

Creates a dense 3D spatial map from depth streams:
- Backprojects depth to 3D point cloud
- Accumulates points across frames with temporal stability
- Builds occupancy/voxel grid
- Tracks object 3D positions
- Enables spatial queries (distance, direction, occlusion)

This is the core of egocentric spatial understanding!

Usage:
    from orion.perception.spatial_map_builder import SpatialMapBuilder
    
    spatial_map = SpatialMapBuilder(
        image_width=1080,
        image_height=1920,
        intrinsics={'fx': 847.6, 'fy': 847.6, 'cx': 540, 'cy': 960}
    )
    
    # Each frame
    point_cloud = spatial_map.add_frame(depth_map, camera_pose, confidence)
    
    # Query the map
    objects_near = spatial_map.query_distance(max_distance=2.0)
    objects_left = spatial_map.query_direction(angle_range=(-90, 0))
    grid = spatial_map.get_voxel_grid(resolution=0.1)  # 10cm voxels

Author: Orion Research
Date: November 11, 2025
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import threading


@dataclass
class Point3D:
    """3D point with metadata"""
    x: float
    y: float
    z: float
    confidence: float = 1.0
    color: Tuple[int, int, int] = (255, 255, 255)
    frame_id: int = 0
    age: int = 1  # How many frames has this point existed
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def distance(self) -> float:
        """Distance from camera origin"""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def angle_horizontal(self) -> float:
        """Horizontal angle from camera (azimuth) in degrees"""
        return np.degrees(np.arctan2(self.x, self.z))
    
    def angle_vertical(self) -> float:
        """Vertical angle from camera (elevation) in degrees"""
        return np.degrees(np.arctan2(self.y, self.z))


@dataclass
class VoxelGrid:
    """3D occupancy grid"""
    resolution: float  # Voxel size in meters
    origin: np.ndarray  # World origin (0, 0, 0) is camera
    grid: np.ndarray  # 3D occupancy array
    confidence_grid: np.ndarray  # 3D confidence array
    
    def point_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert world coords to voxel indices"""
        vox_x = int((x - self.origin[0]) / self.resolution)
        vox_y = int((y - self.origin[1]) / self.resolution)
        vox_z = int((z - self.origin[2]) / self.resolution)
        return (vox_x, vox_y, vox_z)
    
    def voxel_to_point(self, vox_x: int, vox_y: int, vox_z: int) -> Tuple[float, float, float]:
        """Convert voxel indices to world coords (center of voxel)"""
        x = self.origin[0] + (vox_x + 0.5) * self.resolution
        y = self.origin[1] + (vox_y + 0.5) * self.resolution
        z = self.origin[2] + (vox_z + 0.5) * self.resolution
        return (x, y, z)
    
    def query_sphere(self, cx: float, cy: float, cz: float, radius: float) -> np.ndarray:
        """Get all voxels within sphere"""
        min_vox = self.point_to_voxel(cx - radius, cy - radius, cz - radius)
        max_vox = self.point_to_voxel(cx + radius, cy + radius, cz + radius)
        
        occupied = []
        for vox_x in range(max(0, min_vox[0]), min(self.grid.shape[0], max_vox[0] + 1)):
            for vox_y in range(max(0, min_vox[1]), min(self.grid.shape[1], max_vox[1] + 1)):
                for vox_z in range(max(0, min_vox[2]), min(self.grid.shape[2], max_vox[2] + 1)):
                    if self.grid[vox_x, vox_y, vox_z] > 0.5:
                        x, y, z = self.voxel_to_point(vox_x, vox_y, vox_z)
                        dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
                        if dist <= radius:
                            occupied.append((x, y, z, self.confidence_grid[vox_x, vox_y, vox_z]))
        
        return np.array(occupied) if occupied else np.empty((0, 4))


class SpatialMapBuilder:
    """
    Builds 3D spatial understanding from depth streams
    
    Key features:
    - Dense point cloud accumulation
    - Temporal stability filtering
    - Voxel occupancy grid
    - Spatial queries
    - Object tracking in 3D space
    """
    
    def __init__(self,
                 image_width: int,
                 image_height: int,
                 intrinsics: Dict[str, float],
                 max_points: int = 100000,
                 grid_size: float = 10.0,
                 grid_resolution: float = 0.05):
        """
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            intrinsics: Camera intrinsics dict with fx, fy, cx, cy
            max_points: Maximum points to keep in map (for memory)
            grid_size: Size of voxel grid in meters (will be -grid_size to +grid_size)
            grid_resolution: Voxel size in meters (0.05 = 5cm voxels)
        """
        self.width = image_width
        self.height = image_height
        
        # Camera intrinsics
        self.fx = intrinsics.get('fx', 500)
        self.fy = intrinsics.get('fy', 500)
        self.cx = intrinsics.get('cx', image_width / 2)
        self.cy = intrinsics.get('cy', image_height / 2)
        
        print(f"[SpatialMapBuilder] Initialized: {image_width}x{image_height}")
        print(f"  Intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
        print(f"  Grid: {grid_size}m x {grid_resolution}m voxels")
        
        # Point cloud
        self.point_cloud: List[Point3D] = []
        self.max_points = max_points
        
        # Voxel grid
        grid_dims = int(grid_size * 2 / grid_resolution)  # e.g., 20m / 0.05m = 400x400x400
        print(f"  Voxel grid: {grid_dims}x{grid_dims}x{grid_dims} = {grid_dims**3 / 1e6:.1f}M voxels")
        
        self.grid = VoxelGrid(
            resolution=grid_resolution,
            origin=np.array([-grid_size, -grid_size, 0]),
            grid=np.zeros((grid_dims, grid_dims, grid_dims), dtype=np.float32),
            confidence_grid=np.zeros((grid_dims, grid_dims, grid_dims), dtype=np.float32)
        )
        
        # Tracking
        self.frame_count = 0
        self.camera_poses: deque = deque(maxlen=100)  # Last 100 poses
        self.depth_history: deque = deque(maxlen=10)   # Last 10 depth frames
        
    def add_frame(self,
                 depth_map: np.ndarray,
                 camera_pose: Optional[np.ndarray] = None,
                 confidence_map: Optional[np.ndarray] = None,
                 rgb_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Add a depth frame to the spatial map
        
        Args:
            depth_map: (H, W) depth in meters
            camera_pose: (4, 4) transformation matrix or None (identity)
            confidence_map: (H, W) confidence [0, 1] or None (use 1.0)
            rgb_frame: (H, W, 3) for color per point
        
        Returns:
            points_added: (N, 3) array of new points added this frame
        """
        self.frame_count += 1
        
        # Default pose (identity - camera at origin)
        if camera_pose is None:
            camera_pose = np.eye(4)
        
        # Default confidence
        if confidence_map is None:
            confidence_map = np.ones_like(depth_map)
        
        self.camera_poses.append(camera_pose.copy())
        self.depth_history.append(depth_map.copy())
        
        # Backproject depth to 3D points
        points_3d = self._backproject_depth(depth_map, camera_pose, confidence_map, rgb_frame)
        
        if len(points_3d) == 0:
            return np.empty((0, 3))
        
        # Add to point cloud
        self.point_cloud.extend(points_3d)
        
        # Update voxel grid
        self._update_voxel_grid(points_3d)
        
        # Manage point cloud size
        if len(self.point_cloud) > self.max_points:
            self._prune_point_cloud()
        
        points_array = np.array([[p.x, p.y, p.z] for p in points_3d])
        
        return points_array
    
    def _backproject_depth(self,
                          depth_map: np.ndarray,
                          camera_pose: np.ndarray,
                          confidence_map: np.ndarray,
                          rgb_frame: Optional[np.ndarray]) -> List[Point3D]:
        """
        Backproject depth map to 3D points
        
        Creates a dense point cloud by converting each pixel to 3D
        """
        h, w = depth_map.shape
        points = []
        
        # Create coordinate grids
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Normalize image coordinates
        x_norm = (xx - self.cx) / self.fx
        y_norm = (yy - self.cy) / self.fy
        
        # Back-project: P = depth * [x_norm, y_norm, 1]
        z = depth_map
        x = x_norm * z
        y = y_norm * z
        
        # Validity mask
        valid = (z > 0.1) & (z < 20.0) & (confidence_map > 0.1)
        
        if valid.sum() == 0:
            return points
        
        # Extract valid points
        x_valid = x[valid]
        y_valid = y[valid]
        z_valid = z[valid]
        conf_valid = confidence_map[valid]
        
        # Transform by camera pose (if not identity)
        if not np.allclose(camera_pose, np.eye(4)):
            # Apply rotation + translation
            R = camera_pose[:3, :3]
            t = camera_pose[:3, 3]
            
            points_cam = np.stack([x_valid, y_valid, z_valid], axis=1)
            points_world = (R @ points_cam.T).T + t
            
            x_valid = points_world[:, 0]
            y_valid = points_world[:, 1]
            z_valid = points_world[:, 2]
        
        # Extract colors (sample every Nth point for speed)
        stride = max(1, int(np.sqrt(valid.sum() / 1000)))  # Target ~1k points
        
        y_idx, x_idx = np.where(valid)
        for i in range(0, len(y_idx), stride):
            yi, xi = y_idx[i], x_idx[i]
            
            # Color
            if rgb_frame is not None:
                color = tuple(int(c) for c in rgb_frame[yi, xi])
            else:
                color = (255, 255, 255)
            
            point = Point3D(
                x=float(x_valid[i]),
                y=float(y_valid[i]),
                z=float(z_valid[i]),
                confidence=float(conf_valid[i]),
                color=color,
                frame_id=self.frame_count,
                age=1
            )
            points.append(point)
        
        return points
    
    def _update_voxel_grid(self, points: List[Point3D]):
        """Update occupancy grid with new points"""
        for point in points:
            try:
                vox_x, vox_y, vox_z = self.grid.point_to_voxel(point.x, point.y, point.z)
                
                # Check bounds
                if (0 <= vox_x < self.grid.grid.shape[0] and
                    0 <= vox_y < self.grid.grid.shape[1] and
                    0 <= vox_z < self.grid.grid.shape[2]):
                    
                    # Update occupancy (average with confidence weighting)
                    old_conf = self.grid.confidence_grid[vox_x, vox_y, vox_z]
                    new_conf = point.confidence
                    
                    # Exponential moving average
                    alpha = 0.3
                    self.grid.confidence_grid[vox_x, vox_y, vox_z] = (
                        (1 - alpha) * old_conf + alpha * new_conf
                    )
                    
                    if self.grid.confidence_grid[vox_x, vox_y, vox_z] > 0.5:
                        self.grid.grid[vox_x, vox_y, vox_z] = 1.0
            except:
                pass
    
    def _prune_point_cloud(self):
        """Keep only high-confidence, recent points"""
        # Sort by (confidence * recency)
        scores = [(p.confidence * (1 + self.frame_count - p.frame_id)) for p in self.point_cloud]
        indices = np.argsort(scores)[::-1]  # Sort descending
        
        # Keep top 50%
        keep_indices = set(indices[:len(self.point_cloud) // 2])
        self.point_cloud = [p for i, p in enumerate(self.point_cloud) if i in keep_indices]
    
    def query_distance(self, 
                      max_distance: float,
                      min_distance: float = 0.0,
                      only_confident: bool = True) -> List[Point3D]:
        """
        Query points within distance range from camera
        
        Args:
            max_distance: Maximum distance in meters
            min_distance: Minimum distance in meters
            only_confident: Only return high-confidence points (>0.7)
        
        Returns:
            List of Point3D within range
        """
        results = []
        for p in self.point_cloud:
            if only_confident and p.confidence < 0.7:
                continue
            
            dist = p.distance()
            if min_distance <= dist <= max_distance:
                results.append(p)
        
        return results
    
    def query_direction(self,
                       azimuth_range: Tuple[float, float],
                       elevation_range: Optional[Tuple[float, float]] = None,
                       max_distance: float = 10.0) -> List[Point3D]:
        """
        Query points in a direction cone
        
        Args:
            azimuth_range: (min_degrees, max_degrees) horizontal angle
                          0° = forward, -90° = left, 90° = right
            elevation_range: (min_degrees, max_degrees) vertical angle
                            0° = horizontal, 90° = up, -90° = down
            max_distance: Maximum distance in meters
        
        Returns:
            List of Point3D in the specified direction
        """
        results = []
        for p in self.point_cloud:
            if p.distance() > max_distance:
                continue
            
            az = p.angle_horizontal()
            
            # Check azimuth
            if azimuth_range[0] <= az <= azimuth_range[1]:
                if elevation_range is None:
                    results.append(p)
                else:
                    # Check elevation
                    el = p.angle_vertical()
                    if elevation_range[0] <= el <= elevation_range[1]:
                        results.append(p)
        
        return results
    
    def query_plane(self,
                   normal: np.ndarray,
                   distance: float,
                   tolerance: float = 0.1) -> List[Point3D]:
        """
        Query points on a plane
        
        Args:
            normal: Plane normal vector (should be normalized)
            distance: Distance from origin
            tolerance: Distance tolerance in meters
        
        Returns:
            Points on the plane
        """
        results = []
        normal = normal / np.linalg.norm(normal)
        
        for p in self.point_cloud:
            point_dist = abs(np.dot(normal, np.array([p.x, p.y, p.z])) - distance)
            if point_dist <= tolerance:
                results.append(p)
        
        return results
    
    def get_voxel_grid(self) -> VoxelGrid:
        """Get the current voxel grid"""
        return self.grid
    
    def get_statistics(self) -> Dict:
        """Get map statistics"""
        if not self.point_cloud:
            return {'points': 0, 'frames': self.frame_count}
        
        points_array = np.array([[p.x, p.y, p.z] for p in self.point_cloud])
        distances = np.linalg.norm(points_array, axis=1)
        
        return {
            'points': len(self.point_cloud),
            'frames': self.frame_count,
            'distance_min': float(distances.min()),
            'distance_max': float(distances.max()),
            'distance_mean': float(distances.mean()),
            'distance_std': float(distances.std()),
            'coverage': {
                'close': sum(1 for d in distances if d < 1.0),
                'mid': sum(1 for d in distances if 1.0 <= d < 3.0),
                'far': sum(1 for d in distances if d >= 3.0)
            },
            'voxels_occupied': int((self.grid.grid > 0.5).sum())
        }
    
    def visualize_point_cloud(self, 
                             max_points: int = 10000,
                             background: Tuple[int, int, int] = (32, 32, 32)) -> np.ndarray:
        """
        Create a top-down view of the spatial map
        
        Args:
            max_points: Maximum points to render (for speed)
            background: Background color RGB
        
        Returns:
            (480, 640, 3) visualization image
        """
        img = np.full((480, 640, 3), background, dtype=np.uint8)
        
        # Determine scale (5m x 5m view)
        scale = 640 / 10.0  # pixels per meter
        
        # Sample points
        sample_indices = np.random.choice(
            len(self.point_cloud),
            min(max_points, len(self.point_cloud)),
            replace=False
        )
        
        for idx in sample_indices:
            p = self.point_cloud[idx]
            
            # Project to image (top-down)
            px = int(320 + p.x * scale)  # Center at 320
            py = int(240 + p.z * scale)  # Center at 240 (z is depth)
            
            if 0 <= px < 640 and 0 <= py < 480:
                # Use confidence for brightness
                brightness = int(p.confidence * 255)
                color = tuple(int(c * brightness / 255) for c in p.color)
                cv2.circle(img, (px, py), 1, color, -1)
        
        # Draw camera at origin
        cv2.circle(img, (320, 240), 5, (0, 255, 0), -1)
        
        # Add text
        cv2.putText(img, f"Points: {len(self.point_cloud)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Voxels: {int((self.grid.grid > 0.5).sum())}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
