"""
Pose Graph Optimization
=======================

Optimizes camera trajectory given loop closure constraints.

Uses iterative least-squares optimization to minimize pose errors
while respecting odometry and loop closure constraints.

Author: Orion Research Team
Date: November 2025
"""

import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class PoseGraphOptimizer:
    """
    Simple pose graph optimization using Gauss-Newton.
    
    Optimizes poses to minimize:
    - Odometry errors (sequential pose differences)
    - Loop closure errors (pose differences at loop closures)
    """
    
    def __init__(
        self,
        odometry_weight: float = 1.0,
        loop_closure_weight: float = 100.0,  # Higher weight for loop closures
        max_iterations: int = 20,
        convergence_threshold: float = 1e-4,
    ):
        """
        Initialize pose graph optimizer.
        
        Args:
            odometry_weight: Weight for odometry edges
            loop_closure_weight: Weight for loop closure edges
            max_iterations: Max optimization iterations
            convergence_threshold: Convergence threshold for cost
        """
        self.odometry_weight = odometry_weight
        self.loop_weight = loop_closure_weight
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def optimize(
        self,
        poses: List[np.ndarray],  # List of 4x4 poses
        odometry_edges: List[Tuple[int, int, np.ndarray]],  # (i, j, relative_pose)
        loop_edges: List[Tuple[int, int, np.ndarray]],  # (i, j, relative_pose)
        fix_first_pose: bool = True,
    ) -> List[np.ndarray]:
        """
        Optimize poses given odometry and loop closure constraints.
        
        Args:
            poses: Initial poses (4x4 matrices)
            odometry_edges: Sequential pose constraints
            loop_edges: Loop closure constraints
            fix_first_pose: Keep first pose fixed (anchor)
        
        Returns:
            Optimized poses
        """
        if len(poses) == 0:
            return poses
        
        logger.info(f"Pose graph optimization:")
        logger.info(f"  Poses: {len(poses)}")
        logger.info(f"  Odometry edges: {len(odometry_edges)}")
        logger.info(f"  Loop edges: {len(loop_edges)}")
        
        # Convert poses to optimization parameters (x, y, z, yaw)
        # Simplified 2D+height+rotation optimization
        params = self._poses_to_params(poses)
        
        # Initial cost
        cost = self._compute_cost(params, odometry_edges, loop_edges)
        logger.info(f"  Initial cost: {cost:.6f}")
        
        # Gauss-Newton optimization
        for iteration in range(self.max_iterations):
            # Compute gradient and Hessian
            gradient, hessian = self._compute_gradient_hessian(
                params, odometry_edges, loop_edges
            )
            
            # Fix first pose (set gradient to zero)
            if fix_first_pose:
                gradient[0] = 0.0
                hessian[0, :] = 0.0
                hessian[:, 0] = 0.0
                hessian[0, 0] = 1.0
            
            # Solve: H * delta = -g
            try:
                delta = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                logger.warning("Singular Hessian, stopping optimization")
                break
            
            # Update parameters
            params = params + delta
            
            # Compute new cost
            new_cost = self._compute_cost(params, odometry_edges, loop_edges)
            cost_change = abs(cost - new_cost)
            
            if iteration % 5 == 0:
                logger.debug(f"  Iter {iteration}: cost={new_cost:.6f}, change={cost_change:.6e}")
            
            # Check convergence
            if cost_change < self.convergence_threshold:
                logger.info(f"  Converged at iteration {iteration}")
                break
            
            cost = new_cost
        
        logger.info(f"  Final cost: {cost:.6f}")
        
        # Convert parameters back to poses
        optimized_poses = self._params_to_poses(params, poses)
        
        # Compute correction statistics
        corrections = [
            np.linalg.norm(optimized_poses[i][:3, 3] - poses[i][:3, 3])
            for i in range(len(poses))
        ]
        avg_correction = np.mean(corrections)
        max_correction = np.max(corrections)
        
        logger.info(f"  Avg correction: {avg_correction:.1f}mm")
        logger.info(f"  Max correction: {max_correction:.1f}mm")
        
        return optimized_poses
    
    def _poses_to_params(self, poses: List[np.ndarray]) -> np.ndarray:
        """
        Convert 4x4 poses to optimization parameters.
        
        Parameters: [x0, y0, z0, yaw0, x1, y1, z1, yaw1, ...]
        
        Simplified: only optimize x, y, z, yaw (no pitch/roll)
        """
        params = []
        for pose in poses:
            # Translation
            x, y, z = pose[:3, 3]
            
            # Rotation (extract yaw from rotation matrix)
            R = pose[:3, :3]
            yaw = np.arctan2(R[1, 0], R[0, 0])
            
            params.extend([x, y, z, yaw])
        
        return np.array(params)
    
    def _params_to_poses(self, params: np.ndarray, original_poses: List[np.ndarray]) -> List[np.ndarray]:
        """Convert optimization parameters back to 4x4 poses"""
        poses = []
        
        for i in range(len(original_poses)):
            idx = i * 4
            x, y, z, yaw = params[idx:idx+4]
            
            # Build 4x4 pose
            pose = np.eye(4)
            
            # Rotation (yaw only, simplified)
            c, s = np.cos(yaw), np.sin(yaw)
            pose[:3, :3] = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
            
            # Translation
            pose[:3, 3] = [x, y, z]
            
            poses.append(pose)
        
        return poses
    
    def _compute_cost(
        self,
        params: np.ndarray,
        odometry_edges: List[Tuple[int, int, np.ndarray]],
        loop_edges: List[Tuple[int, int, np.ndarray]],
    ) -> float:
        """Compute total cost (sum of squared errors)"""
        cost = 0.0
        
        # Odometry cost
        for i, j, relative_pose in odometry_edges:
            error = self._compute_edge_error(params, i, j, relative_pose)
            cost += self.odometry_weight * np.sum(error ** 2)
        
        # Loop closure cost
        for i, j, relative_pose in loop_edges:
            error = self._compute_edge_error(params, i, j, relative_pose)
            cost += self.loop_weight * np.sum(error ** 2)
        
        return cost
    
    def _compute_edge_error(
        self,
        params: np.ndarray,
        i: int,
        j: int,
        relative_pose: np.ndarray,
    ) -> np.ndarray:
        """
        Compute error for edge (i, j).
        
        Error = measured_relative - predicted_relative
        """
        # Extract poses from parameters
        idx_i = i * 4
        idx_j = j * 4
        
        x_i, y_i, z_i, yaw_i = params[idx_i:idx_i+4]
        x_j, y_j, z_j, yaw_j = params[idx_j:idx_j+4]
        
        # Predicted relative transform (j in i's frame)
        # Simplified: just difference in position and angle
        dx = x_j - x_i
        dy = y_j - y_i
        dz = z_j - z_i
        dyaw = yaw_j - yaw_i
        
        # Measured relative transform
        rel_x, rel_y, rel_z = relative_pose[:3, 3]
        R_rel = relative_pose[:3, :3]
        rel_yaw = np.arctan2(R_rel[1, 0], R_rel[0, 0])
        
        # Error vector [dx, dy, dz, dyaw]
        error = np.array([
            dx - rel_x,
            dy - rel_y,
            dz - rel_z,
            self._angle_diff(dyaw, rel_yaw),
        ])
        
        return error
    
    def _angle_diff(self, a1: float, a2: float) -> float:
        """Compute angle difference in [-pi, pi]"""
        diff = a1 - a2
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
    
    def _compute_gradient_hessian(
        self,
        params: np.ndarray,
        odometry_edges: List[Tuple[int, int, np.ndarray]],
        loop_edges: List[Tuple[int, int, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient and Hessian using finite differences.
        
        Simplified approximation for speed.
        """
        n = len(params)
        gradient = np.zeros(n)
        hessian = np.eye(n) * 1e-6  # Small regularization
        
        # Finite difference epsilon
        eps = 1e-4
        
        # Compute gradient
        cost_0 = self._compute_cost(params, odometry_edges, loop_edges)
        
        for i in range(n):
            params_plus = params.copy()
            params_plus[i] += eps
            cost_plus = self._compute_cost(params_plus, odometry_edges, loop_edges)
            gradient[i] = (cost_plus - cost_0) / eps
        
        # Approximate Hessian (diagonal + neighbors)
        for i in range(n):
            # Diagonal
            params_plus = params.copy()
            params_plus[i] += eps
            cost_plus = self._compute_cost(params_plus, odometry_edges, loop_edges)
            
            params_minus = params.copy()
            params_minus[i] -= eps
            cost_minus = self._compute_cost(params_minus, odometry_edges, loop_edges)
            
            hessian[i, i] = (cost_plus - 2*cost_0 + cost_minus) / (eps**2)
        
        # Add off-diagonal coupling for edges (simplified)
        for i, j, _ in odometry_edges + loop_edges:
            idx_i = i * 4
            idx_j = j * 4
            
            # Couple x, y, z, yaw between connected poses
            for k in range(4):
                hessian[idx_i+k, idx_j+k] += 0.1
                hessian[idx_j+k, idx_i+k] += 0.1
        
        return gradient, hessian
