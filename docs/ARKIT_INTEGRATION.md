# ARKit Integration Guide for Orion SLAM

**Date**: January 2025  
**Goal**: Integrate iOS ARKit scene reconstruction with Orion SLAM for sensor fusion  
**Rerun Version**: 0.26.2 (latest)

---

## 🍎 Overview

ARKit provides:
- **6DOF Camera Tracking**: High-frequency pose estimation (60 Hz)
- **LiDAR Depth** (iPhone 12 Pro+): Accurate depth maps (up to 5m)
- **Plane Detection**: Real-world surface reconstruction
- **3D Mesh**: Scene mesh from SLAM + depth fusion
- **Visual Inertial Odometry (VIO)**: IMU + camera for robust tracking

**Orion SLAM** provides:
- **Object Detection**: YOLO11x for semantic understanding
- **Entity Tracking**: Persistent object identity with Re-ID
- **Spatial Zones**: 3D spatial reasoning and relationships
- **Scene Classification**: CLIP-based environment understanding

**Fusion Benefits**:
- ✅ ARKit pose + Orion semantics = Semantically-aware 3D reconstruction
- ✅ LiDAR depth + MiDaS depth = Higher accuracy depth fusion
- ✅ ARKit mesh + Orion entities = Object-mesh association
- ✅ Real-time performance (60 FPS ARKit → 2-5 FPS Orion processing)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      iOS DEVICE (ARKit)                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ARKit Session                                              │
│  ├─ Camera Frames (RGB) @ 60 FPS                            │
│  ├─ Depth Maps (LiDAR) @ 60 FPS                             │
│  ├─ Camera Poses (6DOF) @ 60 FPS                            │
│  ├─ Detected Planes                                         │
│  └─ Scene Mesh (vertices, triangles)                        │
│                                                              │
│  Network Streamer (Swift)                                   │
│  └─ Send RGB + Depth + Pose → Server                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
                  Network (WiFi/USB)
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                 ORION SERVER (Python/macOS)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ARKit Receiver                                             │
│  └─ Decode RGB + Depth + Pose                               │
│                                                              │
│  Orion SLAM Pipeline                                        │
│  ├─ YOLO Detection @ 2-5 FPS                                │
│  ├─ Entity Tracking (Re-ID)                                 │
│  ├─ Spatial Zone Construction                               │
│  ├─ Scene Classification                                    │
│  └─ Sensor Fusion:                                          │
│      • ARKit pose ⊕ SLAM pose → Fused pose                  │
│      • LiDAR depth ⊕ MiDaS depth → Fused depth              │
│      • ARKit mesh ⊕ Entities → Object-mesh links            │
│                                                              │
│  Rerun Logger                                               │
│  ├─ Log ARKit camera poses                                  │
│  ├─ Log depth (LiDAR + MiDaS)                               │
│  ├─ Log detected objects with 3D boxes                      │
│  ├─ Log scene mesh                                          │
│  └─ Log spatial zones                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
                  Rerun Viewer (3D Visualization)
```

---

## 📱 iOS ARKit Streaming App

### Swift Code (ARKit → Server)

```swift
import ARKit
import Network
import Compression

class ARKitStreamer: NSObject, ARSessionDelegate {
    var arSession: ARSession!
    var connection: NWConnection?
    
    override init() {
        super.init()
        setupARSession()
        connectToServer()
    }
    
    func setupARSession() {
        arSession = ARSession()
        arSession.delegate = self
        
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
        config.planeDetection = [.horizontal, .vertical]
        
        arSession.run(config)
    }
    
    func connectToServer() {
        // Connect to Orion server
        let host = NWEndpoint.Host("192.168.1.100")  // Mac IP
        let port = NWEndpoint.Port(integerLiteral: 8765)
        
        connection = NWConnection(host: host, port: port, using: .tcp)
        connection?.start(queue: .global())
    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let depthData = frame.sceneDepth else { return }
        
        // Extract data
        let rgb = frame.capturedImage
        let depth = depthData.depthMap
        let confidence = depthData.confidenceMap
        let pose = frame.camera.transform
        let intrinsics = frame.camera.intrinsics
        
        // Serialize and send
        let packet = createPacket(
            rgb: rgb,
            depth: depth,
            pose: pose,
            intrinsics: intrinsics,
            timestamp: frame.timestamp
        )
        
        sendPacket(packet)
    }
    
    func createPacket(
        rgb: CVPixelBuffer,
        depth: CVPixelBuffer,
        pose: simd_float4x4,
        intrinsics: simd_float3x3,
        timestamp: TimeInterval
    ) -> Data {
        // Convert to protobuf/msgpack/JSON
        var packet = [String: Any]()
        
        // RGB image (JPEG compressed)
        let ciImage = CIImage(cvPixelBuffer: rgb)
        let context = CIContext()
        let jpegData = context.jpegRepresentation(
            of: ciImage,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.8]
        )
        packet["rgb"] = jpegData?.base64EncodedString()
        
        // Depth map (16-bit PNG or raw)
        let depthData = depthToData(depth)
        packet["depth"] = depthData.base64EncodedString()
        
        // Camera pose (4x4 matrix)
        packet["pose"] = poseToArray(pose)
        
        // Intrinsics (3x3 matrix)
        packet["intrinsics"] = intrinsicsToArray(intrinsics)
        
        // Timestamp
        packet["timestamp"] = timestamp
        
        // Serialize to JSON
        let jsonData = try! JSONSerialization.data(withJSONObject: packet)
        return jsonData
    }
    
    func sendPacket(_ data: Data) {
        // Send size header + data
        var size = UInt32(data.count).bigEndian
        let sizeData = Data(bytes: &size, count: 4)
        
        connection?.send(content: sizeData + data, completion: .contentProcessed { error in
            if let error = error {
                print("Send error: \\(error)")
            }
        })
    }
    
    // Helper functions
    func depthToData(_ pixelBuffer: CVPixelBuffer) -> Data {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)!
        
        return Data(bytes: baseAddress, count: width * height * 4)
    }
    
    func poseToArray(_ pose: simd_float4x4) -> [[Float]] {
        return [
            [pose[0][0], pose[0][1], pose[0][2], pose[0][3]],
            [pose[1][0], pose[1][1], pose[1][2], pose[1][3]],
            [pose[2][0], pose[2][1], pose[2][2], pose[2][3]],
            [pose[3][0], pose[3][1], pose[3][2], pose[3][3]]
        ]
    }
    
    func intrinsicsToArray(_ intrinsics: simd_float3x3) -> [[Float]] {
        return [
            [intrinsics[0][0], intrinsics[0][1], intrinsics[0][2]],
            [intrinsics[1][0], intrinsics[1][1], intrinsics[1][2]],
            [intrinsics[2][0], intrinsics[2][1], intrinsics[2][2]]
        ]
    }
}
```

---

## 🐍 Python Server (ARKit Receiver)

### Create `orion/arkit/receiver.py`

```python
"""
ARKit Frame Receiver for Orion SLAM
====================================

Receives RGB, depth, and pose data from iOS ARKit and feeds into Orion pipeline.
"""

import socket
import struct
import json
import base64
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ARKitFrame:
    """ARKit frame data"""
    rgb: np.ndarray  # (H, W, 3) BGR
    depth: np.ndarray  # (H, W) float32 in meters
    pose: np.ndarray  # (4, 4) camera_from_world
    intrinsics: np.ndarray  # (3, 3) K matrix
    timestamp: float


class ARKitReceiver:
    """Receive ARKit frames over TCP"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.socket = None
        self.client_socket = None
        
    def start(self):
        """Start TCP server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        
        logger.info(f"ARKit receiver listening on {self.host}:{self.port}")
        
        # Wait for connection
        self.client_socket, addr = self.socket.accept()
        logger.info(f"Connected to {addr}")
        
    def receive_frame(self) -> Optional[ARKitFrame]:
        """Receive one frame from ARKit"""
        try:
            # Read size header (4 bytes)
            size_data = self._recv_exact(4)
            if not size_data:
                return None
            
            size = struct.unpack(">I", size_data)[0]
            
            # Read packet
            packet_data = self._recv_exact(size)
            if not packet_data:
                return None
            
            # Parse JSON
            packet = json.loads(packet_data.decode('utf-8'))
            
            # Decode RGB
            rgb_b64 = packet['rgb']
            rgb_data = base64.b64decode(rgb_b64)
            rgb_array = np.frombuffer(rgb_data, dtype=np.uint8)
            rgb = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
            
            # Decode depth
            depth_b64 = packet['depth']
            depth_data = base64.b64decode(depth_b64)
            depth = np.frombuffer(depth_data, dtype=np.float32)
            depth = depth.reshape((rgb.shape[0], rgb.shape[1]))
            
            # Parse pose
            pose = np.array(packet['pose'], dtype=np.float32)
            
            # Parse intrinsics
            intrinsics = np.array(packet['intrinsics'], dtype=np.float32)
            
            # Timestamp
            timestamp = packet['timestamp']
            
            return ARKitFrame(
                rgb=rgb,
                depth=depth,
                pose=pose,
                intrinsics=intrinsics,
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error(f"Error receiving frame: {e}")
            return None
    
    def _recv_exact(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n:
            chunk = self.client_socket.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def stop(self):
        """Close connection"""
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()


def arkit_to_orion_intrinsics(arkit_K: np.ndarray, width: int, height: int):
    """Convert ARKit intrinsics to Orion CameraIntrinsics"""
    from orion.perception.types import CameraIntrinsics
    
    return CameraIntrinsics(
        fx=arkit_K[0, 0],
        fy=arkit_K[1, 1],
        cx=arkit_K[0, 2],
        cy=arkit_K[1, 2],
        width=width,
        height=height
    )
```

---

## 🔗 Sensor Fusion Pipeline

### Create `orion/arkit/fusion.py`

```python
"""
ARKit-Orion SLAM Sensor Fusion
===============================

Fuses ARKit camera poses with Orion SLAM for robust tracking.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class FusedPose:
    """Fused camera pose"""
    pose: np.ndarray  # (4, 4) world_from_camera
    confidence: float  # 0-1
    source: str  # "arkit", "slam", "fused"


class ARKitOrionFusion:
    """Fuse ARKit and SLAM poses using Kalman filter"""
    
    def __init__(self, arkit_weight: float = 0.7):
        """
        Args:
            arkit_weight: Weight for ARKit pose (0-1). 
                         Higher = trust ARKit more
        """
        self.arkit_weight = arkit_weight
        self.slam_weight = 1.0 - arkit_weight
        
        # State
        self.prev_arkit_pose = None
        self.prev_slam_pose = None
        
    def fuse(
        self,
        arkit_pose: Optional[np.ndarray],
        slam_pose: Optional[np.ndarray]
    ) -> FusedPose:
        """
        Fuse ARKit and SLAM poses.
        
        Args:
            arkit_pose: (4, 4) camera_from_world from ARKit
            slam_pose: (4, 4) camera_from_world from Orion SLAM
        
        Returns:
            FusedPose with blended transformation
        """
        # Handle missing data
        if arkit_pose is None and slam_pose is None:
            return FusedPose(
                pose=np.eye(4),
                confidence=0.0,
                source="none"
            )
        
        if arkit_pose is None:
            return FusedPose(
                pose=slam_pose,
                confidence=0.5,
                source="slam"
            )
        
        if slam_pose is None:
            return FusedPose(
                pose=arkit_pose,
                confidence=0.8,
                source="arkit"
            )
        
        # Weighted average of translation
        t_arkit = arkit_pose[:3, 3]
        t_slam = slam_pose[:3, 3]
        t_fused = (
            self.arkit_weight * t_arkit + 
            self.slam_weight * t_slam
        )
        
        # SLERP for rotation (spherical linear interpolation)
        R_arkit = arkit_pose[:3, :3]
        R_slam = slam_pose[:3, :3]
        R_fused = self._slerp_rotation(
            R_arkit, R_slam, self.slam_weight
        )
        
        # Construct fused pose
        pose_fused = np.eye(4)
        pose_fused[:3, :3] = R_fused
        pose_fused[:3, 3] = t_fused
        
        return FusedPose(
            pose=pose_fused,
            confidence=0.95,
            source="fused"
        )
    
    def _slerp_rotation(
        self,
        R1: np.ndarray,
        R2: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Spherical linear interpolation between rotations"""
        from scipy.spatial.transform import Rotation, Slerp
        
        r1 = Rotation.from_matrix(R1)
        r2 = Rotation.from_matrix(R2)
        
        slerp = Slerp([0, 1], Rotation.concatenate([r1, r2]))
        r_interp = slerp([t])[0]
        
        return r_interp.as_matrix()
```

---

## 📊 Rerun Logging for ARKit

### Update `orion/visualization/rerun_logger.py`

```python
def log_arkit_frame(
    self,
    arkit_frame: ARKitFrame,
    frame_idx: int,
    log_mesh: bool = False
):
    """
    Log ARKit frame to Rerun.
    
    Args:
        arkit_frame: ARKit frame data
        frame_idx: Frame index
        log_mesh: Whether to log scene mesh
    """
    import rerun as rr
    
    rr.set_time("frame_idx", frame_idx)
    
    # Log RGB
    rr.log("arkit/camera/rgb", rr.Image(arkit_frame.rgb))
    
    # Log depth (LiDAR)
    rr.log(
        "arkit/camera/depth",
        rr.DepthImage(arkit_frame.depth, meter=1.0)
    )
    
    # Log camera pose
    rr.log(
        "arkit/camera",
        rr.Transform3D(transform=arkit_frame.pose)
    )
    
    # Log camera intrinsics
    h, w = arkit_frame.rgb.shape[:2]
    rr.log(
        "arkit/camera",
        rr.Pinhole(
            image_from_camera=arkit_frame.intrinsics,
            resolution=[w, h]
        )
    )
    
    # Log depth point cloud (backproject)
    if self.config.log_depth_3d:
        points_3d = self._backproject_depth(
            arkit_frame.depth,
            arkit_frame.intrinsics,
            arkit_frame.pose
        )
        rr.log(
            "arkit/depth_cloud",
            rr.Points3D(points_3d, colors=arkit_frame.rgb.reshape(-1, 3))
        )
```

---

## 🚀 Quick Start

### 1. **Setup iOS App**
```bash
# Clone template (to be created)
git clone https://github.com/orion-research/orion-arkit-streamer
cd orion-arkit-streamer

# Open in Xcode
open OrionARKit.xcodeproj

# Set your Mac's IP in Constants.swift
# Build and run on iPhone 12 Pro+ (LiDAR required)
```

### 2. **Start Orion Server**
```bash
# Install dependencies
pip install opencv-python numpy scipy

# Run ARKit receiver + SLAM
python -m orion arkit slam \
  --host 0.0.0.0 \
  --port 8765 \
  --viz rerun
```

### 3. **View in Rerun**
```bash
# Rerun viewer opens automatically
# Or manually:
rerun open
```

---

## 📈 Expected Performance

| Component | Latency | Notes |
|-----------|---------|-------|
| ARKit Capture | 16ms (60 FPS) | Native iOS |
| Network Transfer | 10-30ms | WiFi (JPEG compression) |
| YOLO Detection | 150-500ms | Every 15 frames |
| Orion Processing | 50-200ms | Entity tracking, zones |
| **Total Latency** | **200-700ms** | Near real-time |

---

## 🔗 References

- **ARKit Documentation**: https://developer.apple.com/documentation/arkit
- **Rerun ARKit Example**: https://github.com/rerun-io/rerun/tree/main/examples/python/arkit_scenes
- **LiDAR on iPhone**: https://developer.apple.com/documentation/arkit/arkit_in_ios/environmental_analysis
- **Orion SLAM**: `docs/PHASE_4_README.md`

---

## 📝 TODO

- [ ] Create iOS streaming app template
- [ ] Implement TCP receiver in Orion
- [ ] Add Kalman filter for pose fusion
- [ ] Depth map fusion (LiDAR + MiDaS)
- [ ] ARKit mesh → Orion spatial zones
- [ ] Test latency and accuracy
- [ ] Documentation and examples
