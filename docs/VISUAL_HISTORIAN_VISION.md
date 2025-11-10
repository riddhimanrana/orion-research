# Orion Visual Historian: Long-Term Architecture Vision

**Status**: Design Document  
**Target**: On-device continuous visual memory system  
**Timeline**: Multi-phase roadmap  
**Last Updated**: January 2025

---

## Executive Summary

This document outlines the architectural evolution of Orion from a video analysis system into a **continuous visual historian** capable of:

- Running on-device (phone) for 2-3 days continuously
- Building persistent spatial-temporal memory graphs
- Enabling natural language question answering about past events
- Integrating with ARKit for enhanced spatial understanding (iOS)
- Providing efficient, privacy-preserving personal visual memory

---

## 1. Current State (Phase 4)

### What We Have ✅

**SLAM & 3D Tracking**:
- OpenCV-based visual SLAM with loop closure
- 3D entity tracking with Re-ID
- Spatial zone detection and classification
- Rerun 3D visualization

**Knowledge Graph**:
- Neo4j-based causal inference system
- Spatial relationships and event chains
- Frame-to-frame entity tracking

**Processing Pipeline**:
- YOLO11x for detection
- MiDaS for depth estimation
- FastVLM for scene descriptions
- Transformer-based Re-ID

### Limitations ❌

- **Not Continuous**: Processes pre-recorded videos, not live streams
- **Not Persistent**: Memory lost after processing
- **Not Queryable**: No natural language interface
- **Not On-Device**: Requires workstation-class hardware
- **No Long-Term Memory**: Cannot handle multi-day operation

---

## 2. Vision: The Visual Historian

### Core Concept

Orion becomes a **wearable AI companion** that:

1. **Continuously observes** your environment through a phone camera
2. **Builds persistent memory** of places, people, and events
3. **Answers questions** about what you saw, where, and when
4. **Maintains context** across hours and days
5. **Runs efficiently** on-device with minimal battery impact

### Example Use Cases

**Scenario 1: Lost Keys**
```
User: "Where did I put my keys this morning?"
Orion: "You placed them on the kitchen counter at 8:15 AM, 
        near the coffee maker. Last seen in Zone: Kitchen."
```

**Scenario 2: Meeting Recall**
```
User: "Who was in the conference room yesterday afternoon?"
Orion: "Meeting from 2:30-4:00 PM. Detected: Alice (ID #7), 
        Bob (ID #12), Charlie (ID #3). Location: Room 2B."
```

**Scenario 3: Object Tracking**
```
User: "When did I last see my water bottle?"
Orion: "Water bottle (ID #45) last detected 3 hours ago 
        in Zone: Office Desk. Trajectory shows it was moved 
        from Kitchen → Living Room → Office."
```

---

## 3. Technical Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Phone/Wearable Device                   │
├─────────────────────────────────────────────────────────────┤
│  ARKit (iOS) / ARCore (Android)                             │
│  ├─── Camera Feed (30fps or adaptive)                       │
│  ├─── IMU/Pose Estimation                                   │
│  └─── Depth Sensing (LiDAR on iPhone Pro)                   │
├─────────────────────────────────────────────────────────────┤
│  Orion Processing Engine (On-Device)                        │
│  ├─── Lightweight Detection (YOLO-Nano/MobileNet)           │
│  ├─── Monocular Depth (MiDaS-Small)                         │
│  ├─── Incremental SLAM (ORB-SLAM3 Mobile)                   │
│  ├─── Entity Tracker (ByteTrack + Re-ID)                    │
│  └─── Zone Detector (Spatial Heuristics)                    │
├─────────────────────────────────────────────────────────────┤
│  Persistent Memory System                                   │
│  ├─── SQLite Spatial-Temporal Graph                         │
│  ├─── Entity Embeddings (FP16/INT8)                         │
│  ├─── Key Frames (JPEG compressed)                          │
│  └─── Event Index (Fast retrieval)                          │
├─────────────────────────────────────────────────────────────┤
│  QA Interface                                                │
│  ├─── Speech Recognition (Whisper-Tiny)                     │
│  ├─── Query Parser (TinyLLaMA/Phi-3 INT4)                   │
│  ├─── Graph Query Engine (Cypher-like)                      │
│  └─── Response Generator (Templated + LLM)                  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Component Breakdown

#### A. Perception Layer (On-Device Inference)

**Challenge**: Run models efficiently on phone hardware

**Solutions**:

1. **Object Detection**:
   - Model: YOLO-Nano or MobileNetV3 + SSDLite
   - Quantization: INT8 using CoreML (iOS) or TFLite (Android)
   - Inference: ~15-20ms on iPhone 15 Pro
   - Optimization: Adaptive frame rate (skip frames when static)

2. **Depth Estimation**:
   - **Primary**: ARKit LiDAR (iPhone Pro models) - native depth at 60fps
   - **Fallback**: MiDaS-Small (INT8) for non-Pro devices
   - Fusion: Combine sensor depth with monocular estimates

3. **Re-Identification**:
   - Model: OSNet (lightweight) or MobileNet-based Re-ID
   - Quantization: FP16 embeddings (512-dim → 256-dim)
   - Caching: Store embeddings for known entities
   - Recency bias: Prioritize recent observations

4. **Scene Understanding**:
   - **Live Inference**: Not feasible on-device (FastVLM too heavy)
   - **Solution**: Periodic cloud sync for scene descriptions
   - **Fallback**: Rule-based zone classification from object patterns

#### B. SLAM & Localization

**Challenge**: Continuous tracking without drift over days

**Solutions**:

1. **Hybrid SLAM**:
   - ARKit/ARCore for **short-term tracking** (< 1 hour)
   - Visual features for **loop closure** (revisiting locations)
   - GPS/WiFi for **coarse localization** (building-level)

2. **Memory-Efficient Mapping**:
   - Store **keyframes only** (1 per second max)
   - Prune redundant points (keep <10K active map points)
   - Use **hierarchical structure**: Day → Hour → Minute

3. **Multi-Session SLAM**:
   - Save/load maps between sessions
   - Re-localization when returning to known places
   - Merge maps from different days

#### C. Persistent Memory System

**Challenge**: Store 2-3 days of data (~500K-1M frames) efficiently

**Solutions**:

1. **Storage Architecture**:
   ```
   On-Device SQLite Database:
   ├─── entities/
   │    ├─── entity_observations (time, zone_id, bbox, embedding)
   │    └─── entity_metadata (class, first_seen, last_seen, count)
   ├─── zones/
   │    ├─── zone_definitions (bounds, type, level)
   │    └─── zone_transitions (entity_id, from_zone, to_zone, time)
   ├─── events/
   │    ├─── interactions (entity_a, entity_b, type, time)
   │    └─── notable_events (type, description, keyframe_id)
   ├─── keyframes/
   │    └─── images (JPEG, 10% quality, 480x270 res)
   └─── slam/
        ├─── poses (camera_pose, time)
        └─── map_points (3D coordinates, descriptor)
   ```

2. **Data Retention Policy**:
   - **First 24 hours**: Full resolution tracking (1 entity update/sec)
   - **24-48 hours**: Downsample to key events only
   - **48-72 hours**: Keep zone transitions + notable events
   - **After 3 days**: Export to cloud or delete

3. **Estimated Storage**:
   - Keyframes: ~5-10 MB/hour (600-1200 frames at 10% JPEG)
   - Entity data: ~2-5 MB/hour (embeddings + bboxes)
   - SLAM data: ~10-20 MB/hour (poses + sparse map)
   - **Total**: ~20-40 MB/hour → **1.5-3 GB for 3 days**

#### D. Question Answering System

**Challenge**: Parse natural language queries and retrieve relevant memories

**Solutions**:

1. **Query Processing Pipeline**:
   ```
   User Speech → Whisper-Tiny (STT) → Query Text
        ↓
   TinyLLaMA/Phi-3 (INT4) → Query Intent + Entities
        ↓
   Query Planner → Generate SQL/Cypher-like queries
        ↓
   SQLite Execution → Retrieve relevant observations
        ↓
   Response Generator → Natural language answer
   ```

2. **Query Types**:
   - **Spatial**: "Where is X?" → Last known zone for entity X
   - **Temporal**: "When did I see Y?" → Timestamp of first/last observation
   - **Tracking**: "Where did Z go?" → Zone transition sequence
   - **Counting**: "How many people in room?" → Unique entities in zone
   - **Re-ID**: "Who was that person?" → Match to known entity

3. **On-Device LLM**:
   - Model: TinyLLaMA (1.1B) or Phi-3-mini (3.8B) in INT4
   - Inference: ~50-100ms per query on iPhone 15 Pro
   - Role: Intent parsing + response generation (not reasoning)

4. **Graph Query Language**:
   ```python
   # Example: "Where did I put my keys this morning?"
   query = {
       "entity_class": "keys",
       "time_range": ("today 00:00", "now"),
       "query_type": "last_location",
       "result": "zone_name + timestamp"
   }
   
   # SQL equivalent:
   SELECT zones.name, observations.timestamp
   FROM entity_observations observations
   JOIN zones ON observations.zone_id = zones.id
   WHERE observations.entity_id IN (
       SELECT id FROM entities WHERE class='keys'
   )
   AND timestamp > '2025-01-15 00:00:00'
   ORDER BY timestamp DESC LIMIT 1;
   ```

---

## 4. On-Device Efficiency Strategies

### 4.1 Power Management

**Target**: Run 8-12 hours on single charge (background mode)

**Strategies**:

1. **Adaptive Frame Rate**:
   - Static scene: 1 fps
   - Motion detected: 15-30 fps
   - Use accelerometer to detect movement

2. **Selective Processing**:
   - Detection: Every frame
   - Re-ID: Only for new/uncertain entities
   - Depth: Only when needed (new zones, object interactions)
   - Scene description: Periodic (1 per minute) or on-demand

3. **Background Processing**:
   - Run in low-power mode when phone locked
   - Use iOS Background Tasks API
   - Batch processing during charging

4. **Model Optimization**:
   - All models quantized to INT8/INT4
   - Use Neural Engine (iOS) / NPU (Android)
   - Compile models with CoreML/NNAPI

### 4.2 Memory Management

**Target**: <500 MB RAM usage

**Strategies**:

1. **Streaming Processing**:
   - Process frame-by-frame (no video buffering)
   - Keep only last 100 frames in memory for tracking

2. **Feature Caching**:
   - Cache Re-ID embeddings for known entities
   - Limit cache to 1000 most recent entities

3. **Map Pruning**:
   - Keep only active map points (last 1000 keyframes)
   - Store historical maps in compressed format

### 4.3 Thermal Management

**Challenge**: Continuous processing causes overheating

**Solutions**:

1. **Duty Cycling**:
   - Process at full rate for 10 minutes
   - Drop to 1 fps for 5 minutes (cooldown)
   - Monitor device temperature via APIs

2. **Workload Distribution**:
   - Run heavy models (Re-ID) less frequently
   - Offload scene descriptions to cloud during WiFi

---

## 5. ARKit Integration (iOS)

### 5.1 Why ARKit?

- **High-Quality Depth**: LiDAR provides accurate depth maps (iPhone 12 Pro+)
- **6-DOF Tracking**: Precise camera pose estimation
- **Scene Understanding**: Plane detection, mesh reconstruction
- **World Tracking**: Persistent anchors across sessions

### 5.2 Integration Points

```swift
// ARKit Session Setup
let config = ARWorldTrackingConfiguration()
config.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
config.planeDetection = [.horizontal, .vertical]

// Pass to Orion
func session(_ session: ARSession, didUpdate frame: ARFrame) {
    let rgbImage = frame.capturedImage  // CVPixelBuffer
    let depthMap = frame.sceneDepth?.depthMap  // CVPixelBuffer (LiDAR)
    let cameraPose = frame.camera.transform  // 4x4 matrix
    
    OrionEngine.process(rgb: rgbImage, depth: depthMap, pose: cameraPose)
}
```

### 5.3 Benefits

- **No SLAM Needed**: ARKit handles localization
- **Better Depth**: LiDAR > monocular estimation
- **Anchors**: Attach entities to world coordinates
- **Mesh**: Use ARMeshAnchor for zone detection

---

## 6. Implementation Roadmap

### Phase 5: On-Device Prototyping (Months 1-2)

**Goal**: Prove feasibility of on-device inference

**Milestones**:
1. ✅ Export models to CoreML/TFLite
2. ✅ Benchmark inference times on target devices
3. ✅ Implement streaming video processing
4. ✅ Measure power consumption

**Deliverables**:
- iOS/Android app with live detection
- Performance report (FPS, power, memory)

### Phase 6: Memory System (Months 3-4)

**Goal**: Build persistent spatial-temporal database

**Milestones**:
1. ✅ Design SQLite schema for entities/zones/events
2. ✅ Implement incremental graph updates
3. ✅ Add data retention and pruning
4. ✅ Build keyframe compression pipeline

**Deliverables**:
- Persistent memory system running 24+ hours
- Storage profiling and optimization

### Phase 7: SLAM Integration (Months 5-6)

**Goal**: Continuous tracking without drift

**Milestones**:
1. ✅ Integrate ARKit/ARCore for pose estimation
2. ✅ Implement multi-session localization
3. ✅ Add loop closure for long-term consistency
4. ✅ Test 2-3 day continuous operation

**Deliverables**:
- Multi-day tracking demo
- Map persistence across sessions

### Phase 8: QA System (Months 7-8)

**Goal**: Natural language interface

**Milestones**:
1. ✅ Deploy on-device LLM (TinyLLaMA/Phi-3)
2. ✅ Build query parser and graph search
3. ✅ Implement voice interface (Whisper)
4. ✅ Design response templates

**Deliverables**:
- Voice-activated QA system
- Demo: "Where did I put X?" queries

### Phase 9: Optimization & Deployment (Months 9-10)

**Goal**: Production-ready system

**Milestones**:
1. ✅ Battery optimization (8+ hours runtime)
2. ✅ Thermal management
3. ✅ Privacy features (local-only mode)
4. ✅ Cloud sync (optional)

**Deliverables**:
- App Store / Play Store release
- User testing and feedback

---

## 7. Technical Challenges & Solutions

### Challenge 1: Real-Time Processing on Phone

**Problem**: Current pipeline too slow for 30fps processing

**Solutions**:
1. **Model Compression**:
   - YOLO11x → YOLO-Nano (10x faster)
   - MiDaS → MiDaS-Small or ARKit LiDAR
   - FastVLM → Periodic cloud calls

2. **Selective Processing**:
   - Detect every frame, Re-ID only uncertain entities
   - Zone detection: 1 fps (stable environments)
   - Depth: Only when objects detected

3. **Hardware Acceleration**:
   - CoreML for iOS Neural Engine
   - TensorFlow Lite for Android NPU

**Validation**: Benchmark on iPhone 15 Pro + Pixel 8 Pro

---

### Challenge 2: 2-3 Day Memory Without Drift

**Problem**: SLAM accumulates error over long sessions

**Solutions**:
1. **Hybrid Localization**:
   - ARKit/ARCore: Short-term (< 1 hour)
   - Loop closure: Correct drift when revisiting
   - GPS/WiFi: Coarse anchoring

2. **Memory Consolidation**:
   - Keep sparse map (10K points max)
   - Prune redundant observations
   - Store entity summaries, not raw frames

3. **Multi-Session SLAM**:
   - Save map on session end
   - Re-localize on restart
   - Merge maps using place recognition

**Validation**: Run 72-hour test with daily restarts

---

### Challenge 3: Privacy & Security

**Problem**: Continuous recording raises privacy concerns

**Solutions**:
1. **Local-Only Mode**:
   - All data stored on-device
   - No cloud sync unless user opts in
   - Delete data after 3 days (default)

2. **Anonymization**:
   - Re-ID embeddings only (no face images)
   - Blur faces in keyframes
   - User can mark "private zones" (no recording)

3. **Transparency**:
   - Show what's being recorded (live indicator)
   - Let users review and delete memories
   - Export data in open format (SQLite)

---

### Challenge 4: Battery Life

**Problem**: Continuous processing drains battery fast

**Solutions**:
1. **Adaptive Processing** (see Section 4.1)
2. **Background Mode**:
   - Reduce to 1 fps when screen off
   - Pause during phone calls
   - Resume automatically

3. **User Control**:
   - "Low Power Mode" (1 fps, detection only)
   - "Full Tracking" (30 fps, all features)
   - Auto-switch based on battery level

**Target**: 8-12 hours on single charge (background mode)

---

## 8. Competitive Landscape

### Existing Systems

| System | Approach | Limitations |
|--------|----------|-------------|
| **Google Lens** | Cloud-based visual search | No continuous memory, no tracking |
| **Apple Visual Look Up** | On-device scene understanding | No temporal memory, no QA |
| **Rewind AI** | Screen recording + OCR | No spatial understanding, privacy concerns |
| **Humane AI Pin** | Wearable with camera | Cloud-dependent, expensive, poor battery |
| **Ray-Ban Meta** | Smart glasses | No SLAM, no persistent memory |

### Orion's Advantages

- ✅ **Fully On-Device**: Privacy-preserving, works offline
- ✅ **Spatial-Temporal Memory**: 3D understanding + timeline
- ✅ **Question Answering**: Natural language interface
- ✅ **Long-Term Tracking**: Multi-day memory (not just snapshots)
- ✅ **Open Source**: Transparent, customizable

---

## 9. Success Metrics

### Technical Metrics

- **Inference Speed**: 20+ FPS on iPhone 15 Pro / Pixel 8 Pro
- **Battery Life**: 8-12 hours continuous operation
- **Memory Usage**: <500 MB RAM, <3 GB storage for 3 days
- **Tracking Accuracy**: >90% entity Re-ID accuracy over 24 hours
- **QA Accuracy**: >80% correct answers on benchmark queries

### User Metrics

- **Utility**: Users ask 10+ questions per day
- **Retention**: 70% of users continue after 1 week
- **Privacy**: 90% of users comfortable with local-only mode
- **Performance**: <5 second response time for queries

---

## 10. Future Directions (Beyond Phase 9)

### Multi-Modal Memory

- **Audio**: Record conversations (opt-in) for context
- **Text**: Extract text from signs, documents (OCR)
- **Gestures**: Detect pointing, waving for intent

### Shared Memories

- **Multi-Device**: Sync across phone + watch + glasses
- **Collaborative**: Share memories with family/friends
- **Place-Based**: Public memory graph for locations (e.g., museum)

### Advanced Reasoning

- **Causal Inference**: "Why did X happen?" (beyond spatial-temporal)
- **Prediction**: "What usually happens next?" (learned patterns)
- **Summarization**: "What did I do today?" (daily digests)

---

## 11. References & Prior Art

### Academic Papers

- **ORB-SLAM3** (Campos et al., 2021): Multi-session visual SLAM
- **DeepSORT** (Wojke et al., 2017): Online multi-object tracking
- **Rewind** (Rewind.ai, 2023): Personal memory assistant
- **LLaMA** (Touvron et al., 2023): On-device language models

### Open Source Projects

- **ARKit**: Apple's AR framework
- **MediaPipe**: Google's on-device ML
- **TensorFlow Lite**: Mobile ML framework
- **Rerun**: 3D visualization for robotics

---

## 12. Conclusion

The **Visual Historian** vision transforms Orion from a video analysis tool into a **personal AI companion** that continuously builds and maintains spatial-temporal memory. By leveraging on-device inference, persistent memory graphs, and natural language interfaces, we can create a system that:

- Runs efficiently on modern phones
- Respects user privacy (local-only processing)
- Enables powerful question-answering about past events
- Maintains memory across days (not just minutes)

This document provides a **roadmap** for the next 10 months of development, balancing technical feasibility with user value. The phased approach allows us to validate assumptions early and iterate based on real-world testing.

---

**Next Steps**:
1. Review and refine this vision with stakeholders
2. Begin Phase 5 (On-Device Prototyping) with CoreML exports
3. Set up test devices (iPhone 15 Pro, Pixel 8 Pro)
4. Start benchmarking inference times and power consumption

**Questions? Contact**: Orion Research Team
