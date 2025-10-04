# Part 1: The Asynchronous Perception Engine

## Overview

This module implements a sophisticated two-tier video processing system that efficiently extracts rich visual observations from video files. It's designed to balance speed with semantic depth by decoupling fast real-time tasks from slower detail-oriented processing.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO INPUT                                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│         INTELLIGENT FRAME SELECTION                              │
│  - Sample at 4 FPS                                               │
│  - FastViT scene change detection (cosine similarity < 0.98)     │
│  - Filter redundant frames                                       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 1: REAL-TIME PROCESSING ("Field Reporter")                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. YOLOv11m Object Detection                             │   │
│  │    - Confidence threshold: 0.25                          │   │
│  │    - Filter objects < 32x32 pixels                       │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        │                                         │
│  ┌─────────────────────▼───────────────────────────────────┐   │
│  │ 2. Object Cropping (10% padding)                         │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        │                                         │
│  ┌─────────────────────▼───────────────────────────────────┐   │
│  │ 3. Visual Embedding (ResNet50/OSNet)                     │   │
│  │    - 512-dim normalized vector                           │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        │                                         │
│  ┌─────────────────────▼───────────────────────────────────┐   │
│  │ 4. Create RichPerceptionObject                           │   │
│  │ 5. Queue cropped image for Tier 2                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ Processing    │
              │ Queue         │
              │ (max 1000)    │
              └───────┬───────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 2: ASYNCHRONOUS DESCRIPTION ("Analyst")                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Worker Process 1                                         │   │
│  │ - Pull cropped image from queue                         │   │
│  │ - Generate rich description (FastVLM)                    │   │
│  │ - Update RichPerceptionObject                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Worker Process 2                                         │   │
│  │ - Parallel processing                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│         COMPLETE PERCEPTION LOG                                  │
│  List of RichPerceptionObject dictionaries                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Intelligent Frame Selection
- **Adaptive Sampling**: Samples video at 4 FPS to reduce computational load
- **Scene Change Detection**: Uses FastViT embeddings to detect significant scene changes
- **Similarity Threshold**: Only processes frames with cosine similarity < 0.98 to previous frame
- **Memory Efficient**: Maintains minimal state (last processed frame embedding only)

### 2. Two-Tier Asynchronous Processing
- **Tier 1 (Fast)**: Object detection and embedding generation (~50-100ms per frame)
- **Tier 2 (Slow)**: Rich description generation via FastVLM (~1-2s per object)
- **Decoupled**: Tier 1 continues processing while Tier 2 works asynchronously
- **Scalable**: Multiple worker processes for parallel description generation

### 3. Rich Perception Objects
Each detected object contains:
- **Temporal**: `timestamp`, `frame_number`
- **Spatial**: `bounding_box`, `crop_size`
- **Visual**: `visual_embedding` (512-dim fingerprint)
- **Semantic**: `rich_description` (natural language)
- **Metadata**: `object_class`, `detection_confidence`, `temp_id`

## Data Structure

```python
RichPerceptionObject {
    "timestamp": 15.25,                    # Video time in seconds
    "frame_number": 61,                    # Frame index
    "bounding_box": [120, 340, 250, 580],  # [x1, y1, x2, y2]
    "visual_embedding": [0.12, -0.45, ...], # 512-dim vector
    "detection_confidence": 0.87,          # YOLO confidence
    "object_class": "person",              # YOLO class label
    "crop_size": [143, 264],               # Width x Height
    "rich_description": "A person wearing...", # FastVLM output
    "entity_id": null,                     # Filled in Part 2
    "temp_id": "det_000042"               # Temporary unique ID
}
```

## Model Components

### 1. FastViT (Scene Detection)
- **Model**: `timm/fastvit_t8.apple_in1k`
- **Purpose**: Lightweight scene change detection
- **Input**: 768x768 RGB frames
- **Output**: Scene embedding vector
- **Speed**: ~10-20ms per frame on GPU

### 2. YOLOv11m (Object Detection)
- **Model**: `ultralytics/yolov11m.pt`
- **Purpose**: Real-time object detection
- **Input**: Full resolution frames
- **Output**: Bounding boxes, classes, confidence scores
- **Speed**: ~50-100ms per frame on GPU

### 3. ResNet50 (Visual Embeddings)
- **Model**: `timm/resnet50` (pretrained)
- **Purpose**: Generate visual fingerprints for object permanence
- **Input**: Cropped object images
- **Output**: 512-dim normalized embeddings
- **Speed**: ~5-10ms per crop on GPU

### 4. FastVLM (Rich Descriptions)
- **Model**: Custom LLaVA-based model from `ml-fastvlm/`
- **Purpose**: Generate detailed natural language descriptions
- **Input**: Cropped object images + prompts
- **Output**: Textual descriptions
- **Speed**: ~1-2s per object (placeholder: 0.1s)
- **Note**: Currently using placeholder; full integration pending

## Configuration

All configurable parameters are centralized in the `Config` class:

```python
class Config:
    # Video Processing
    TARGET_FPS = 4.0
    SCENE_SIMILARITY_THRESHOLD = 0.98
    FRAME_RESIZE_DIM = 768
    
    # Object Detection
    YOLO_CONFIDENCE_THRESHOLD = 0.25
    YOLO_IOU_THRESHOLD = 0.45
    MIN_OBJECT_SIZE = 32
    BBOX_PADDING_PERCENT = 0.10
    
    # Visual Embedding
    OSNET_INPUT_SIZE = (256, 128)
    EMBEDDING_DIM = 512
    
    # Multiprocessing
    NUM_WORKERS = 2
    QUEUE_MAX_SIZE = 1000
    QUEUE_TIMEOUT = 0.5
    
    # Logging
    LOG_LEVEL = logging.INFO
    PROGRESS_BAR = True
```

## Installation

### 1. Install Dependencies

```bash
# Make sure you're in the orion-research directory
cd /Users/riddhiman.rana/Desktop/Coding/Orion/orion-research

# Activate conda environment
conda activate orion

# Install new dependencies
pip install opencv-python==4.10.0.84
pip install ultralytics==8.3.0
pip install hdbscan==0.8.39
pip install sentence-transformers==3.3.1
pip install neo4j==5.26.0
```

### 2. Download Models

The script will automatically download required models on first run:
- FastViT: Auto-downloaded by timm
- YOLOv11m: Auto-downloaded by ultralytics (~40MB)
- ResNet50: Auto-downloaded by timm (~100MB)

## Usage

### Basic Usage

```python
from production.part1_perception_engine import run_perception_engine

# Process a video
video_path = "path/to/your/video.mp4"
perception_log = run_perception_engine(video_path)

# Access results
print(f"Detected {len(perception_log)} objects")
for obj in perception_log[:5]:
    print(f"  {obj['object_class']} at {obj['timestamp']:.2f}s: {obj['rich_description']}")
```

### Advanced Usage

```python
from production.part1_perception_engine import (
    ModelManager, VideoFrameSelector, RealTimeObjectProcessor
)

# Custom configuration
from production.part1_perception_engine import Config
Config.TARGET_FPS = 8.0  # Higher sampling rate
Config.NUM_WORKERS = 4   # More parallel workers
Config.SCENE_SIMILARITY_THRESHOLD = 0.95  # More sensitive scene detection

# Run with custom config
perception_log = run_perception_engine(video_path)
```

### Running the Script Directly

```bash
# Update VIDEO_PATH in the script's __main__ block, then:
python production/part1_perception_engine.py
```

## Performance Characteristics

### Throughput
- **Frame Selection**: ~100-200 frames/sec (with scene detection)
- **Tier 1 Processing**: ~5-10 frames/sec (with object detection)
- **Overall Pipeline**: ~3-5 frames/sec (bottlenecked by description generation)

### Memory Usage
- **Base**: ~2-3 GB (models loaded)
- **Processing**: +500MB per 1000 frames processed
- **Queue**: +100MB for full queue (1000 items)

### Scalability
- **Multiple Workers**: Linear speedup in Tier 2 (up to CPU core count)
- **GPU Acceleration**: 3-5x speedup for Tier 1 operations
- **Batch Processing**: Future optimization for embedding generation

## Output Format

The perception engine returns a list of dictionaries, which can be saved as JSON:

```python
import json

with open('perception_log.json', 'w') as f:
    json.dump(perception_log, f, indent=2)
```

Example output structure:
```json
[
  {
    "timestamp": 0.0,
    "frame_number": 0,
    "bounding_box": [245, 156, 489, 623],
    "visual_embedding": [0.123, -0.456, ...],
    "detection_confidence": 0.92,
    "object_class": "person",
    "crop_size": [268, 514],
    "rich_description": "A person wearing casual clothing...",
    "entity_id": null,
    "temp_id": "det_000000"
  },
  ...
]
```

## Error Handling

The engine includes robust error handling:
- **Missing Video**: Clear error message if file not found
- **Corrupted Frames**: Skip and continue processing
- **Model Loading Failures**: Graceful degradation or clear error
- **Queue Overflow**: Logs warning and skips description for that object
- **Worker Crashes**: Timeout and force termination if needed

## Logging

Detailed logging at multiple levels:
- **INFO**: Progress updates, statistics
- **DEBUG**: Detailed per-operation logs
- **WARNING**: Non-fatal issues (queue full, missing descriptions)
- **ERROR**: Failures that might affect results

Example log output:
```
2025-10-03 14:30:15 - PerceptionEngine - INFO - Video loaded: sample.mp4
2025-10-03 14:30:15 - PerceptionEngine - INFO -   Resolution: 1920x1080
2025-10-03 14:30:15 - PerceptionEngine - INFO -   FPS: 30.00
2025-10-03 14:30:15 - PerceptionEngine - INFO -   Duration: 45.20s
2025-10-03 14:30:16 - PerceptionEngine - INFO - Selected 98 interesting frames from 1356 total
2025-10-03 14:30:45 - PerceptionEngine - INFO - Tier 1 complete: 247 objects detected
2025-10-03 14:31:12 - PerceptionEngine - INFO - Description generation complete: 247/247 objects
```

## Limitations & Future Work

### Current Limitations
1. **FastVLM Integration**: Using placeholder descriptions (full integration pending)
2. **Single Video Processing**: No batch video processing yet
3. **No Checkpointing**: Can't resume interrupted processing
4. **Limited Error Recovery**: Some failures stop entire pipeline

### Future Enhancements
1. Integrate actual FastVLM model from `ml-fastvlm/`
2. Add checkpoint saving/loading for long videos
3. Implement batch embedding generation
4. Add video preprocessing (stabilization, enhancement)
5. Support streaming video input
6. GPU memory optimization for multiple models

## Testing

To test with a sample video:

```python
# Create a simple test video if needed
import cv2
import numpy as np

# Generate 10-second test video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_video.mp4', fourcc, 30.0, (640, 480))

for i in range(300):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Add a moving rectangle
    x = int(200 + 100 * np.sin(i * 0.1))
    cv2.rectangle(frame, (x, 200), (x+100, 300), (0, 255, 0), -1)
    out.write(frame)

out.release()

# Process the test video
perception_log = run_perception_engine('test_video.mp4')
```

## Integration with Parts 2 & 3

The output of Part 1 serves as input to Part 2 (Semantic Uplift):

```python
# Part 1: Perception
perception_log = run_perception_engine(video_path)

# Part 2: Semantic Uplift (to be implemented)
from production.part2_semantic_uplift import run_semantic_uplift
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
run_semantic_uplift(perception_log, driver)

# Part 3: Query & Evaluation (to be implemented)
from production.part3_query_evaluation import agent_augmented_sota
answer = agent_augmented_sota(query, video_clips, driver)
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch sizes or use CPU fallback
```python
Config.NUM_WORKERS = 1  # Reduce parallel workers
```

### Issue: "Queue is full" warnings
**Solution**: Increase queue size or reduce worker count
```python
Config.QUEUE_MAX_SIZE = 2000
```

### Issue: Models not downloading
**Solution**: Check internet connection and disk space
```bash
# Manually download models
python -c "import timm; timm.create_model('fastvit_t8.apple_in1k', pretrained=True)"
python -c "from ultralytics import YOLO; YOLO('yolov11m.pt')"
```

### Issue: Slow processing
**Solution**: Check GPU availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## License

Part of the Orion Research project. See LICENSE file for details.

## Contributors

- Riddhiman Rana
- Aryav Semwal
- Yogesh Atluru
- Jason Zhang

## Contact

For questions or issues, please contact the Orion Research team.
