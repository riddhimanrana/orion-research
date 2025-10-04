"""
Visual System Architecture Diagram for Part 1
==============================================

This module generates an ASCII diagram of the complete perception engine architecture.
Run this file to see the detailed system flow.
"""

ARCHITECTURE_DIAGRAM = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ASYNCHRONOUS PERCEPTION ENGINE - PART 1                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│                                  VIDEO INPUT                                  │
│                          (MP4, AVI, MOV, etc.)                               │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    MODULE 1: VIDEO INGESTION & FRAME SELECTION                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. Load Video (OpenCV)                                               │    │
│  │    - Extract metadata (FPS, resolution, duration)                    │    │
│  │    - Calculate frame interval for 4 FPS sampling                     │    │
│  └──────────────────────────────┬───────────────────────────────────────┘    │
│                                  ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 2. Sample Frames at 4 FPS                                            │    │
│  │    - Read every Nth frame based on original FPS                      │    │
│  └──────────────────────────────┬───────────────────────────────────────┘    │
│                                  ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3. Scene Change Detection (FastViT-T8)                               │    │
│  │    - Resize frame to 768x768                                         │    │
│  │    - Generate scene embedding                                        │    │
│  │    - Compare with last processed frame (cosine similarity)           │    │
│  │    - If similarity < 0.98 → INTERESTING FRAME                        │    │
│  └──────────────────────────────┬───────────────────────────────────────┘    │
│                                  │                                            │
└──────────────────────────────────┼────────────────────────────────────────────┘
                                  │
                   Only "Interesting" Frames Pass Through
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  MODULE 2: TIER 1 - REAL-TIME PROCESSING                      │
│                        ("Field Reporter" - Fast Path)                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. Object Detection (YOLOv11m)                                       │    │
│  │    - Confidence threshold: 0.25                                      │    │
│  │    - IoU threshold: 0.45                                             │    │
│  │    - Max detections: 100 per frame                                   │    │
│  │    - Output: Bounding boxes, classes, confidence scores              │    │
│  └──────────────────────────────┬───────────────────────────────────────┘    │
│                                  ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 2. Filter Detections                                                 │    │
│  │    - Remove objects < 32x32 pixels                                   │    │
│  │    - Apply NMS (Non-Maximum Suppression)                             │    │
│  └──────────────────────────────┬───────────────────────────────────────┘    │
│                                  ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3. Crop Objects                                                      │    │
│  │    - Extract bounding box region                                     │    │
│  │    - Add 10% padding around edges                                    │    │
│  │    - Handle boundary conditions                                      │    │
│  └──────────────────────────────┬───────────────────────────────────────┘    │
│                                  ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 4. Generate Visual Embeddings (ResNet50)                             │    │
│  │    - Resize crop to model input size                                 │    │
│  │    - Extract 512-dim feature vector                                  │    │
│  │    - L2 normalize embedding                                          │    │
│  │    - This is the "visual fingerprint" for object permanence          │    │
│  └──────────────────────────────┬───────────────────────────────────────┘    │
│                                  ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 5. Create RichPerceptionObject (Preliminary)                         │    │
│  │    {                                                                 │    │
│  │      timestamp: float,                                               │    │
│  │      frame_number: int,                                              │    │
│  │      bounding_box: [x1, y1, x2, y2],                                 │    │
│  │      visual_embedding: [512 floats],                                 │    │
│  │      detection_confidence: float,                                    │    │
│  │      object_class: string,                                           │    │
│  │      crop_size: [width, height],                                     │    │
│  │      rich_description: null,  ← TO BE FILLED BY TIER 2               │    │
│  │      temp_id: "det_XXXXXX"                                           │    │
│  │    }                                                                 │    │
│  └──────────────────────────────┬───────────────────────────────────────┘    │
│                                  ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 6. Queue for Tier 2                                                  │    │
│  │    - Add to shared perception log (multiprocessing.Manager.list)     │    │
│  │    - Queue cropped image + index for workers                         │    │
│  │    - Non-blocking put with timeout                                   │    │
│  └──────────────────────────────┬───────────────────────────────────────┘    │
│                                  │                                            │
└──────────────────────────────────┼────────────────────────────────────────────┘
                                  │
                    Continue processing next frame
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   MULTIPROCESSING QUEUE     │
                    │   (maxsize: 1000)           │
                    │                             │
                    │   Items:                    │
                    │   - Cropped image (ndarray) │
                    │   - Object index (int)      │
                    │   - Temporary ID (string)   │
                    └──────────────┬──────────────┘
                                  │
                    Workers pull from queue (async)
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  MODULE 3: TIER 2 - ASYNCHRONOUS DESCRIPTION                  │
│                        ("Analyst" - Slow Path)                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌───────────────────────────┐  ┌───────────────────────────┐               │
│  │   WORKER PROCESS 1        │  │   WORKER PROCESS 2        │               │
│  ├───────────────────────────┤  ├───────────────────────────┤               │
│  │                           │  │                           │               │
│  │ While not stopped:        │  │ While not stopped:        │               │
│  │                           │  │                           │               │
│  │ 1. Pull from queue        │  │ 1. Pull from queue        │               │
│  │    (timeout: 0.5s)        │  │    (timeout: 0.5s)        │               │
│  │                           │  │                           │               │
│  │ 2. Get object class       │  │ 2. Get object class       │               │
│  │                           │  │                           │               │
│  │ 3. Generate description   │  │ 3. Generate description   │               │
│  │    via FastVLM:           │  │    via FastVLM:           │               │
│  │    - Prepare image        │  │    - Prepare image        │               │
│  │    - Create prompt        │  │    - Create prompt        │               │
│  │    - Run inference        │  │    - Run inference        │               │
│  │    - Extract text         │  │    - Extract text         │               │
│  │                           │  │                           │               │
│  │ 4. Update shared log      │  │ 4. Update shared log      │               │
│  │    (thread-safe)          │  │    (thread-safe)          │               │
│  │                           │  │                           │               │
│  │ 5. Repeat                 │  │ 5. Repeat                 │               │
│  │                           │  │                           │               │
│  └───────────────────────────┘  └───────────────────────────┘               │
│                                                                               │
│  Shutdown: Poison pill (None) sent to each worker                            │
│  Workers join with timeout, force terminate if needed                        │
│                                                                               │
└──────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                    All workers complete
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE PERCEPTION LOG                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  List of RichPerceptionObject dictionaries, fully populated:                 │
│                                                                               │
│  [                                                                            │
│    {                                                                          │
│      "timestamp": 15.25,                                                      │
│      "frame_number": 61,                                                      │
│      "bounding_box": [120, 340, 250, 580],                                    │
│      "visual_embedding": [0.12, -0.45, ..., 0.89],  // 512 floats             │
│      "detection_confidence": 0.87,                                            │
│      "object_class": "person",                                                │
│      "crop_size": [143, 264],                                                 │
│      "rich_description": "A person wearing a blue jacket, standing...",       │
│      "entity_id": null,  // To be filled in Part 2                            │
│      "temp_id": "det_000042"                                                  │
│    },                                                                         │
│    { ... },                                                                   │
│    { ... }                                                                    │
│  ]                                                                            │
│                                                                               │
│  Saved as JSON for Part 2 (Semantic Uplift)                                  │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘


╔══════════════════════════════════════════════════════════════════════════════╗
║                              KEY DESIGN DECISIONS                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. TWO-TIER ARCHITECTURE
   Why: Decouple fast operations (detection) from slow operations (description)
   Benefit: Tier 1 doesn't wait for Tier 2, maximizing throughput

2. SCENE CHANGE DETECTION
   Why: Many consecutive frames are nearly identical
   Benefit: 50-80% reduction in frames processed, saves compute

3. MULTIPROCESSING (not threading)
   Why: GIL prevents true parallelism in Python threads
   Benefit: True parallel processing of descriptions

4. SHARED MEMORY (Manager.list)
   Why: Workers need to update perception log in-place
   Benefit: Avoid pickling overhead, faster updates

5. VISUAL EMBEDDINGS
   Why: Enable entity tracking in Part 2
   Benefit: Object permanence across frames via clustering

6. QUEUE WITH TIMEOUT
   Why: Prevent deadlock if workers crash
   Benefit: Graceful degradation, continues processing

7. LAZY MODEL LOADING
   Why: Models are large, not all may be needed
   Benefit: Faster startup, lower memory if models disabled


╔══════════════════════════════════════════════════════════════════════════════╗
║                            PERFORMANCE BREAKDOWN                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Per Frame (Tier 1):
  - Scene detection:     10-20 ms  (FastViT)
  - Object detection:    50-100 ms (YOLOv11m)
  - Embedding per obj:   5-10 ms   (ResNet50)
  - Total Tier 1:        ~100-200 ms per frame

Per Object (Tier 2):
  - Description gen:     1-2 seconds (FastVLM)
  - With 2 workers:      Parallel processing

Overall Throughput:
  - Bottleneck: Tier 2 (description generation)
  - Expected:   3-5 frames/sec end-to-end
  - Optimization: Add more workers (linear speedup up to CPU cores)


╔══════════════════════════════════════════════════════════════════════════════╗
║                          INTEGRATION WITH PART 2                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Part 1 Output → Part 2 Input:

perception_log = run_perception_engine(video_path)
                        │
                        │ List of RichPerceptionObjects
                        │
                        ▼
            ┌───────────────────────────┐
            │  PART 2: SEMANTIC UPLIFT  │
            ├───────────────────────────┤
            │ 1. Entity Tracking        │
            │    (HDBSCAN clustering)   │
            │                           │
            │ 2. State Change Detection │
            │    (sentence similarity)  │
            │                           │
            │ 3. Event Composition      │
            │    (LLM reasoning)        │
            │                           │
            │ 4. Knowledge Graph        │
            │    (Neo4j ingestion)      │
            └───────────────────────────┘

The visual_embedding field is crucial for Part 2's clustering algorithm!


╔══════════════════════════════════════════════════════════════════════════════╗
║                               FILE STRUCTURE                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

production/
├── part1_perception_engine.py  ← Main implementation (850+ lines)
│   ├── Config class
│   ├── ModelManager class
│   ├── VideoFrameSelector class
│   ├── RealTimeObjectProcessor class
│   ├── description_worker function
│   └── run_perception_engine function (MAIN ENTRY POINT)
│
├── part1_config.py             ← Configuration presets (250+ lines)
│   ├── FAST_CONFIG
│   ├── BALANCED_CONFIG
│   ├── ACCURATE_CONFIG
│   ├── apply_config()
│   └── recommend_config()
│
├── test_part1.py               ← Testing utilities (150+ lines)
│   ├── generate_sample_video()
│   ├── analyze_perception_log()
│   └── main() with CLI
│
└── Documentation:
    ├── README_PART1.md         ← Full documentation (500+ lines)
    ├── QUICKSTART_PART1.md     ← Quick reference
    └── IMPLEMENTATION_SUMMARY.md ← This overview

"""

DATA_FLOW_DIAGRAM = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DATA FLOW & TRANSFORMATION PIPELINE                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

INPUT: video.mp4 (1920x1080, 30 FPS, 60 seconds = 1800 frames)
   │
   ├─> Frame Selection (4 FPS sampling) = 240 frames
   │
   ├─> Scene Change Detection (similarity < 0.98) = ~120 "interesting" frames
   │
   ├─> Object Detection (avg 3 objects/frame) = ~360 detections
   │
   ├─> Visual Embedding (512-dim per object) = 360 x 512 floats
   │
   ├─> Description Generation (async) = 360 text descriptions
   │
   └─> OUTPUT: perception_log.json

   Size: ~5-10 MB for 360 objects
   Time: ~60-120 seconds processing
   Memory: ~3-4 GB peak


DATA TRANSFORMATIONS:

Video Frame (1920x1080x3 BGR)
   ↓ resize
768x768x3 RGB → FastViT → 512-dim scene embedding
                              ↓ cosine similarity
                            0.94 < 0.98 → INTERESTING!

Interesting Frame (1920x1080x3 BGR)
   ↓ YOLOv11m
[
  {bbox: [x,y,w,h], class: "person", conf: 0.87},
  {bbox: [x,y,w,h], class: "car", conf: 0.92}
]
   ↓ crop + pad
[
  (320x480x3 BGR),  # person crop
  (280x180x3 BGR)   # car crop
]
   ↓ ResNet50
[
  [0.12, -0.45, ..., 0.89],  # 512-dim embedding
  [-0.23, 0.67, ..., -0.34]  # 512-dim embedding
]
   ↓ FastVLM (async)
[
  "A person wearing blue jacket, standing near building",
  "A red sedan parked on the street"
]
   ↓ combine
[
  RichPerceptionObject{...},
  RichPerceptionObject{...}
]
"""


def main():
    """Display all diagrams"""
    print(ARCHITECTURE_DIAGRAM)
    print("\n\n")
    print(DATA_FLOW_DIAGRAM)
    
    print("\n" + "="*80)
    print("For more details, see:")
    print("  - production/README_PART1.md")
    print("  - production/QUICKSTART_PART1.md")
    print("  - production/IMPLEMENTATION_SUMMARY.md")
    print("="*80)


if __name__ == "__main__":
    main()
