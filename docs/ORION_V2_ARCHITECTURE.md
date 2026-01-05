# Orion v2: Complete Architecture Redesign

## Executive Summary

Orion v2 is a memory-centric video understanding system that tracks objects persistently across video, builds semantic scene graphs, and enables natural language querying. This document outlines the complete architectural redesign based on mentor feedback (Shivank Garg) and iterative testing.

**Key Changes from v1:**
1. **Re-ID Embedder**: CLIP → V-JEPA2 (3D-aware video encoder)
2. **Detection**: YOLO11x → YOLO-World (open-vocabulary with custom prompts)
3. **Removed**: Depth Anything V3 (FastVLM handles spatial relationships)
4. **Added**: Causal Influence Scoring (CIS) for object relationships
5. **Memory**: Memgraph persistent graph database with structured schema
6. **LLM**: Ollama with 20B+ model for reasoning and Q&A

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    ORION V2 PIPELINE                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: DETECTION + TRACKING                                                          │
│  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐            │
│  │   Video Input      │───▶│  YOLO-World v2     │───▶│  Online Tracker    │            │
│  │   (frames @ 5 FPS) │    │  (custom prompts)  │    │  (Hungarian + IoU) │            │
│  └────────────────────┘    └────────────────────┘    └────────────────────┘            │
│                                     │                          │                        │
│                          ┌──────────┴──────────┐               │                        │
│                          │ Object Categories:  │               │                        │
│                          │ person, furniture,  │               │                        │
│                          │ electronics, food,  │               │                        │
│                          │ container, tool...  │               │                        │
│                          └─────────────────────┘               │                        │
│                                                                ▼                        │
│                                                   ┌────────────────────┐               │
│                                                   │   tracks.jsonl     │               │
│                                                   │   (per-frame obs)  │               │
│                                                   └────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: VIDEO EMBEDDING (RE-ID) - THE CRITICAL FIX                                    │
│  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐            │
│  │  Track Crops       │───▶│  V-JEPA2 Encoder   │───▶│  Cosine Clustering │            │
│  │  (multi-frame)     │    │  (3D-aware ViT)    │    │  (threshold tuning)│            │
│  └────────────────────┘    └────────────────────┘    └────────────────────┘            │
│           │                         │                          │                        │
│           │                         │                          ▼                        │
│           │                         │               ┌────────────────────┐             │
│           ▼                         ▼               │  memory.json       │             │
│  ┌────────────────────────────────────────────┐    │  (clustered objs)  │             │
│  │  Option A: Best single frame per track     │    └────────────────────┘             │
│  │  Option B: Multi-crop video (16 frames)    │                                        │
│  │            → encode as "mini video"        │                                        │
│  └────────────────────────────────────────────┘                                        │
│                                                                                         │
│  WHY V-JEPA2?                                                                           │
│  • 3D-aware: handles same object from different angles                                  │
│  • Trained for robotics/prediction: understands object permanence                       │
│  • Lightweight ViT encoder: faster than full video models                               │
│  • Can treat single image as 1-frame video (backward compatible)                        │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: SEMANTIC FILTERING (FastVLM + Sentence Transformer)                          │
│  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐            │
│  │  Scene Sampling    │───▶│  FastVLM 0.5B      │───▶│  all-mpnet-base-v2 │            │
│  │  (cosine trigger)  │    │  (scene captions)  │    │  (768-dim text emb)│            │
│  └────────────────────┘    └────────────────────┘    └────────────────────┘            │
│           │                         │                          │                        │
│           │                         ▼                          ▼                        │
│           │               ┌─────────────────────────────────────────────┐              │
│           │               │  FILTERING LOGIC:                           │              │
│           │               │  1. FastVLM describes each object crop      │              │
│           │               │  2. FastVLM describes scene context         │              │
│           │               │  3. Sentence embed: object desc + scene desc│              │
│           │               │  4. Cosine similarity > threshold → KEEP    │              │
│           │               │  5. Low similarity → FALSE POSITIVE → REMOVE│              │
│           │               └─────────────────────────────────────────────┘              │
│           │                                        │                                    │
│           ▼                                        ▼                                    │
│  ┌────────────────────┐               ┌────────────────────┐                           │
│  │  vlm_scene.jsonl   │               │ tracks_filtered.jsonl │                        │
│  │  (scene captions)  │               │ (validated tracks)    │                        │
│  └────────────────────┘               └────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: SCENE GRAPH + CAUSAL INFLUENCE SCORING (CIS)                                  │
│  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐            │
│  │  Spatial Relations │───▶│  Temporal Tracking │───▶│  CIS Computation   │            │
│  │  (bbox overlap,    │    │  (object presence  │    │  (influence scores)│            │
│  │   distance, size)  │    │   across frames)   │    │                    │            │
│  └────────────────────┘    └────────────────────┘    └────────────────────┘            │
│           │                         │                          │                        │
│           ▼                         ▼                          ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐           │
│  │  SCENE GRAPH EDGES (per frame):                                          │           │
│  │  • NEAR(obj_a, obj_b, distance)                                          │           │
│  │  • ON(obj_a, obj_b)                                                      │           │
│  │  • HELD_BY(object, person)                                               │           │
│  │  • CONTAINS(container, object)                                           │           │
│  │                                                                           │           │
│  │  CIS EDGES (temporal):                                                    │           │
│  │  • CAUSES(event_a, event_b, influence_score)                             │           │
│  │  • CO_OCCURS(obj_a, obj_b, frequency)                                    │           │
│  │  • MOVES_WITH(obj_a, obj_b, correlation)                                 │           │
│  └─────────────────────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: MEMGRAPH PERSISTENCE                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐        │
│  │                           GRAPH SCHEMA                                      │        │
│  │                                                                             │        │
│  │   (:Episode {id, video_path, duration, fps})                               │        │
│  │        │                                                                    │        │
│  │        ├──[:HAS_FRAME]──▶ (:Frame {id, timestamp, scene_caption})          │        │
│  │        │                       │                                            │        │
│  │        │                       ├──[:CONTAINS]──▶ (:Observation {           │        │
│  │        │                       │                    track_id, bbox,         │        │
│  │        │                       │                    confidence, label})     │        │
│  │        │                       │                                            │        │
│  │        │                       └──[:NEAR|ON|HELD_BY]──▶ (:Observation)     │        │
│  │        │                                                                    │        │
│  │        └──[:HAS_OBJECT]──▶ (:MemoryObject {                                │        │
│  │                               id, canonical_label, description,            │        │
│  │                               embedding_vector, first_seen, last_seen,     │        │
│  │                               total_observations})                          │        │
│  │                                   │                                         │        │
│  │                                   └──[:OBSERVED_AS]──▶ (:Track {id, ...})  │        │
│  │                                                                             │        │
│  │   CIS RELATIONSHIPS:                                                        │        │
│  │   (:MemoryObject)──[:INFLUENCES {score, type}]──▶ (:MemoryObject)          │        │
│  │   (:MemoryObject)──[:CO_OCCURS {frequency}]──▶ (:MemoryObject)             │        │
│  │                                                                             │        │
│  └────────────────────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: LLM REASONING (Ollama)                                                        │
│  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐            │
│  │  Query Parser      │───▶│  Graph Retrieval   │───▶│  LLM (Qwen2.5-20B) │            │
│  │  (natural lang)    │    │  (Cypher queries)  │    │  via Ollama        │            │
│  └────────────────────┘    └────────────────────┘    └────────────────────┘            │
│                                                                │                        │
│  EXAMPLE QUERIES:                                              ▼                        │
│  • "What objects are in the kitchen?"              ┌────────────────────┐              │
│  • "When did the person pick up the cup?"          │  Natural Language  │              │
│  • "How many times did X interact with Y?"         │  Response          │              │
│  • "What caused the cup to fall?"                  └────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Detection: YOLO-World v2

**Why YOLO-World over YOLO11x?**
- Open-vocabulary: can detect any object class via text prompts
- No retraining needed for new categories
- "Prompt-then-detect" strategy enables domain adaptation
- Similar speed to YOLO11, slightly lower mAP but more flexible

**Configuration:**
```python
YOLO_WORLD_CONFIG = {
    "model": "yolov8x-worldv2",  # Largest, best quality
    "confidence": 0.25,
    "iou_threshold": 0.45,
    "classes": [
        # Core objects for indoor/activity tracking
        "person", "face", "hand",
        # Furniture
        "chair", "table", "desk", "couch", "sofa", "bed", "cabinet", "shelf",
        # Electronics
        "laptop", "phone", "tv", "monitor", "keyboard", "mouse", "remote",
        # Kitchen
        "cup", "mug", "glass", "bottle", "plate", "bowl", "utensil", "food",
        # Tools/Items
        "book", "bag", "backpack", "box", "container", "tool",
        # Background class for improved detection
        ""
    ]
}
```

**Implementation Pattern:**
```python
from ultralytics import YOLOWorld

model = YOLOWorld("yolov8x-worldv2.pt")
model.set_classes(YOLO_WORLD_CONFIG["classes"])
# Save custom model for faster inference
model.save("orion_yoloworld_custom.pt")
```

---

### 2. Re-ID: V-JEPA2 Video Encoder (THE CRITICAL FIX)

**Problem with CLIP (current approach):**
- CLIP is a 2D encoder trained on image-text pairs
- Cannot understand same object from different viewing angles
- Embeddings vary wildly when object rotates or camera moves
- Result: Low Re-ID success even with 0.4 cosine threshold

**Solution: V-JEPA2**
- 3D-aware video encoder from Meta
- Trained for prediction and robotics tasks
- Understands object permanence across viewpoints
- Lightweight ViT architecture (not as heavy as full video models)

**Two Implementation Options:**

**Option A: Single Best Frame (simpler, start here)**
```python
from transformers import AutoVideoProcessor, AutoModel
import torch

processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

def embed_object_crop(crop_image):
    """Embed single frame as 1-frame video"""
    # V-JEPA2 expects video format: T x C x H x W
    # For single image: T=1
    video = crop_image.unsqueeze(0)  # 1 x C x H x W
    inputs = processor(video, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Pool over patches
```

**Option B: Multi-Crop Video (better, implement after A works)**
```python
def embed_track_as_video(track_crops: list[np.ndarray], max_frames=16):
    """
    Combine multiple crops from a track into a mini-video.
    Even if noisy, V-JEPA2 handles temporal noise well.
    """
    # Sample evenly across track
    indices = np.linspace(0, len(track_crops)-1, min(len(track_crops), max_frames), dtype=int)
    frames = [track_crops[i] for i in indices]
    
    # Stack as video: T x C x H x W
    video = torch.stack([preprocess(f) for f in frames])
    
    inputs = processor(video, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use predictor output for better temporal understanding
    return outputs.predictor_output.last_hidden_state.mean(dim=(1,2))
```

**VideoMAE Alternative:**
```python
from transformers import VideoMAEImageProcessor, VideoMAEModel

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

# Similar usage, but VideoMAE is heavier and may need more frames
```

**Recommendation:** Start with V-JEPA2 Option A (single best frame per track), then upgrade to Option B (multi-crop video) once baseline works.

---

### 3. Semantic Filtering: FastVLM + all-mpnet-base-v2

**Pipeline:**
1. FastVLM describes each object crop: "A white ceramic coffee mug on a wooden table"
2. FastVLM describes scene: "A kitchen with a person cooking at the counter"
3. Sentence transformer embeds both descriptions
4. Cosine similarity between object description and scene context
5. Low similarity → object doesn't belong in scene → false positive

**Implementation:**
```python
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def validate_detection(object_desc: str, scene_desc: str, threshold=0.3) -> bool:
    """
    Check if object description is semantically compatible with scene.
    Low threshold because we're checking plausibility, not exact match.
    """
    embeddings = sentence_model.encode([object_desc, scene_desc])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity > threshold
```

**Why all-mpnet-base-v2?**
- 768-dim embeddings (richer than MiniLM's 384)
- Trained on 1B+ sentence pairs
- Best general-purpose sentence embedder
- Fast enough for real-time filtering

---

### 4. Causal Influence Scoring (CIS)

**Concept:** Understand how objects influence each other over time.

**Metrics:**
1. **Co-occurrence Frequency**: How often do objects A and B appear in the same frame?
2. **Motion Correlation**: When A moves, does B move too?
3. **Temporal Precedence**: Does A's state change predict B's state change?
4. **Spatial Proximity Over Time**: Do A and B maintain relative positions?

**Implementation:**
```python
@dataclass
class CausalInfluence:
    source_object_id: str
    target_object_id: str
    influence_type: str  # "co_occurs", "moves_with", "precedes", "causes"
    score: float  # 0-1
    evidence_frames: list[int]

def compute_cis(tracks: dict[str, list], window_size=10) -> list[CausalInfluence]:
    """
    Compute causal influence scores between all object pairs.
    """
    influences = []
    object_ids = list(tracks.keys())
    
    for i, obj_a in enumerate(object_ids):
        for obj_b in object_ids[i+1:]:
            # Co-occurrence
            frames_a = set(t['frame_id'] for t in tracks[obj_a])
            frames_b = set(t['frame_id'] for t in tracks[obj_b])
            co_occur = len(frames_a & frames_b) / len(frames_a | frames_b)
            
            if co_occur > 0.5:
                influences.append(CausalInfluence(
                    source_object_id=obj_a,
                    target_object_id=obj_b,
                    influence_type="co_occurs",
                    score=co_occur,
                    evidence_frames=list(frames_a & frames_b)[:10]
                ))
            
            # Motion correlation (compute bbox center deltas, correlate)
            # ... (more complex analysis)
    
    return influences
```

---

### 5. Memgraph Schema

**Docker Setup:**
```bash
docker run -d --name memgraph \
  -p 7687:7687 -p 7444:7444 \
  -v /path/to/data:/var/lib/memgraph \
  memgraph/memgraph-platform
```

**Schema (Cypher):**
```cypher
// Create constraints and indexes
CREATE CONSTRAINT ON (e:Episode) ASSERT e.id IS UNIQUE;
CREATE CONSTRAINT ON (m:MemoryObject) ASSERT m.id IS UNIQUE;
CREATE INDEX ON :Frame(timestamp);
CREATE INDEX ON :Observation(track_id);

// Example: Insert episode data
CREATE (e:Episode {
  id: "room_scan_001",
  video_path: "/data/videos/room_scan.mp4",
  duration_sec: 60.96,
  fps: 29.97,
  created_at: datetime()
})

// Insert memory object
CREATE (m:MemoryObject {
  id: "mem_001",
  canonical_label: "coffee_mug",
  description: "White ceramic mug with blue handle",
  first_seen_sec: 5.2,
  last_seen_sec: 45.8,
  total_observations: 127
})

// Link observations to memory objects
MATCH (m:MemoryObject {id: "mem_001"})
CREATE (t:Track {id: "track_3", observations: 127})
CREATE (m)-[:OBSERVED_AS]->(t)

// CIS relationship
MATCH (a:MemoryObject {id: "mem_001"}), (b:MemoryObject {id: "mem_002"})
CREATE (a)-[:CO_OCCURS {frequency: 0.85, evidence_frames: [10,20,30]}]->(b)
```

**Query Examples:**
```cypher
// What objects are always seen together?
MATCH (a:MemoryObject)-[r:CO_OCCURS]->(b:MemoryObject)
WHERE r.frequency > 0.8
RETURN a.canonical_label, b.canonical_label, r.frequency

// Object trajectory over time
MATCH (m:MemoryObject {id: $obj_id})-[:OBSERVED_AS]->(t:Track)
MATCH (o:Observation {track_id: t.id})
RETURN o.frame_id, o.bbox, o.confidence
ORDER BY o.frame_id

// Scene context at specific time
MATCH (f:Frame)
WHERE f.timestamp >= $start AND f.timestamp <= $end
RETURN f.timestamp, f.scene_caption
```

---

### 6. Ollama LLM Integration

**Model Selection for A10 (24GB VRAM):**
- `qwen2.5:14b-instruct-q8_0` (14GB) - Good balance
- `qwen2.5:32b-instruct-q4_K_M` (18GB) - Larger, quantized
- `deepseek-r1:14b` (14GB) - Strong reasoning
- `llama3.3:70b-instruct-q4_0` (40GB) - Won't fit, too large

**Recommended:** `qwen2.5:14b-instruct-q8_0`

**Setup:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5:14b-instruct-q8_0
```

**Query Pipeline:**
```python
import requests

def query_orion(question: str, context: dict) -> str:
    """
    Query Orion memory via LLM.
    """
    # 1. Parse question to extract entities/times
    # 2. Query Memgraph for relevant data
    # 3. Format context for LLM
    # 4. Generate response
    
    prompt = f"""You are Orion, a video understanding assistant. You have access to a memory graph of objects detected in video.

SCENE CONTEXT:
{context['scene_descriptions']}

OBJECTS IN MEMORY:
{context['memory_objects']}

RELATIONSHIPS:
{context['relationships']}

USER QUESTION: {question}

Provide a concise, accurate answer based only on the data above."""

    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'qwen2.5:14b-instruct-q8_0',
        'prompt': prompt,
        'stream': False
    })
    return response.json()['response']
```

---

## CLI Design: `orion` Commands

```bash
# Initialize a new episode
orion init --episode room_scan --video /path/to/video.mp4

# Run full pipeline
orion analyze --episode room_scan --fps 5 --model yoloworld

# Run specific stages
orion detect --episode room_scan    # Stage 1: Detection + Tracking
orion embed --episode room_scan     # Stage 2: V-JEPA2 Re-ID
orion filter --episode room_scan    # Stage 3: FastVLM filtering
orion graph --episode room_scan     # Stage 4: Scene graph + CIS
orion export --episode room_scan    # Stage 5: Memgraph export

# Query interface
orion query "What objects are in the kitchen?"
orion query --episode room_scan "When did the person pick up the mug?"

# Diagnostics
orion diagnose reid --episode room_scan --track-id 3
orion diagnose detection --episode room_scan --frame 100
orion status --episode room_scan
```

---

## Evaluation Framework

### Metrics

1. **Detection Quality**
   - Precision/Recall vs ground truth (manual annotation or Gemini)
   - False positive rate before/after FastVLM filtering

2. **Re-ID Accuracy** (THE KEY METRIC)
   - Intra-track similarity: Same object, different frames → should be high
   - Inter-track similarity: Different objects → should be low
   - Threshold ROC curve: Find optimal cosine threshold per object class

3. **Memory Coherence**
   - Number of memory objects vs expected (manual count)
   - Over-segmentation: One real object → multiple memory objects (bad)
   - Under-segmentation: Multiple real objects → one memory object (bad)

4. **Query Accuracy**
   - Manual Q&A evaluation: 100 questions, score 0-1 for correctness
   - Latency: Query response time

### Evaluation Script

```python
# scripts/eval_orion_v2.py

def evaluate_reid(episode: str):
    """
    For each memory object, compute:
    - Intra-object similarity (should be > 0.85)
    - Inter-object similarity (should be < 0.5)
    """
    memory = load_memory(episode)
    tracks = load_tracks(episode)
    
    intra_sims = []
    inter_sims = []
    
    for obj in memory['objects']:
        # Get embeddings for all observations of this object
        obj_embeddings = get_embeddings_for_object(obj['id'], tracks)
        
        # Intra-similarity: pairwise within object
        for i, e1 in enumerate(obj_embeddings):
            for e2 in obj_embeddings[i+1:]:
                intra_sims.append(cosine_similarity(e1, e2))
        
        # Inter-similarity: compare to other objects
        for other_obj in memory['objects']:
            if other_obj['id'] == obj['id']:
                continue
            other_embeddings = get_embeddings_for_object(other_obj['id'], tracks)
            for e1 in obj_embeddings[:5]:  # Sample
                for e2 in other_embeddings[:5]:
                    inter_sims.append(cosine_similarity(e1, e2))
    
    print(f"Intra-object similarity: mean={np.mean(intra_sims):.3f}, std={np.std(intra_sims):.3f}")
    print(f"Inter-object similarity: mean={np.mean(inter_sims):.3f}, std={np.std(inter_sims):.3f}")
    
    # Good Re-ID: intra >> inter with clear separation
    separation = np.mean(intra_sims) - np.mean(inter_sims)
    print(f"Separation (intra - inter): {separation:.3f} (want > 0.3)")
```

---

## Research Paper Integration

### Paper Title (Proposed)
"Orion: Memory-Centric Video Understanding with 3D-Aware Object Re-Identification and Causal Scene Graphs"

### Key Contributions
1. **V-JEPA2 for Video Re-ID**: First application of video prediction encoders for long-term object tracking
2. **CIS Framework**: Novel causal influence scoring for understanding object relationships over time
3. **VLM Filtering Pipeline**: Using lightweight VLMs as semantic validators for detection outputs
4. **Memgraph Integration**: Scalable graph-based video memory for natural language querying

### Experiments
1. **Re-ID Ablation**: CLIP vs DINO vs V-JEPA2 on same-object similarity
2. **Detection Comparison**: YOLO11x vs YOLO-World on open-vocabulary indoor scenes
3. **Filtering Effectiveness**: Detection F1 before/after FastVLM filtering
4. **Query Accuracy**: Human evaluation of Orion Q&A vs baseline (GPT-4V on raw video)

### Datasets
- ActionGenome (existing)
- Custom indoor activity videos
- EgoHOS (egocentric hand-object)

---

## Implementation Roadmap

### Phase 1: Foundation (Days 1-3)
- [ ] Set up Lambda environment (Python, dependencies)
- [ ] Implement YOLO-World detector backend
- [ ] Integrate V-JEPA2 embedder (Option A: single frame)
- [ ] Update CLI with new `orion` commands

### Phase 2: Filtering & Memory (Days 4-6)
- [ ] Integrate all-mpnet-base-v2 for semantic filtering
- [ ] Update FastVLM pipeline
- [ ] Set up Memgraph on Lambda
- [ ] Implement graph export

### Phase 3: CIS & Querying (Days 7-9)
- [ ] Implement CIS computation
- [ ] Add CIS edges to Memgraph
- [ ] Set up Ollama with Qwen2.5
- [ ] Build query pipeline

### Phase 4: Evaluation & Polish (Days 10-14)
- [ ] Run evaluation suite
- [ ] Tune Re-ID thresholds per class
- [ ] Fix edge cases
- [ ] Document for paper

---

## Quick Start (Lambda A10)

```bash
# 1. SSH into Lambda
ssh lambda-orion

# 2. Navigate to filesystem
cd /lambda/nfs/orion-core-fs/orion-research

# 3. Set up Python environment
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e .
pip install accelerate transformers sentence-transformers

# 4. Pull V-JEPA2 (will download on first use)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/vjepa2-vitl-fpc64-256')"

# 5. Set up Memgraph
docker run -d --name memgraph -p 7687:7687 memgraph/memgraph-platform

# 6. Set up Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5:14b-instruct-q8_0

# 7. Run Orion
orion init --episode test --video /path/to/video.mp4
orion analyze --episode test
orion query "What objects are in the video?"
```

---

## Summary of Changes from v1

| Component | v1 | v2 |
|-----------|----|----|
| Detection | YOLO11x (closed vocab) | YOLO-World v2 (open vocab) |
| Re-ID Embedder | CLIP/DINO (2D) | V-JEPA2 (3D video) |
| Depth | Depth Anything V3 | REMOVED |
| Sentence Embedding | all-MiniLM-L6-v2 | all-mpnet-base-v2 |
| Graph Database | None (JSON files) | Memgraph |
| LLM | None | Ollama (Qwen2.5-14B) |
| Causal Analysis | None | CIS (Causal Influence Scoring) |
| CLI | run_showcase, run_vlm_filter | orion init/analyze/query |

This architecture is designed to address the specific Re-ID failures identified by Shivank, while adding the semantic and reasoning capabilities needed for research paper contribution.
