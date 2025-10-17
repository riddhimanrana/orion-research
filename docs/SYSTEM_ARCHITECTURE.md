# Orion Enhanced System Architecture

## Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORION VIDEO UNDERSTANDING SYSTEM                     │
└─────────────────────────────────────────────────────────────────────────────┘

                                    INPUT
                                   video.mp4
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: TRACKING ENGINE (tracking_engine.py)                              │
│  ────────────────────────────────────────────────────────────────────────   │
│                                                                               │
│  ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │ ObservationCollector  │ │  EntityTracker    │     │ SmartDescriber   │    │
│  │  - YOLO11x: detect   │ │  - CLIP: embed    │     │  - FastVLM: describe│ │
│  │  - Per-frame objects │ │  - HDBSCAN: cluster│    │  - State changes  │    │
│  │  - 445 observations  │ │  - 21 entities     │    │  - 13 descriptions│    │
│  └─────────────────┘     └──────────────────┘     └──────────────────┘    │
│                                                                               │
│  Output: tracking_results.json (21 entities, 445 observations, 21x efficiency)│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: ENHANCED KNOWLEDGE GRAPH (enhanced_knowledge_graph.py)            │
│  ────────────────────────────────────────────────────────────────────────   │
│                                                                               │
│  ┌────────────────────┐  ┌─────────────────┐  ┌──────────────────────┐    │
│  │ SceneClassifier    │  │ SpatialAnalyzer │  │ CausalReasoningEngine│    │
│  │  - Pattern matching│  │  - Co-occurrence│  │  - Temporal ordering │    │
│  │  - 8 scene types   │  │  - 9 rel types  │  │  - Spatial proximity │    │
│  │  - Confidence      │  │  - Distance calc│  │  - Semantic scoring  │    │
│  └────────────────────┘  └─────────────────┘  └──────────────────────┘    │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ ContextualEmbeddingGenerator                                       │    │
│  │  - Visual (CLIP) + Textual (scene + spatial + relationships)      │    │
│  │  - 0.6 * visual + 0.4 * textual context                           │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  Output: Neo4j Graph (21 entities, 9 scenes, 24 spatial rels, 8 transitions)│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: ENHANCED VIDEO QA (enhanced_video_qa.py)                          │
│  ────────────────────────────────────────────────────────────────────────   │
│                                                                               │
│  User Question: "What type of room is this?"                                 │
│         │                                                                     │
│         ▼                                                                     │
│  ┌────────────────────┐                                                      │
│  │ Question Classifier│  →  "scene" type detected                           │
│  └────────────────────┘                                                      │
│         │                                                                     │
│         ▼                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │ Context Retrieval (targeted by question type)                  │         │
│  │  - Spatial: Get SPATIAL_REL relationships                       │         │
│  │  - Scene: Get Scene nodes with types and objects               │         │
│  │  - Temporal: Get Scene timeline with TRANSITIONS_TO            │         │
│  │  - Causal: Get POTENTIALLY_CAUSED relationships                │         │
│  │  - Entity: Get Entity details with APPEARS_IN scenes           │         │
│  └────────────────────────────────────────────────────────────────┘         │
│         │                                                                     │
│         ▼                                                                     │
│  ┌────────────────────┐                                                      │
│  │ LLM Answer Gen     │  →  Gemma 3 4B (Ollama)                             │
│  │  with rich context │     ~3s latency                                      │
│  └────────────────────┘                                                      │
│         │                                                                     │
│         ▼                                                                     │
│  Answer: "The video contains Workspace, Bedroom, and Unknown scenes..."      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MODEL MANAGER (Singleton Pattern)                                          │
│  ────────────────────────────────────────────────────────────────────────   │
│                                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   YOLO11x        │  │   CLIP ViT-B/32  │  │   FastVLM 0.5B   │          │
│  │                  │  │                  │  │                  │          │
│  │  56.9M params    │  │  512-dim embed   │  │  MLX + CoreML    │          │
│  │  80 COCO classes │  │  Vision + Text   │  │  FastViTHD       │          │
│  │  Detection       │  │  Re-ID embeddings│  │  Rich descriptions│         │
│  │                  │  │  Multimodal      │  │                  │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│         ▲                      ▲                      ▲                      │
│         │                      │                      │                      │
│         └──────────────────────┴──────────────────────┘                     │
│                        Lazy Loading                                          │
│                   (7.5x faster startup)                                      │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │  Gemma 3 4B (via Ollama) - External LLM for Q&A                  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Neo4j Knowledge Graph Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  KNOWLEDGE GRAPH NODES & RELATIONSHIPS                                       │
└─────────────────────────────────────────────────────────────────────────────┘

    (Entity)                      (Scene)
    ┌──────────────┐             ┌──────────────┐
    │ id           │             │ id           │
    │ class        │─────┐       │ scene_type   │
    │ description  │     │       │ confidence   │
    │ appearance_ct│     │       │ frame_range  │
    │ first_seen   │     │       │ timestamps   │
    │ last_seen    │     │       │ dominant_objs│
    │ spatial_zone │     │       │ description  │
    │ scene_types  │     │       └──────────────┘
    └──────────────┘     │              │
           │             │              │
           │             │              │
           │    ┌────────▼──────┐       │
           │    │  APPEARS_IN   │       │
           │    └───────────────┘       │
           │                            │
    ┌──────▼────────┐            ┌─────▼──────────┐
    │  SPATIAL_REL  │            │ TRANSITIONS_TO │
    │  - type       │            │  - frame_gap   │
    │  - confidence │            └────────────────┘
    │  - co_occur   │
    │  - avg_dist   │
    └───────────────┘
           │
    ┌──────▼────────────┐
    │ POTENTIALLY_CAUSED│
    │  - temporal_gap   │
    │  - spatial_prox   │
    │  - confidence     │
    │  - cause_state    │
    │  - effect_state   │
    └───────────────────┘
```

## Data Flow

```
┌──────────┐
│  Video   │
│  Frame   │
└────┬─────┘
     │
     ▼
┌──────────────────┐
│ YOLO Detection   │  → bbox, class, confidence
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│ CLIP Embedding   │  → 512-dim visual vector
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│ HDBSCAN Cluster  │  → entity_id (object permanence)
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│ FastVLM Describe │  → rich text description
└────┬─────────────┘
     │
     ├─────────────────────┐
     │                     │
     ▼                     ▼
┌──────────────┐    ┌─────────────────┐
│ Entity Node  │    │ Observation     │
│ (persistent) │    │ (per-frame)     │
└──────┬───────┘    └────────┬────────┘
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
         ┌────────────────┐
         │ Scene Analysis │
         │  - Classify    │
         │  - Spatial     │
         │  - Causal      │
         └────┬───────────┘
              │
              ▼
         ┌────────────┐
         │  Neo4j KG  │
         └────┬───────┘
              │
              ▼
         ┌────────────┐
         │  Q&A with  │
         │  Gemma 3   │
         └────────────┘
```

## Scene Classification Logic

```
Object Patterns → Scene Type
───────────────────────────────────────────────────────────

keyboard, mouse, monitor  →  office (confidence: 0.7+)
oven, refrigerator, sink  →  kitchen (confidence: 0.8+)
bed, clock               →  bedroom (confidence: 0.7+)
couch, tv                →  living_room (confidence: 0.6+)
laptop, keyboard         →  workspace (confidence: 0.6+)
tree, car                →  outdoor (confidence: 0.5+)

Scoring:
confidence = (required_match * 0.7 + common_match * 0.3) * scene_weight
```

## Spatial Relationship Detection

```
Distance Thresholds:
───────────────────────────────────────────────────

< 15% frame diagonal  →  very_near (confidence: 0.9)
< 30% frame diagonal  →  near      (confidence: 0.7)
< 50% frame diagonal  →  same_region (confidence: 0.5)

Position Analysis:
───────────────────────────────────────────────────

y2_a < y1_b - 10px   →  above     (confidence: 0.9)
y2_b < y1_a - 10px   →  below     (confidence: 0.9)
x2_a < x1_b - 10px   →  left_of   (confidence: 0.85)
x2_b < x1_a - 10px   →  right_of  (confidence: 0.85)

Containment:
───────────────────────────────────────────────────

bbox_a contains bbox_b  →  contains  (confidence: 0.95)
bbox_b contains bbox_a  →  inside    (confidence: 0.95)
```

## Causal Reasoning Pipeline

```
State Changes → Potential Causes
───────────────────────────────────────────────────────────

1. Temporal Constraint:
   cause_time < effect_time
   gap < 5.0 seconds

2. Spatial Constraint:
   entities must be co-located
   proximity score > 0.3

3. Semantic Plausibility:
   agents (person, dog, car) → high (0.9)
   static objects            → low (0.3)

4. Confidence Score:
   confidence = 0.3*temporal + 0.4*spatial + 0.3*semantic
   threshold = 0.5

Result: POTENTIALLY_CAUSED relationship
```

## Question Answering Flow

```
User Question
     │
     ▼
┌────────────────────┐
│ Classify Question  │
│  Keywords:         │
│  - "where" → spatial
│  - "room"  → scene
│  - "when"  → temporal
│  - "why"   → causal
│  - "what"  → entity
└────┬───────────────┘
     │
     ▼
┌────────────────────┐
│ Retrieve Context   │
│  (type-specific)   │
│                    │
│  Spatial → SPATIAL_REL
│  Scene   → Scene nodes
│  Temporal→ TRANSITIONS_TO
│  Causal  → POTENTIALLY_CAUSED
│  Entity  → Entity + APPEARS_IN
└────┬───────────────┘
     │
     ▼
┌────────────────────┐
│ Generate Answer    │
│  LLM: Gemma 3 4B   │
│  Context: Graph    │
│  Latency: ~3s      │
└────┬───────────────┘
     │
     ▼
  Natural Language
     Answer
```

## Performance Metrics

```
TRACKING ENGINE:
─────────────────────────────────────
Input:         1978 frames @ 30fps
Observations:  445 detected
Entities:      21 unique (HDBSCAN)
Descriptions:  13 generated (8 skipped)
Efficiency:    21.2x reduction
Total time:    192.5s (~8 fps)

KNOWLEDGE GRAPH:
─────────────────────────────────────
Entities:      21 nodes
Scenes:        9 nodes (4 bedroom, 2 workspace, 3 unknown)
Spatial Rels:  24 relationships
Causal Chains: 0 (static video)
Transitions:   8 scene changes
Build time:    ~2s

VIDEO QA:
─────────────────────────────────────
Question Classification: <10ms
Context Retrieval:       10-50ms
LLM Generation:          2-5s
Total latency:           ~3s avg
Accuracy:                High (grounded in graph)
```

## Component Integration

```
┌─────────────────────────────────────────────────────────────┐
│  UNIFIED ARCHITECTURE                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  OrionConfig     │◀────────│  ModelManager    │         │
│  │  (Centralized)   │         │  (Singleton)     │         │
│  └──────────────────┘         └──────────────────┘         │
│         │                              │                    │
│         │                              │                    │
│         ▼                              ▼                    │
│  ┌──────────────────────────────────────────────┐          │
│  │  TrackingEngine                              │          │
│  │   ├── ObservationCollector                   │          │
│  │   ├── EntityTracker                          │          │
│  │   └── SmartDescriber                         │          │
│  └──────────────┬───────────────────────────────┘          │
│                 │                                           │
│                 ▼                                           │
│  ┌──────────────────────────────────────────────┐          │
│  │  EnhancedKnowledgeGraphBuilder               │          │
│  │   ├── SceneClassifier                        │          │
│  │   ├── SpatialAnalyzer                        │          │
│  │   ├── ContextualEmbeddingGenerator           │          │
│  │   └── CausalReasoningEngine                  │          │
│  └──────────────┬───────────────────────────────┘          │
│                 │                                           │
│                 ▼                                           │
│  ┌──────────────────────────────────────────────┐          │
│  │  EnhancedVideoQASystem                       │          │
│  │   ├── Question Classifier                    │          │
│  │   ├── Context Retriever                      │          │
│  │   └── LLM Answer Generator                   │          │
│  └──────────────────────────────────────────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Benefits:
- Shared ModelManager → 19x memory efficiency
- Unified Config → Easy tuning
- Lazy Loading → 7.5x faster startup
- Modular Design → Independent testing
```

---

**Legend:**
- 📦 Component
- → Data flow
- ◀─ Configuration/dependency
- ├── Subcomponent
- └── Terminal subcomponent
