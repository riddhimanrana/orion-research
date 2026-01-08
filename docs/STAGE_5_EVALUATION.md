# Stage 5 Evaluation Results

## Summary (January 2026)

Stage 5 implements the full Orion pipeline with Memgraph RAG integration:

```
Video → Detection (YOLO11m) → Tracking → Re-ID (V-JEPA2) → Memgraph → RAG Queries
```

### Evaluation Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **F1 Score** | 49-59% | Variable based on Gemini evaluation |
| **Precision** | 43-80% | Better with YOLO11m vs YOLO-World |
| **Recall** | 47-57% | Limited by COCO vocabulary |
| **Processing Time** | ~30s/video | 66s video on A10 GPU |

### Detection Quality

**Correct Detections:**
- ✓ laptop, keyboard, mouse, person, book, cell phone, bed, sink

**False Positives (to improve):**
- ✗ remote (AirPods case misidentified)
- ✗ suitcase (bedsheets misidentified)

**Missed Objects (COCO vocabulary limitation):**
- ! monitor, desk, chair, door, bookshelf, staircase

### Memgraph Integration

Successfully stores:
- 169 frames
- 10 unique entities
- 142 spatial relationships (NEAR, ON, HELD_BY)
- 870 observations with bboxes

### RAG Query Capabilities

Tested queries:
1. **Object listing:** "What objects are in the video?" → Found 10 object types
2. **Interactions:** "What did the person interact with?" → book, laptop, keyboard held by person
3. **Location:** "Where did the book appear?" → 94 appearances, 23.8s to 65.9s
4. **Object info:** "Tell me about the laptop" → 69 detections, 0.5s to 29.7s

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORION v2 PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│  Input: video.mp4 (66s, 1978 frames)                           │
│                                                                 │
│  Phase 1: Detection + Tracking (9s)                            │
│    ├── YOLO11m detector (80 COCO classes)                      │
│    ├── EnhancedTracker (IoU + appearance matching)             │
│    └── Output: 870 track observations, 21 unique tracks        │
│                                                                 │
│  Phase 2: Re-ID + Memory (16s)                                 │
│    ├── V-JEPA2 embeddings (3D-aware video encoder)             │
│    ├── Cosine similarity clustering                            │
│    └── Output: 10 memory objects, reid_clusters.json           │
│                                                                 │
│  Phase 3: Scene Graph                                          │
│    ├── Spatial relations: NEAR, ON, HELD_BY                    │
│    └── Output: 169 frames, 142 edges                           │
│                                                                 │
│  Phase 4: Memgraph Ingest                                      │
│    ├── Entity nodes with embeddings                            │
│    ├── Frame nodes with timestamps                             │
│    ├── OBSERVED_IN relationships                               │
│    └── Spatial relationships (NEAR, ON, HELD_BY)               │
│                                                                 │
│  Phase 5: RAG Queries                                          │
│    ├── Object queries: "What objects are in the video?"        │
│    ├── Spatial queries: "What was near the laptop?"            │
│    ├── Temporal queries: "What happened at 25s?"               │
│    └── Interaction queries: "What did the person hold?"        │
└─────────────────────────────────────────────────────────────────┘
```

### Known Limitations

1. **COCO Vocabulary:** YOLO11m only detects 80 COCO classes
   - Cannot detect: monitor, desk, chair, door, bookshelf, staircase
   - Solution: Use YOLO-World with semantic filtering (lower precision)

2. **False Positives:** Some object hallucinations persist
   - remote (AirPods case), suitcase (bedsheets)
   - Solution: Tune confidence thresholds per class

3. **Temporal Consistency:** Objects persist after leaving frame
   - High detection count for objects no longer visible
   - Solution: Implement temporal decay in tracker

### Commands

```bash
# Run full pipeline with Memgraph
python -m orion.cli.run_showcase --episode test --video video.mp4 --memgraph

# Run RAG queries
python -m orion.query.rag -q "What objects are in the video?"

# Run Gemini evaluation
python scripts/eval_video_gemini.py --video video.mp4 --episode eval_001
```

### Next Steps

1. **Improve recall:** Add more classes via YOLO-World + semantic filtering
2. **Reduce false positives:** Tune confidence thresholds per class
3. **Vector search:** Implement embedding-based similarity queries
4. **Multi-video support:** Query across multiple episodes
5. **LLM integration:** Natural language query parsing with Gemini/GPT
