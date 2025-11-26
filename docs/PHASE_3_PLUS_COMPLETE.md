# Phase 3+ Complete: Events, Relations, and Scene Graphs

## What We Built

### 1. Advanced Event Detection (`orion/perception/memory/events.py`)

**Debounced State Changes**
- `build_state_events()`: Tracks `held_by_person` state with configurable debounce window
- Suppresses single-frame noise by requiring N consecutive frames before emitting transition
- Default debounce: 2 frames

**Spatial Relations**
- `build_relation_events()`: Detects `near`, `on`, `held_by` relations between memory objects
- Debounced per-frame transitions to avoid flicker
- Relations:
  - `near`: Objects with low normalized centroid distance, minimal overlap
  - `on`: Subject above object with horizontal overlap and small vertical gap
  - `held_by`: Non-person object overlapping or inside person bbox

**Split Detection**
- `build_split_events()`: Identifies when single object likely became multiple memories
- Uses embedding similarity + temporal adjacency + spatial proximity
- Generates candidate split events for manual review

**Merge Suggestions**
- `build_merge_suggestions()`: Ranked list of memory pairs likely to be same object
- Score combines similarity (60%), temporal proximity (20%), spatial proximity (20%)
- Outputs top-K suggestions with detailed metrics

### 2. Scene Graph Generation (`orion/perception/graph/scene_graph.py`)

**Per-Frame Graphs**
- `build_scene_graphs()`: Generates per-frame snapshots with nodes (objects) and edges (relations)
- Each graph contains: frame_id, nodes list, edges list
- Saved as JSONL for efficient streaming

**Summary Statistics**
- `build_graph_summary()`: Aggregate stats across all frames
- Tracks: total frames, node/edge counts, relation distribution, averages

### 3. CLI Tools

**`orion/cli/run_memory.py`** - Enhanced event generation
```bash
python -m orion.cli.run_memory --results results/video_validation \
  --relations near on held_by \
  --debounce 2 \
  --near-dist 0.08 \
  --on-overlap 0.3 \
  --held-iou 0.3
```

**`orion/cli/run_scene_graph.py`** - Scene graph generation
```bash
python -m orion.cli.run_scene_graph --results results/video_validation \
  --relations near on held_by \
  --near-dist 0.08 \
  --on-overlap 0.3 \
  --held-iou 0.3
```

### 4. Validation & Export Scripts

**`scripts/visualize_scene_graph.py`** - Show detailed metrics
```bash
python scripts/visualize_scene_graph.py --results results/video_validation --num-samples 8
```

**`scripts/export_graph_samples.py`** - Extract annotated frames
```bash
python scripts/export_graph_samples.py \
  --results results/video_validation \
  --video data/examples/video.mp4 \
  --output results/video_validation/graph_samples \
  --max-samples 8
```

**`scripts/prepare_gemini_validation.py`** - Generate validation prompt
```bash
python scripts/prepare_gemini_validation.py \
  --samples results/video_validation/graph_samples \
  --output results/video_validation/gemini_validation_prompt.txt
```

## Current Results (video_validation)

### Scene Graph Summary
- 197 frames with detections
- 12 frames with relations (6%)
- Avg 1.86 nodes/frame, 0.08 edges/frame
- Relations: 4 near, 6 held_by, 5 on

### Events Summary
- 50 total events (42 lifecycle + 8 relations)
- 0 state_change events with debounce=2
- 8 relation_change events
- 0 split events detected
- 1 merge suggestion (2 TV memories with sim=0.88)

## Validation Checklist

### Ready for Gemini Review
1. ✅ 8 annotated sample frames exported to `results/video_validation/graph_samples/`
2. ✅ Detailed text reports with bbox coordinates and metrics
3. ✅ Validation prompt generated in `gemini_validation_prompt.txt`
4. ⏳ **Next**: Upload JPGs + prompt to Gemini for ground truth validation

### Identified Issues
1. **held_by overfires on large furniture**
   - Frame 710: "bed held_by person" (IoU 0.30)
   - Should exclude large/static objects (bed, table, couch)

2. **on relation may be inverted**
   - Frame 1075: "bed on book" detected
   - Semantically should be "book on bed"
   - Need directionality constraint (smaller/higher object as subject)

3. **Low relation density**
   - Only 6% of frames have relations
   - May need looser thresholds or more relation types

## Tuning Parameters

### Current Defaults
```python
# State
debounce_window = 2        # frames
held_by_iou = 0.3          # IoU threshold

# Relations  
near_dist_norm = 0.08      # normalized centroid distance
on_h_overlap = 0.3         # horizontal overlap ratio
on_vgap_norm = 0.02        # vertical gap (normalized)
iou_exclude = 0.1          # exclude overlaps from 'near'
```

### Recommended Adjustments (Post-Gemini)

**Option A: Stricter (Reduce False Positives)**
```bash
--held-iou 0.5 --near-dist 0.05 --on-overlap 0.5
```

**Option B: Looser (Increase Recall)**
```bash
--held-iou 0.2 --near-dist 0.12 --on-overlap 0.2
```

**Option C: Add Class Constraints**
- Modify `build_scene_graphs()` to exclude furniture from held_by subject
- Use object size to determine on directionality
- Add semantic rules (e.g., "book can be on bed" but not vice versa)

## Next Steps

### Immediate (Before Phase 4)
1. **Gemini validation**: Upload samples + prompt, collect feedback
2. **Tune thresholds**: Based on false positive/negative analysis
3. **Add class constraints**: Implement semantic filtering in scene_graph.py
4. **Re-run pipeline**: Regenerate events and graphs with tuned params
5. **Document final thresholds**: Update config with validated values

### Phase 4: Query & Reasoning
Once relations are validated:
- Build temporal query engine over events + scene graphs
- Natural language interface to memory (e.g., "when was the book held?")
- Spatial reasoning (e.g., "what objects were near the person?")
- Event sequence queries (e.g., "show all appeared→held_by→disappeared chains")

## Generated Files

### Per Episode (e.g., results/video_validation/)
- `tracks.jsonl` - Frame-by-frame detections with track IDs
- `memory.json` - Persistent object memories with embeddings
- `events.jsonl` - Lifecycle + state + relation change events
- `scene_graph.jsonl` - Per-frame spatial graphs
- `graph_summary.json` - Aggregate statistics
- `merge_suggestions.json` - Ranked memory merge candidates
- `graph_samples/` - Exported annotated frames for validation

### Validation Outputs
- `SCENE_GRAPH_VALIDATION.md` - Analysis of detected relations
- `gemini_validation_prompt.txt` - Ready-to-use prompt for Gemini

## Testing Status

✅ All tests passing (6/6):
- `test_events.py`: Lifecycle events
- `test_events_heuristics.py`: State changes + split detection
- `test_relations_and_suggestions.py`: Debounced relations + merge scoring

## How to Reproduce

```bash
# Full pipeline for one episode
python -m orion.cli.run_tracks --episode test_validation --video data/examples/test.mp4
python -m orion.cli.run_reid --episode test_validation --video data/examples/test.mp4
python -m orion.cli.run_memory --results results/test_validation --debounce 2
python -m orion.cli.run_scene_graph --results results/test_validation

# Export samples for validation
python scripts/export_graph_samples.py \
  --results results/test_validation \
  --video data/examples/test.mp4 \
  --output results/test_validation/graph_samples

# Generate Gemini prompt
python scripts/prepare_gemini_validation.py \
  --samples results/test_validation/graph_samples \
  --output results/test_validation/gemini_prompt.txt
```
