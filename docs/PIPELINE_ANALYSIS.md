# Orion Pipeline Analysis & Issues

**Date**: October 18, 2025  
**Issue**: Contextual engine running forever on 445 objects

---

## 🔍 THE PROBLEM

Your contextual engine is **making ~191 separate LLM calls** (one per frame) to analyze **174 objects**, and each Ollama call to gemma3:4b takes **2-5 seconds**. This means:

```
191 frames × 3 seconds/call = ~10 minutes minimum
```

**But it's worse than that** because the LLM calls are **sequential** (not parallel), so if any call hangs or retries, the whole thing stalls.

---

## 📊 CURRENT PIPELINE FLOW

```
VIDEO (34 seconds, 1020 frames)
    ↓
┌─────────────────────────────────────┐
│ PART 1: SMART PERCEPTION            │
│ • Sample frames (fast/balanced/acc) │
│ • YOLO detection: 445 objects       │
│ • CLIP embeddings per detection     │
│ • HDBSCAN clustering → ~50 entities │
│ • FastVLM: Describe once per entity │
│ • Output: perception_log.json      │
│ Time: ~30-60 seconds ✅ EFFICIENT   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ CONTEXTUAL UNDERSTANDING (NEW)      │
│ • Read 445 objects from log         │
│ • Compute spatial zones (fast)      │
│ • Filter: 174 need LLM correction   │
│ • Group by frame: 191 unique frames │
│ • For each frame:                   │
│   - Build JSON prompt                │
│   - Call Ollama gemma3:4b           │
│   - Parse JSON response              │
│   - Apply corrections                │
│ Time: 191 × 3s = 10-15 min ❌ SLOW  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ PART 2: SEMANTIC UPLIFT             │
│ • Read enhanced perception log      │
│ • Build Neo4j knowledge graph       │
│ • Track entities, detect states     │
│ • Infer causal relationships        │
│ Time: ~20-40 seconds ✅ EFFICIENT   │
└─────────────────────────────────────┘
```

---

## 🐛 ROOT CAUSES

### 1. **Per-Frame LLM Batching is Still Too Granular**

**Current code** (contextual_engine.py line 283-323):
```python
def _batch_llm_analysis(self, entities: List[Dict]):
    # Group by frame
    frame_groups = defaultdict(list)
    for entity in entities:
        frame_groups[entity['frame']].append(entity)  # ← 191 unique frames
    
    # Process each frame
    for frame_num, frame_entities in frame_groups.items():  # ← 191 iterations!
        result = self._call_llm_batch(frame_entities)
        self.stats['llm_calls'] += 1  # Each call takes 2-5 seconds
```

**The issue**: You have **191 unique frames**, so you make **191 LLM calls**. Even though you batch objects *within* each frame, you're still calling the LLM hundreds of times.

### 2. **Overly Aggressive "needs_llm" Filter**

**Current logic** (contextual_engine.py line 256-280):
```python
def _needs_llm_analysis(self, entity: Dict) -> bool:
    # Known problematic YOLO classes
    problematic = {'hair drier', 'cell phone', 'remote', 'potted plant'}
    if yolo_class in problematic:
        return True  # ← Immediately triggers LLM
    
    # Low confidence or mismatch
    if confidence < 0.5 or (yolo_class not in description and confidence < 0.7):
        return True  # ← Catches ~174/445 objects
```

**Result**: 39% of your objects (174/445) are flagged for LLM correction.

### 3. **No Timeout or Async Handling**

If Ollama hangs on a single call, the whole pipeline freezes. You have **no timeout, no parallelization, no fallback**.

### 4. **Redundant Work**

You're analyzing objects that are **already tracked**. Your smart_perception already:
- Clustered 445 detections into ~50 entities
- Described each entity once with FastVLM

But then contextual_engine **ignores** the entity clustering and re-processes **all 445 individual detections**.

---

## 📈 WHAT'S GOOD

✅ **Smart Perception is Excellent**
- "Track first, describe once" is brilliant
- 445 detections → 50 entities = **9x efficiency**
- Using CLIP + HDBSCAN clustering is smart
- FastVLM descriptions are high quality

✅ **Spatial Zone Detection**
- The heuristic spatial zone calculation is fast and accurate
- No LLM needed for this

✅ **Semantic Uplift is Solid**
- Neo4j knowledge graph structure is well-designed
- Entity tracking and state change detection work well
- Causal inference engine is sophisticated

---

## 🔧 WHAT TO FIX

### Immediate Fix (5 minutes): **Disable LLM Correction Entirely**

The fastest way to unblock your pipeline:

```python
# In contextual_engine.py, line 91-100
entities_needing_llm = [e for e in entities if e.get('needs_llm')]
if entities_needing_llm:
    # TEMPORARY: Skip LLM correction
    logger.warning(f"Skipping LLM correction for {len(entities_needing_llm)} objects (disabled)")
    # self._batch_llm_analysis(entities_needing_llm)  # ← Comment this out
```

**Result**: Pipeline completes in ~1 minute instead of 10-15 minutes.

---

### Short-Term Fix (30 minutes): **Smarter Batching**

Instead of per-frame batching, batch **across all frames**:

```python
def _batch_llm_analysis(self, entities: List[Dict]):
    """Single LLM call for ALL entities needing correction"""
    if not self.model_manager or len(entities) == 0:
        return
    
    MAX_BATCH_SIZE = 50  # Gemma can handle ~50 objects at once
    
    # Process in chunks of 50
    for i in range(0, len(entities), MAX_BATCH_SIZE):
        batch = entities[i:i + MAX_BATCH_SIZE]
        logger.info(f"Processing batch {i//MAX_BATCH_SIZE + 1}: {len(batch)} objects")
        
        try:
            result = self._call_llm_batch(batch)
            self.stats['llm_calls'] += 1
            # Apply results...
        except Exception as e:
            logger.warning(f"Batch {i//MAX_BATCH_SIZE + 1} failed: {e}, skipping")
```

**Result**: 174 objects → 4 batches × 3 seconds = **12 seconds** instead of 10 minutes.

---

### Medium-Term Fix (2 hours): **Work at Entity Level, Not Detection Level**

The contextual engine should process **entities** (the 50 unique objects), not **detections** (the 445 observations).

**Why**: You've already done the hard work of clustering! Don't throw it away.

```python
# In contextual_engine.py
def process(self, entities: List[Dict], observations: List[Dict]):
    """Process entities (clustered), not individual observations"""
    
    # 1. Analyze the ~50 entities
    for entity in entities:
        entity['spatial_zone'] = self._infer_entity_zone(entity)
        entity['needs_correction'] = self._check_if_problematic(entity)
    
    # 2. Single LLM call for all problematic entities
    needs_llm = [e for e in entities if e['needs_correction']]
    if needs_llm:
        self._correct_entity_batch(needs_llm)  # One call!
    
    # 3. Propagate corrections to observations
    for obs in observations:
        entity = find_entity_for_observation(obs)
        obs['spatial_zone'] = entity['spatial_zone']
        obs['object_class'] = entity.get('corrected_class', entity['class'])
```

**Result**: ~50 entities in a single LLM call = **3-5 seconds total**.

---

### Long-Term Fix (1 day): **Async LLM with Timeout**

Use Python's `asyncio` to parallelize LLM calls:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError

async def _call_llm_with_timeout(self, entities, timeout=10):
    """Call LLM with timeout"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(pool, self._call_llm_batch, entities),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"LLM call timed out after {timeout}s, using fallback")
            return {'objects': [{'index': i, 'needs_correction': False} 
                               for i in range(len(entities))]}
```

---

## 🎯 RECOMMENDED SOLUTION

**Do this NOW** (literally 2 lines of code):

1. **Comment out the LLM calls** in contextual_engine.py line 98:
   ```python
   # self._batch_llm_analysis(entities_needing_llm)
   ```

2. **Test your pipeline** to verify it completes quickly

3. **Then** implement the "single batch for all entities" approach

---

## 📊 PERFORMANCE COMPARISON

| Approach | LLM Calls | Time | Status |
|----------|-----------|------|--------|
| **Current (per-frame)** | 191 | 10-15 min | ❌ Too slow |
| **Disabled** | 0 | ~1 min | ✅ Works, no correction |
| **All-at-once batch** | 4 | ~12 sec | ✅ Good compromise |
| **Entity-level** | 1 | ~5 sec | ✅ Optimal |
| **Async + timeout** | 4-8 (parallel) | ~4 sec | ✅ Best |

---

## 🤔 KEY QUESTIONS

### Q: Do we even need LLM correction?

**A**: Probably not! Your FastVLM descriptions are already pretty accurate. The main corrections are:
- `hair drier` → `door_knob` (YOLO bug)
- `cell phone` → `remote` (similar objects)

You could **hardcode these rules** instead of using an LLM:

```python
YOLO_CORRECTIONS = {
    'hair drier': lambda desc: 'door_knob' if 'knob' in desc or 'handle' in desc else 'hair_dryer',
    'cell phone': lambda desc: 'remote' if 'remote' in desc else 'cell_phone',
}
```

### Q: Why not correct during perception?

**A**: You could! If YOLO says "hair drier" but FastVLM says "metal knob on door", just correct it immediately and save the corrected class.

---

## 🚀 NEXT STEPS

1. **Immediately**: Comment out `_batch_llm_analysis()` call to unblock pipeline
2. **Today**: Implement single-batch or hardcoded correction rules
3. **This week**: Refactor to work at entity level, not detection level
4. **Future**: Add async LLM calls with timeout for robustness

---

## 📝 CODE LOCATIONS

| Component | File | Line |
|-----------|------|------|
| Contextual engine | `orion/contextual_engine.py` | 35-418 |
| Batch LLM calls | Line 283-323 | The bottleneck |
| needs_llm filter | Line 256-280 | Too aggressive |
| Smart perception | `orion/smart_perception.py` | Already efficient ✅ |
| Semantic uplift | `orion/semantic_uplift.py` | Already efficient ✅ |

---

## 💡 THE BIG PICTURE

Your pipeline is **80% excellent**:
- ✅ Smart perception is brilliant
- ✅ Knowledge graph is well-designed  
- ✅ Spatial zones work great

The **20% problem** is this new contextual LLM correction step that:
- Undoes your entity clustering efficiency
- Makes 191 sequential Ollama calls
- Has no timeout or fallback
- Might not even be necessary

**Solution**: Skip it for now, then implement it smarter (single batch or entity-level).
