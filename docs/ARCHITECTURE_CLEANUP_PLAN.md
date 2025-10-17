# Orion Architecture Cleanup Plan

## Current Problem

The current architecture uses too many overlapping models:

```
❌ MESSY CURRENT STACK
├── YOLO11m (detection)
├── ResNet50 (re-ID) → Actually CLIP now
├── FastVLM (descriptions)
│   └── FastViTHD (vision backbone)
├── Gemma3:4b (Q&A)
└── "EmbeddingGemma" → Actually CLIP (confusing!)
```

**Issues:**
1. Multiple vision encoders (FastViTHD + ResNet50/CLIP)
2. Confusing naming ("EmbeddingGemma" is actually CLIP)
3. Redundant computations
4. Hard to maintain

---

## Recommended Architecture

### Clean, Minimal Stack

```
✅ CLEAN PROPOSED STACK

1. YOLO11x (Detection)
   └── "What is in the frame and where?"
   
2. CLIP ViT-B/32 (Unified Vision Encoder)
   └── Embeddings for re-identification
   └── Vision+text verification
   └── Semantic similarity
   
3. FastVLM (Descriptions)
   └── "Describe this object in detail"
   └── Used ONLY when needed
   
4. Gemma3:4b (Q&A)
   └── Query knowledge graph
   └── Generate insights
```

---

## Implementation Plan

### Phase 1: Rename & Clarify (Immediate)

**Goal:** Make it clear we're using CLIP, not EmbeddingGemma

**Changes:**

1. **Rename the file:**
   ```bash
   mv src/orion/backends/embedding_gemma.py \
      src/orion/backends/clip_backend.py
   ```

2. **Rename the class:**
   ```python
   # OLD
   class EmbeddingGemmaVision:
       pass
   
   # NEW
   class CLIPEmbedder:
       """
       OpenAI CLIP for multimodal vision+text embeddings.
       Used for object re-identification and semantic verification.
       """
       pass
   ```

3. **Update config:**
   ```python
   # In tracking_engine.py Config class
   
   # OLD
   EMBEDDING_MODEL = 'embedding-gemma'
   
   # NEW
   EMBEDDING_MODEL = 'clip'  # or 'clip-vit-base-patch32'
   ```

4. **Update imports everywhere:**
   ```python
   # OLD
   from orion.backends.embedding_gemma import get_embedding_gemma
   
   # NEW
   from orion.backends.clip_backend import get_clip_embedder
   ```

---

### Phase 2: Consolidate Vision Encoders (Optional)

**Goal:** Remove ResNet50 completely, use only CLIP

**Status:** ✅ Already done! You're using CLIP now.

**Verify by checking:**
```bash
grep -r "resnet50" src/orion/
# Should return minimal/no results
```

---

### Phase 3: Optimize Model Loading (Future)

**Goal:** Load models on-demand, share weights

**Current:**
```python
# Every component loads its own model
yolo = YOLO("yolo11x.pt")          # 120MB
clip = CLIPModel(...)               # 600MB
fastvlm = FastVLM(...)              # 1.2GB
gemma = Ollama("gemma3:4b")         # 4GB+ (external)
```

**Better:**
```python
class ModelManager:
    """Singleton manager for all models"""
    _clip: Optional[CLIPModel] = None
    _yolo: Optional[YOLO] = None
    _fastvlm: Optional[FastVLM] = None
    
    def get_clip(self):
        if self._clip is None:
            self._clip = CLIPModel.from_pretrained(...)
        return self._clip
```

---

### Phase 4: Remove Redundant Models (Future)

**Consider removing:**

1. **ResNet50** → ✅ Already replaced with CLIP
2. **Separate embedding model** → Use CLIP image encoder
3. **Multiple Gemma models** → Just use Gemma3:4b for everything

---

## File Renaming Checklist

### Files to Rename

- [ ] `src/orion/backends/embedding_gemma.py` → `clip_backend.py`
- [ ] Class `EmbeddingGemmaVision` → `CLIPEmbedder`
- [ ] Function `get_embedding_gemma()` → `get_clip_embedder()`

### Files to Update (imports)

- [ ] `src/orion/tracking_engine.py`
- [ ] `src/orion/video_qa.py` (if used)
- [ ] `tests/*` (any tests)
- [ ] `docs/EMBEDDING_GEMMA_*.md` → Rename to `CLIP_*.md`

---

## Configuration Cleanup

### Before (Confusing)

```python
# Config in tracking_engine.py
EMBEDDING_MODEL = 'embedding-gemma'  # Actually CLIP!
USE_MULTIMODAL_EMBEDDINGS = True     # What does this mean?
EMBEDDING_DIM = 512                  # For what model?
```

### After (Clear)

```python
# Config in tracking_engine.py
VISION_ENCODER = 'clip'              # Or 'clip-vit-base-patch32'
CLIP_MODEL = 'openai/clip-vit-base-patch32'
USE_TEXT_CONDITIONING = True         # Use CLIP's text encoder
EMBEDDING_DIM = 512                  # CLIP ViT-B/32 dimension
```

---

## Benefits of Cleanup

### Before Cleanup
- 🔴 6+ models loaded
- 🔴 ~2GB GPU memory
- 🔴 Confusing code (EmbeddingGemma = CLIP?)
- 🔴 Hard to optimize

### After Cleanup
- ✅ 3 main models (YOLO + CLIP + FastVLM)
- ✅ ~1.5GB GPU memory
- ✅ Clear responsibilities
- ✅ Easy to swap components

---

## Model Responsibilities (Final)

```
┌──────────────────────────────────────────────────┐
│                  VIDEO FRAME                     │
└──────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────┐
│  YOLO11x: "What objects? Where?"                │
│  Output: Bounding boxes + class labels          │
└──────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────┐
│  CLIP: "Is this the same object?"                │
│  Output: 512-dim embedding for re-ID             │
│          + Semantic verification                 │
└──────────────────────────────────────────────────┘
                      ↓
         ┌────────────────────────┐
         │  Same object seen      │
         │  multiple times?       │
         └────────────────────────┘
                ↓            ↓
              YES           NO
                ↓            ↓
         Skip description   Describe once
                            ↓
                ┌────────────────────────┐
                │  FastVLM: "Describe"   │
                │  Output: Rich text     │
                └────────────────────────┘
                            ↓
                ┌────────────────────────┐
                │  Neo4j Knowledge Graph │
                └────────────────────────┘
                            ↓
                ┌────────────────────────┐
                │  Gemma3:4b: Q&A only   │
                └────────────────────────┘
```

---

## Quick Wins (Do First)

### 1. Rename Files (5 minutes)

```bash
cd src/orion/backends
mv embedding_gemma.py clip_backend.py

# Update imports
find src -type f -name "*.py" -exec sed -i '' \
  's/from orion.backends.embedding_gemma/from orion.backends.clip_backend/g' {} \;
```

### 2. Rename Class (10 minutes)

```python
# In clip_backend.py
class CLIPEmbedder:  # was: EmbeddingGemmaVision
    """
    OpenAI CLIP for multimodal embeddings.
    
    NOT using Google's EmbeddingGemma because:
    - EmbeddingGemma is text-only
    - CLIP supports vision + text
    - CLIP is industry standard for multimodal
    """
```

### 3. Update Config (5 minutes)

```python
# In tracking_engine.py
class Config:
    # Visual Embedding
    VISION_ENCODER = 'clip'  # Clear!
    CLIP_MODEL = 'openai/clip-vit-base-patch32'
    USE_TEXT_CONDITIONING = True
    EMBEDDING_DIM = 512
```

### 4. Update Docs (10 minutes)

```bash
# Rename docs
mv docs/EMBEDDING_GEMMA_QUICKSTART.md docs/CLIP_QUICKSTART.md
mv docs/EMBEDDING_GEMMA_GUIDE.md docs/CLIP_GUIDE.md

# Update references
sed -i '' 's/EmbeddingGemma/CLIP/g' docs/*.md
```

---

## Long-Term Optimizations (Future)

### 1. Unified Model Manager

```python
# src/orion/model_manager.py
class UnifiedModelManager:
    """
    Single point of access for all models.
    Handles lazy loading, memory management, device placement.
    """
    
    def __init__(self):
        self._yolo = None
        self._clip = None
        self._fastvlm = None
        
    @property
    def yolo(self) -> YOLO:
        if self._yolo is None:
            self._yolo = self._load_yolo()
        return self._yolo
    
    @property
    def clip(self) -> CLIPModel:
        if self._clip is None:
            self._clip = self._load_clip()
        return self._clip
```

### 2. Shared Vision Encoder

Currently: FastVLM has its own FastViTHD encoder.

**Future:** Could we extract CLIP features and feed to FastVLM's LLM?

```python
# Experimental: Use CLIP features with FastVLM LLM
clip_features = clip.get_image_features(image)
# Convert to FastVLM's expected format
fastvlm_features = adapt_clip_to_fastvlm(clip_features)
description = fastvlm_llm.generate(fastvlm_features)
```

**Benefit:** Only load CLIP once, reuse for embeddings + descriptions.

**Challenge:** Architecture mismatch (CLIP ViT vs FastViTHD).

### 3. Model Quantization

Reduce memory usage:

```python
# 4-bit quantization
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    load_in_4bit=True  # 600MB → 150MB
)
```

---

## FAQ

### Q: Should I remove FastVLM and use CLIP for descriptions?

**A: No.** CLIP is for embeddings/classification, not detailed descriptions.

- **CLIP**: "This is a bottle" (classification)
- **FastVLM**: "A blue plastic water bottle with a white cap, sitting on a wooden table..." (rich description)

Keep both, use for different purposes.

### Q: Should I remove YOLO and use CLIP for detection?

**A: No.** CLIP is not a detector, it's a classifier.

- **YOLO**: Finds bounding boxes (where objects are)
- **CLIP**: Verifies what objects are (classification)

YOLO is much faster for detection (10ms vs 100ms).

### Q: Can I use one model for everything?

**A: Technically yes (vision LLMs), but not practical.**

Models like **GPT-4V** or **Gemini Vision** can do detection + description + embedding in one model, but:
- ❌ Slower (1-2 seconds per frame)
- ❌ Expensive (API costs)
- ❌ Less control over embeddings
- ❌ Harder to optimize

Specialized models are better for video analysis.

### Q: What about EmbeddingGemma? Should I use it anywhere?

**A: Only for text embeddings, not vision.**

If you need text embeddings (e.g., for knowledge graph queries), EmbeddingGemma is good:

```python
# For text-only embeddings
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("google/embeddinggemma-300m")
tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")

# Embed text queries
query_embedding = model(**tokenizer("Where is the keyboard?", return_tensors="pt"))
```

But for vision, use CLIP.

---

## Summary

### Do This Now (30 minutes)

1. ✅ Rename `embedding_gemma.py` → `clip_backend.py`
2. ✅ Rename class `EmbeddingGemmaVision` → `CLIPEmbedder`
3. ✅ Update config: `EMBEDDING_MODEL = 'clip'`
4. ✅ Update docs to say "CLIP" not "EmbeddingGemma"

### Final Clean Stack

```
YOLO11x    → Detection (what + where)
CLIP       → Embeddings (re-ID + verification)
FastVLM    → Descriptions (rich text)
Gemma3:4b  → Q&A (knowledge graph queries)
```

**Benefits:**
- Clear responsibilities
- No redundancy
- Easy to explain
- Easy to optimize

---

**Next:** Want me to implement the renaming for you?
