# Understanding EmbeddingGemma and Evaluation Strategy

## 1. What is EmbeddingGemma?

### Overview
**EmbeddingGemma** is Google's embedding model from the Gemma family, specifically designed to convert text into dense vector representations (embeddings). It's available through Ollama and optimized for semantic similarity tasks.

### How It Works in Orion

```
Text Description → EmbeddingGemma (via Ollama) → 768-dim vector
```

#### Use Case in Semantic Uplift
In `semantic_uplift.py`, EmbeddingGemma is used to:

1. **State Change Detection**:
   ```python
   description_before = "closed door"
   description_after = "open door"
   
   embedding_before = embeddinggemma.embed(description_before)  # [768 floats]
   embedding_after = embeddinggemma.embed(description_after)    # [768 floats]
   
   similarity = cosine_similarity(embedding_before, embedding_after)
   if similarity < 0.85:  # Threshold
       # State change detected!
   ```

2. **Semantic Scene Clustering**:
   - Embeddings of scene descriptions are used to cluster similar scenes
   - Helps identify when camera returns to same location

3. **Why Not ResNet50/OSNet for This?**
   - **Visual embeddings** (OSNet/ResNet50): For tracking objects visually across frames
   - **Text embeddings** (EmbeddingGemma): For comparing semantic meaning of descriptions
   - Two different purposes!

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Perception Engine                                       │
│  • OSNet/ResNet50 → Visual embeddings (512-dim)        │
│    Purpose: Track objects across frames visually       │
│    Input: Image crops                                   │
│    Output: [0.23, -0.45, 0.78, ...]                   │
└─────────────────────────────────────────────────────────┘
                         ↓
              Perception Objects with
              visual_embedding + rich_description
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Semantic Uplift Engine                                  │
│  • EmbeddingGemma → Text embeddings (768-dim)          │
│    Purpose: Compare semantic meaning of descriptions   │
│    Input: Text strings ("open door", "closed door")    │
│    Output: [0.12, 0.89, -0.34, ...]                   │
│                                                         │
│  Use Cases:                                             │
│  1. Detect state changes (description similarity)       │
│  2. Cluster similar scenes (location identification)    │
│  3. Entity disambiguation (same object, new context)    │
└─────────────────────────────────────────────────────────┘
```

### Alternative: Sentence Transformers
If Ollama/EmbeddingGemma isn't available, the system falls back to:
- **all-MiniLM-L6-v2** (384-dim)
- Smaller, faster, but slightly less accurate
- Works offline without Ollama

### Configuration

```python
# In semantic_uplift.py
class Config:
    EMBEDDING_MODEL_TYPE = "embeddinggemma"  # or "sentence-transformer"
    STATE_CHANGE_THRESHOLD = 0.85  # Cosine similarity threshold
```

---

## 2. Removing ResNet50 Fallback

You're right - we should commit to OSNet or fail gracefully. Let me fix that.

**Current Issue**: The code tries OSNet, then falls back to ResNet50 if OSNet isn't available.

**Better Approach**: 
- Try to use OSNet (best for Re-ID)
- If unavailable, use torchreid's OSNet implementation
- If that fails, raise informative error with installation instructions

---

## 3. SOTA Benchmarks & Comparison

### Current State
Your current system can be compared against several SOTA approaches:

#### A. Video Scene Graph Generation (VidSGG) Benchmarks

**1. Action Genome (AG)**
- **What**: 10K video clips with dense scene graph annotations
- **Metrics**: Scene Graph Detection (SGDet), Classification (SGCls), Predicate Classification (PredCls)
- **SOTA Models**:
  - STTran (Spatial-Temporal Transformer) - 2020
  - TRACE (Temporal Relation-Aware Causal Enhancement) - 2022
  - TEMPURA - 2023
  - **Your advantage**: CIS provides mathematical grounding vs. pure attention

**2. VidVRD (Video Visual Relation Detection)**
- **What**: 1000 videos with relation triplets <subject, predicate, object>
- **Metrics**: Relation detection @ K (top-K predictions)
- **SOTA Models**:
  - VidVRD baseline
  - STGCN (Spatio-Temporal Graph CNN)
  - **Your advantage**: Motion-aware causal scoring

**3. VSGR (Video Scene Graph Recognition)** ← You already support this!
- Custom dataset format
- Good for testing your specific approach

#### B. Causal Reasoning Benchmarks

**4. CLEVRER (Causal Reasoning from Events in Videos)**
- **What**: Synthetic videos with physics events
- **Metrics**: Causal question answering accuracy
- **Task**: "What caused the sphere to fall?"
- **SOTA Models**:
  - NS-DR (Neuro-Symbolic Dynamic Reasoning)
  - ALOE (Adversarial Learning of Object Embeddings)
  - **Your advantage**: Explicit CIS scoring vs. implicit learning

**5. Something-Something V2** (with causal extensions)
- **What**: 220K videos of humans performing actions
- **Causal Extension**: Identify which object/action caused state changes
- **Your advantage**: Directed motion detection

#### C. Direct SOTA Comparisons

| Benchmark | SOTA Model | Their Approach | Your Advantage |
|-----------|------------|----------------|----------------|
| Action Genome | TEMPURA (2023) | Transformer + GNN | Mathematical CIS grounding |
| VidVRD | TRACE (2022) | Attention + Causal Learning | Explicit motion modeling |
| CLEVRER | NS-DR (2021) | Neuro-symbolic | Hybrid symbolic+LLM |

### Recommended Evaluation Strategy

#### Tier 1: Quick Validation (Week 1)
```python
# Use your heuristic baseline as sanity check
python scripts/run_evaluation.py --video test.mp4

# Expected: Your method > Heuristic baseline
# Metrics: Causal Precision, Edge F1
```

#### Tier 2: Public Benchmark (Weeks 2-4)
```python
# Action Genome evaluation
# 1. Download AG dataset: https://github.com/JingweiJ/ActionGenome
# 2. Convert annotations to Orion format
# 3. Run evaluation

python scripts/run_evaluation.py \
    --mode benchmark \
    --benchmark action_genome \
    --dataset-path /path/to/AG

# Compare against published TEMPURA/TRACE results
```

#### Tier 3: SOTA Comparison (Months 1-2)
```python
# Implement SOTA baselines in your framework
# - STTran (attention-based)
# - TRACE (causal learning)
# - Your CIS+LLM

# Head-to-head comparison on same test set
```

### Concrete Accuracy Measurement

**Ground Truth Required**:
For proper accuracy measurement, you need datasets with:
1. Annotated object trajectories
2. Labeled relationships (spatial, temporal)
3. Annotated causal events with agent/patient labels

**Available Datasets**:

1. **Action Genome** ✅ (Best choice)
   - 10K clips with full scene graph annotations
   - Includes causal relationships
   - Widely used in research
   - Download: https://github.com/JingweiJ/ActionGenome

2. **VidVRD** ✅
   - 1000 videos with relation triplets
   - Standard benchmark
   - Download: https://xdshang.github.io/docs/vrdchallenge.html

3. **CLEVRER** ✅ (For causal reasoning)
   - Synthetic but rigorous
   - Explicit causality labels
   - Download: http://clevrer.csail.mit.edu/

### Metrics for Accuracy

```python
# Scene Graph Metrics
- SGDet (Scene Graph Detection): Detect objects + relationships
- SGCls (Scene Graph Classification): Given objects, classify relationships
- PredCls (Predicate Classification): Given object pairs, classify predicate

# Causal Metrics
- Causal Precision: TP_causal / (TP_causal + FP_causal)
- Causal Recall: TP_causal / (TP_causal + FN_causal)
- Causal F1: Harmonic mean

# Your Novel Contribution
- CIS Correlation: How well CIS scores correlate with ground truth causality
- Motion Impact: Ablation study removing f_motion component
```

---

## 4. Implementation Plan

### Step 1: Fix OSNet (No ResNet50 fallback)
```python
# perception_engine.py
def get_embedding_model(self):
    # Try OSNet from timm
    # If fails, try torchreid
    # If fails, raise error with instructions
    # NO fallback to ResNet50
```

### Step 2: Unit Tests
```python
# tests/test_motion_tracker.py
def test_velocity_estimation()
def test_direction_detection()

# tests/test_causal_inference.py
def test_cis_calculation()
def test_proximity_score()

# tests/test_integration.py
def test_full_pipeline()
```

### Step 3: Action Genome Support
```python
# src/orion/evaluation/benchmarks/action_genome_loader.py
class ActionGenomeDataset:
    def load_annotations(self)
    def to_orion_format(self)
    def evaluate(self)
```

### Step 4: SOTA Comparison
```python
# src/orion/evaluation/sota/
├── sttran_baseline.py      # Implement STTran
├── trace_baseline.py       # Implement TRACE
└── compare_methods.py      # Head-to-head comparison
```

---

## 5. Your Competitive Advantages

### vs. Transformer-based (STTran, TRACE)
✅ **Mathematical grounding**: CIS provides interpretable scores
✅ **Explainability**: Can show why a causal link was detected
✅ **Efficiency**: Filter before LLM (70% reduction in calls)

### vs. Pure Neural (GNNs, CNNs)
✅ **Hybrid approach**: Combines symbolic (CIS) + neural (LLM)
✅ **Less data hungry**: Math + small LLM vs. huge transformers
✅ **Motion-aware**: Explicit velocity/direction modeling

### vs. Rule-based (Your Heuristic)
✅ **Semantic understanding**: LLM provides rich labels
✅ **Generalization**: Handles novel event types
✅ **Adaptive**: CIS weights can be learned

---

## 6. Recommended Next Steps

### This Week
1. ✅ Fix OSNet fallback issue
2. ✅ Add unit tests
3. ✅ Test on sample video

### Next 2 Weeks
1. Download Action Genome dataset
2. Implement AG loader (similar to VSGR)
3. Run first benchmark comparison

### Month 1
1. Compare against published STTran/TRACE results
2. Tune CIS weights for optimal performance
3. Run ablation studies

### Month 2+
1. Implement SOTA baselines for direct comparison
2. Publish research paper with results
3. Release code and pretrained models

Would you like me to proceed with fixing the OSNet issue, adding unit tests, and creating the Action Genome loader?
