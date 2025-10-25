# Orion Research & Evaluation Plan

**Status**: Design Phase  
**Last Updated**: October 25, 2025  
**Goal**: Establish comprehensive evaluation of Orion's semantic video understanding against state-of-the-art baselines

---

## 📊 Evaluation Strategy Overview

### Datasets (2 Primary)

| Dataset | Domain | Purpose | Metrics |
|---------|--------|---------|---------|
| **Action Genome** | Third-person indoor videos, annotated triplets | Quantitative SGG/SGA evaluation | Triplet F1, Causal F1, Entity Continuity, Recall@K |
| **EASG (Ego4D)** | First-person egocentric manipulation videos | Cross-domain + egocentric validation | Triplet F1, Action understanding, Interaction graphs |

**Decision**: Action Genome provides established benchmark comparison. EASG validates Orion on egocentric data (hands, objects, manipulation) which is Orion's target domain.

---

## 🎯 Evaluation Axes

### What Each Baseline Tests

| Baseline | Tests | Orion Advantage Validated |
|----------|-------|---------------------------|
| **Heuristic Uplift** | Can simple rules build scene graphs? | Semantic reasoning superiority |
| **LLM-Only (VoT)** | Can vision-language models alone understand scenes? | Structured representation necessity |
| **HyperGLM** | Multimodal structured reasoning | Annotation-free operation |
| **SceneSayer-ODE** (optional) | Temporal ODE-based relational modeling | Efficiency + accuracy tradeoff |
| **Orion (full)** | Unique entity tracking + semantic uplift + causal reasoning | – |

---

## 📁 Research Folder Structure

```
research/
├── EVALUATION_PLAN.md              # This file
├── README.md                        # Quick start guide for evaluation
│
├── datasets/                        # Dataset loaders & evaluators
│   ├── __init__.py
│   ├── action_genome.py             # Action Genome loader + GT matching
│   ├── easg.py                      # EASG (Ego4D) loader + egocentric evaluation
│   └── utils.py                     # IoU matching, entity alignment
│
├── baselines/                       # Baseline implementations
│   ├── __init__.py
│   ├── heuristic.py                 # Rule-based SGG
│   ├── llm_vot.py                   # FastVLM → LLM triplet extraction
│   ├── hyperglm/                    # Full HyperGLM implementation
│   │   ├── __init__.py
│   │   ├── entity_graph.py          # Per-frame entity scene graphs
│   │   ├── procedural_graph.py      # Temporal transition matrix
│   │   ├── hypergraph.py            # Random walk sampling (Alg. 1)
│   │   ├── visual_encoder.py        # CLIP ViT-L-336 + MLP projector
│   │   ├── llm_reasoning.py         # Mistral-7B + LoRA inference
│   │   └── inference.py             # Full pipeline orchestration
│   └── scenesayer/                  # (Optional) SceneSayer-ODE
│       └── ...
│
├── evaluation/                      # Metrics & analysis
│   ├── __init__.py
│   ├── metrics.py                   # Triplet F1, Recall@K, Entity Continuity, Causal F1
│   ├── captioning_metrics.py        # BLEU, METEOR, CIDEr
│   ├── matcher.py                   # Prediction-to-GT matching logic
│   └── analysis.py                  # Results visualization, error analysis
│
├── ablations/                       # Ablation study framework
│   ├── __init__.py
│   ├── runner.py                    # Unified ablation experiment runner
│   └── configs/                     # Ablation config presets
│       ├── no_tracking.yaml
│       ├── no_llm.yaml
│       ├── no_uplift.yaml
│       └── no_graph_reasoning.yaml
│
├── run_evaluation.py                # Main experiment orchestrator
├── analyze_results.py               # Generate tables, plots, LaTeX output
│
└── results/                         # Experiment outputs (gitignored)
    ├── action_genome/
    │   ├── orion_full.json
    │   ├── heuristic.json
    │   ├── llm_vot.json
    │   └── hyperglm.json
    ├── easg/
    │   └── ...
    └── tables/                      # Generated LaTeX tables
        └── comparison.tex
```

---

## 🧪 Evaluation Pipeline Design

### 1. Action Genome Evaluation (Quantitative)

**Input**: All annotated videos from Action Genome test split

**Process**:
1. **Detection + Tracking**: Run Orion perception to assign unique entity IDs across frames
2. **SGG**: Generate per-frame scene graphs (triplets)
3. **SGA**: Predict next-frame or causal relations
4. **Matching**: Map predicted entities to GT entities (IoU > 0.5 + class match)
5. **Triplet Evaluation**: Compare predicted triplets to GT triplets

**Output Metrics**:
- Triplet Precision / Recall / F1
- Recall@10, Recall@20, Recall@50
- Entity Continuity (% entities correctly tracked across frames)
- Causal F1 (for future relation prediction)
- Runtime (FPS)

### 2. EASG (Ego4D) Evaluation

**Input**: EASG egocentric manipulation videos from Ego4D

**Process**:
1. Run same Orion pipeline
2. Evaluate against EASG scene graph annotations
3. Focus on hand-object interactions and manipulation actions
4. Compute action understanding metrics

**Output Metrics**:
- Triplet F1 (egocentric context)
- Hand-object interaction accuracy
- Action prediction accuracy
- Graph consistency score

---

## 🔬 Baseline Implementation Details

### 1. Heuristic Baseline (`research/baselines/heuristic.py`)

**Strategy**: Use YOLO11x detections + rule-based relation assignment

```python
# Pseudo-implementation
def heuristic_baseline(video_path):
    frames = extract_frames(video_path)
    detections = [yolo_detect(f) for f in frames]
    
    # Rule-based relation assignment
    for frame_dets in detections:
        for obj1, obj2 in combinations(frame_dets, 2):
            # Spatial heuristics
            if is_above(obj1, obj2):
                relations.append((obj1, "above", obj2))
            elif is_near(obj1, obj2):
                relations.append((obj1, "near", obj2))
            # etc.
    
    return build_scene_graph(detections, relations)
```

**Expected Performance**: Low F1, no entity continuity (frame-independent)

---

### 2. LLM-Only VoT Baseline (`research/baselines/llm_vot.py`)

**Strategy**: Caption frames with FastVLM → Extract triplets with LLM

```python
def llm_vot_baseline(video_path):
    frames = extract_frames(video_path)
    
    # Generate captions
    captions = [fastvlm.caption(f) for f in frames]
    
    # Extract triplets via LLM prompt
    prompt = f"Extract all <subject, predicate, object> triplets from: {caption}"
    triplets = gemma3.generate(prompt)
    
    return parse_triplets(triplets)
```

**Expected Performance**: Better than heuristic, but lacks tracking and structured reasoning

---

### 3. HyperGLM Baseline (`research/baselines/hyperglm/`)

**Strategy**: Full implementation from HyperGLM paper (not EASG-specific)

**Components** (must implement from scratch):

#### 3.1 Entity Graph Builder (`entity_graph.py`)
- Faster R-CNN detection (or adapt YOLO11x)
- Per-frame entity nodes V^e_t
- Pairwise relation classifier → edges E^e_t

#### 3.2 Procedural Graph (`procedural_graph.py`)
- Compute transition matrix w(r_m, r_n) from training data
- Model relation transitions across frames

#### 3.3 HyperGraph Sampler (`hypergraph.py`)
- Implement Algorithm 1 (random walk sampling)
- Hyperparams: N_w = 60, N_l = 7
- Sample hyperedges encoding multi-object interactions

#### 3.4 Visual Encoder (`visual_encoder.py`)
- CLIP ViT-L-336 frame encoding
- MLP projector (2-layer) → LLM token space

#### 3.5 LLM Reasoning (`llm_reasoning.py`)
- Mistral-7B-Instruct (or compatible)
- LoRA adapters (rank=128, scale=256)
- Interleave visual tokens + hypergraph prompt

#### 3.6 Full Pipeline (`inference.py`)
```python
def hyperglm_inference(video_path):
    frames = extract_frames(video_path)
    
    # 1. Detect objects
    detections = [detector(f) for f in frames]
    
    # 2. Build entity graphs per frame
    entity_graphs = [build_entity_graph(d) for d in detections]
    
    # 3. Load/compute procedural transition matrix
    proc_graph = load_procedural_graph()
    
    # 4. Unify into HyperGraph
    H = build_hypergraph(entity_graphs, proc_graph)
    
    # 5. Sample hyperedges
    H_aug = random_walk_sample(H, Nw=60, Nl=7)
    
    # 6. Encode frames with CLIP
    visual_tokens = clip_encode_and_project(frames)
    
    # 7. Build LLM prompt
    prompt = hypergraph_to_prompt(H_aug)
    
    # 8. LLM reasoning
    outputs = llm_with_lora(visual_tokens, prompt)
    
    return parse_triplets(outputs)
```

**Expected Performance**: Strong SGG/SGA (best baseline), but requires annotations for procedural graph

**Key Difference from Orion**: Requires ground-truth relation annotations for transition matrix training; Orion is fully annotation-free.

---

### 4. SceneSayer-ODE (Optional, Time Permitting)

**Status**: Deferred unless resources allow  
**Complexity**: High (continuous-time ODE solver for relational dynamics)

---

## 🧩 Ablation Study Design

### Ablation Configurations

| Ablation | Modification | Expected Impact |
|----------|-------------|-----------------|
| **No Tracking** | Disable entity ID linking → per-frame detections only | ↓ Entity Continuity, ↓ Causal F1 |
| **No Semantic Uplift** | Use YOLO classes only, skip VLM descriptions | ↓ Triplet F1 (less semantic accuracy) |
| **No Graph Reasoning** | Disable hypergraph/attention layer (if present) | ↓ Complex relation accuracy |
| **No LLM** | Visual → symbolic only, no language layer | ↓ Causal reasoning, ↓ Caption quality |
| **LLM Prompt Variants** | Test structured vs. natural prompts | Measure LLM reasoning dependency |

### Ablation Runner (`research/ablations/runner.py`)

```python
def run_ablation_study(dataset, config):
    """
    Run all ablations systematically
    
    Args:
        dataset: "action_genome" or "vsgr_aspire"
        config: Base OrionConfig
    
    Returns:
        DataFrame with results for each ablation
    """
    results = []
    
    for ablation in ["full", "no_tracking", "no_llm", "no_uplift"]:
        cfg = modify_config(config, ablation)
        metrics = evaluate_orion(dataset, cfg)
        results.append({"ablation": ablation, **metrics})
    
    return pd.DataFrame(results)
```

---

## 📊 Expected Results Table (Target for Paper)

| Model | Dataset | Triplet F1 | Causal F1 | Entity Cont. | Caption CIDEr | Runtime (fps) |
|-------|---------|------------|-----------|--------------|---------------|---------------|
| Heuristic | AG | 31.2 | – | 42.1 | – | 48 |
| LLM-Only (FastVLM) | AG | 38.5 | 22.1 | – | 0.45 | 5 |
| HyperGLM | AG | 51.4 | 48.7 | – | 1.02 | 2 |
| **Orion (full)** | **AG** | **63.8** | **61.2** | **91.5** | **1.26** | **7** |
| Orion (–tracking) | AG | 56.1 | 42.8 | 63.4 | – | – |
| Orion (–uplift) | AG | 44.0 | 33.7 | 58.2 | – | – |
| Orion (–LLM) | AG | 39.9 | 25.6 | 72.0 | – | – |

*(Numbers are hypothetical targets — replace with actual experimental results)*

---

## 🚀 Implementation Priorities

### Phase 1: Core Infrastructure (Week 1-2)
1. ✅ **Metrics Module** (`research/evaluation/metrics.py`)
2. ✅ **Action Genome Loader** (`research/datasets/action_genome.py`)
3. ✅ **Entity Matching Logic** (`research/evaluation/matcher.py`)

### Phase 2: Baselines (Week 3-4)
4. ✅ **Heuristic Baseline** (simple rules)
5. ✅ **LLM-Only Baseline** (FastVLM + Gemma)
6. 🔄 **HyperGLM Implementation** (complex, 1-2 weeks)

### Phase 3: Evaluation & Ablations (Week 5)
7. ✅ **Unified Experiment Runner** (`research/run_evaluation.py`)
8. ✅ **Ablation Framework** (`research/ablations/runner.py`)

### Phase 4: VSGR Dataset (Week 6, Optional)
9. 🔄 **ASPIRe Loader** (if Action Genome results insufficient)

### Phase 5: Analysis & Paper Integration (Week 7)
10. ✅ **Results Analysis** (tables, plots, LaTeX generation)
11. ✅ **Qualitative Examples** (success/failure cases)

---

## 🔧 Key Technical Decisions

### Entity Matching Protocol
- **IoU Threshold**: 0.5 (standard for object detection)
- **Class Match**: Required (strict evaluation)
- **Temporal Window**: ±2 frames for relation matching (to handle annotation jitter)

### Triplet Evaluation Protocol
```python
def match_triplet(pred_triplet, gt_triplet, entity_map, iou_thresh=0.5):
    """
    Match predicted triplet to GT triplet
    
    Args:
        pred_triplet: (subj_id, pred_label, obj_id)
        gt_triplet: (gt_subj_id, gt_pred_label, gt_obj_id)
        entity_map: Dict mapping pred entity IDs to GT IDs (via IoU)
    
    Returns:
        bool: True if match (TP), False otherwise (FP)
    """
    subj_match = entity_map.get(pred_triplet.subj_id) == gt_triplet.subj_id
    obj_match = entity_map.get(pred_triplet.obj_id) == gt_triplet.obj_id
    pred_match = pred_triplet.predicate == gt_triplet.predicate
    
    return subj_match and obj_match and pred_match
```

### HyperGLM Adaptation Notes
- **Detection Backend**: Can substitute Faster R-CNN with YOLO11x + ROI pooling adapter
- **Procedural Graph**: Compute from Action Genome training split (requires preprocessing)
- **LLM Choice**: Use Mistral-7B-Instruct or fallback to Gemma-7B if LoRA weights unavailable

---

## 📝 Open Questions & TODOs

### Research Questions
- [ ] **Q1**: Should we evaluate on Action Genome full test split or subsample for speed?
- [ ] **Q2**: Is ASPIRe alone sufficient or do we need full VSGR (+ AeroEye)?
- [ ] **Q3**: For HyperGLM, do we train procedural graph ourselves or use uniform prior?

### Implementation TODOs
- [ ] Obtain Action Genome dataset annotations (download links)
- [ ] Obtain VSGR/ASPIRe annotations
- [ ] Verify Orion output format includes all required fields for evaluation
- [ ] Create config flags for ablations (`--disable-tracking`, etc.)
- [ ] Implement captioning metrics (pycocoeval or similar)

### Paper Integration
- [ ] Draft evaluation section structure (Methods 4.1–4.4 rewrite)
- [ ] Prepare qualitative figure showing Orion vs. baselines
- [ ] Error analysis categories (entity mismatch, wrong predicate, missed relation)

---

## 📚 Reference Links

- **Action Genome**: https://www.actiongenome.org/ (dataset homepage)
- **VSGR Benchmark**: (paper reference for ASPIRe/AeroEye dataset)
- **HyperGLM Paper**: (uploaded PDF — implementation guide)
- **SceneSayer-ODE**: (optional baseline, may have released code)

---

## 🎯 Success Criteria

### Minimum Viable Evaluation
1. ✅ Action Genome quantitative results (Orion vs. 3 baselines)
2. ✅ 3 ablation studies showing component contributions
3. ✅ Qualitative examples (2-3 videos with visualized graphs)

### Full Evaluation (Ideal)
4. 🔄 VSGR/ASPIRe cross-domain generalization results
5. 🔄 HyperGLM comparison (if implementation feasible)
6. 🔄 Runtime/efficiency analysis (FPS, memory usage)
7. 🔄 Error analysis breakdown by predicate type

---

**Next Steps**: Start with Phase 1 (metrics + Action Genome loader). Review this plan and confirm priorities before implementation begins.
