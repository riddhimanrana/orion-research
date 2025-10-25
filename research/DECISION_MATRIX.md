# Orion Evaluation: Decision Matrix & Recommendations

**Date**: October 25, 2025  
**Purpose**: Help you make strategic decisions about evaluation scope and timeline

---

## 🎯 Quick Recommendations (TL;DR)

### For a Strong Paper (Minimum Viable Evaluation)
**Timeline**: 4-5 weeks  
**Effort**: Moderate

```
✅ Datasets: Action Genome only
✅ Baselines: Heuristic + LLM-Only VoT + Orion
✅ Metrics: Triplet F1, Entity Continuity, Causal F1
✅ Ablations: 3 key ablations (no-tracking, no-LLM, no-uplift)
❌ Skip: HyperGLM, VSGR, SceneSayer-ODE
```

**Justification**: This gives you quantitative results against two diverse baselines (symbolic + language-only), validates your unique contributions (entity tracking, semantic uplift), and is achievable in 1 month.

---

### For a Comprehensive Evaluation (Ideal)
**Timeline**: 6-7 weeks  
**Effort**: High

```
✅ Datasets: Action Genome + VSGR/ASPIRe
✅ Baselines: Heuristic + VoT + HyperGLM + Orion
✅ Metrics: All (Triplet F1, Recall@K, Entity Continuity, Causal F1, Captioning)
✅ Ablations: 4+ ablations including prompt variants
✅ Analysis: Error breakdown, qualitative examples, runtime analysis
❌ Optional: SceneSayer-ODE (defer if time-constrained)
```

**Justification**: This provides SOTA comparison (HyperGLM), cross-domain validation (VSGR), and comprehensive analysis. Strong for top-tier venue submission.

---

## 📊 Decision Matrix

### Decision 1: Dataset Scope

| Option | Pros | Cons | Time Cost | Recommendation |
|--------|------|------|-----------|----------------|
| **Action Genome only** | ✅ Quantitative metrics<br>✅ Established benchmark<br>✅ GT annotations available | ❌ Single domain<br>❌ No generalization claim | 0 weeks (baseline) | **Minimum Viable** |
| **+ VSGR/ASPIRe** | ✅ Cross-domain validation<br>✅ Egocentric videos<br>✅ Procedural reasoning | ❌ Annotation format may differ<br>❌ Extra loader implementation | +1 week | **Nice-to-have** |
| **Full VSGR (ASPIRe + AeroEye)** | ✅ Comprehensive coverage<br>✅ Aerial + egocentric | ❌ Significant extra effort<br>❌ May not add much value | +2 weeks | **Overkill** (skip) |

**👍 Recommended**: Start with **Action Genome only**. Add ASPIRe later if reviewers request cross-domain validation.

---

### Decision 2: Baseline Coverage

| Baseline | Value | Effort | Priority | Notes |
|----------|-------|--------|----------|-------|
| **Heuristic** | High — shows value of ML | Low (✅ already exists) | 🔴 **Must-have** | Rule-based, fast to run |
| **LLM-Only VoT** | High — shows structured representation value | Low (✅ already exists) | 🔴 **Must-have** | Language-only approach |
| **HyperGLM** | Very High — SOTA comparison | High (❌ needs full implementation) | 🟡 **Highly recommended** | 1-2 weeks of work |
| **SceneSayer-ODE** | Medium — temporal modeling | Very High (❌ complex ODE solver) | 🟢 **Optional** | Defer unless critical |

**Effort Breakdown for HyperGLM**:
```
Entity Graph Builder:        2 days
Procedural Graph:            2 days (if using uniform prior)
                         or  4 days (if training from AG data)
HyperGraph Sampler:          2 days
Visual Encoder (CLIP+MLP):   1 day
LLM Reasoning (LoRA):        2 days
Integration + Testing:       2-3 days
---
Total:                       11-15 days (~2 weeks)
```

**👍 Recommended**: Include **HyperGLM** if you have 6+ weeks. It's the difference between "good evaluation" and "comprehensive evaluation."

**💡 Pro Tip**: If you skip HyperGLM initially, you can always add it in a revision phase after reviewer feedback.

---

### Decision 3: Metrics Scope

| Metric | Importance | Effort | Use Case |
|--------|-----------|--------|----------|
| **Triplet Precision/Recall/F1** | 🔴 Critical | Low | Core SGG evaluation |
| **Entity Continuity** | 🔴 Critical | Medium | Your unique strength |
| **Causal F1** | 🔴 Critical | Low | CIS validation |
| **Recall@10/20/50** | 🟡 Recommended | Medium | Standard SGG metric |
| **Captioning (BLEU/CIDEr)** | 🟢 Optional | Medium | Graph-to-language quality |
| **Runtime (FPS)** | 🟡 Recommended | Low | Efficiency claim |

**👍 Recommended Minimum**:
- Triplet F1
- Entity Continuity
- Causal F1
- Runtime/FPS

**Add if time permits**:
- Recall@K (standard for SGG papers)
- Captioning metrics (if you have graph→caption module)

---

### Decision 4: Ablation Study Depth

| Ablation | Validates | Effort | Priority |
|----------|-----------|--------|----------|
| **No Tracking** | Entity tracking benefit | Low (config flag) | 🔴 Must-have |
| **No Semantic Uplift** | VLM description value | Low (config flag) | 🔴 Must-have |
| **No LLM** | Language reasoning benefit | Low (config flag) | 🔴 Must-have |
| **No Graph Reasoning** | HyperGraph/attention value | Medium (if implemented) | 🟡 Recommended |
| **LLM Prompt Variants** | Prompt engineering impact | Low | 🟢 Optional |

**👍 Recommended Minimum**: First 3 ablations (no-tracking, no-uplift, no-LLM)

---

## 🗓️ Timeline Scenarios

### Scenario A: Fast Track (4 weeks)
**Goal**: Minimum viable evaluation for paper submission

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | Metrics extension + Entity matcher + AG verification | Working evaluation pipeline |
| 2 | Update Heuristic + VoT baselines to standard format | 2 baselines ready |
| 3 | Run evaluation on Action Genome (50 videos) | Quantitative results |
| 4 | Ablations + analysis + LaTeX tables | Paper-ready results |

**Output**: Table comparing Orion vs. Heuristic vs. VoT + 3 ablations

---

### Scenario B: Comprehensive (6 weeks)
**Goal**: Strong evaluation with SOTA baseline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | Metrics + Matcher + AG verification | Evaluation pipeline |
| 2 | Update existing baselines + start HyperGLM | Baselines ready |
| 3 | Finish HyperGLM implementation | 3 baselines complete |
| 4 | Run full evaluation on Action Genome (100+ videos) | Quantitative results |
| 5 | Ablations + runtime analysis | Ablation study complete |
| 6 | Results analysis + qualitative examples + paper integration | Camera-ready |

**Output**: Full comparison table + ablations + error analysis + qualitative figures

---

### Scenario C: Ideal (7-8 weeks)
**Goal**: Publication-ready with cross-domain validation

Same as Scenario B, plus:

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 7 | Implement VSGR/ASPIRe loader | Cross-domain dataset ready |
| 8 | Run on VSGR + cross-domain analysis | Generalization results |

**Output**: Action Genome results + VSGR results + cross-domain transfer analysis

---

## 💰 Cost-Benefit Analysis

### What You Get from Each Component

#### Action Genome Evaluation
**Cost**: Baseline (already planned)  
**Benefit**: 
- ✅ Quantitative results (required for any paper)
- ✅ Direct comparison with prior work
- ✅ Standard benchmark recognition

**Verdict**: 🔴 **Non-negotiable**

---

#### HyperGLM Baseline
**Cost**: 2 weeks implementation + compute for Mistral-7B inference  
**Benefit**:
- ✅ Comparison against SOTA multimodal reasoning
- ✅ Shows your annotation-free advantage
- ✅ Validates performance despite no procedural graph training
- ⚠️ Requires GPU with 16GB+ VRAM for LLM inference

**Verdict**: 🟡 **High value, but can defer if time-constrained**

**Alternative**: Cite HyperGLM's reported numbers and note:
> "Direct comparison with HyperGLM is challenging as it requires training 
> a procedural graph on Action Genome annotations, while Orion operates 
> annotation-free. We provide comparison against annotation-free baselines 
> (Heuristic, LLM-Only) which are more directly comparable to our approach."

---

#### VSGR/ASPIRe Dataset
**Cost**: 1 week loader implementation + evaluation  
**Benefit**:
- ✅ Cross-domain generalization claim
- ✅ Egocentric video validation
- ⚠️ May not have comprehensive GT annotations

**Verdict**: 🟢 **Nice-to-have for broader impact claim**

---

#### Captioning Metrics (BLEU/CIDEr)
**Cost**: 1 day (pycocoeval integration)  
**Benefit**:
- ✅ Graph→language quality validation
- ⚠️ Only useful if you have caption generation module

**Verdict**: 🟢 **Optional — only if you already generate captions**

---

## 🧭 Strategic Recommendations

### For Conference Submission (Tight Deadline)
➡️ **Go with Scenario A (4 weeks)**

Focus on:
1. Action Genome evaluation
2. Heuristic + VoT baselines
3. 3 core ablations
4. Triplet F1 + Entity Continuity + Causal F1

**Why**: This gives you solid quantitative results and validates your unique contributions. You can always add HyperGLM in camera-ready or journal extension.

---

### For Journal Submission (More Time)
➡️ **Go with Scenario B (6 weeks)**

Add:
1. HyperGLM baseline
2. Recall@K metrics
3. Error analysis
4. Qualitative examples

**Why**: Journals expect comprehensive evaluation. HyperGLM comparison strengthens your contribution claim.

---

### For Top-Tier Venue (CVPR/ICCV/NeurIPS)
➡️ **Go with Scenario C (7-8 weeks)**

Add everything:
1. Full baseline suite
2. Cross-domain validation (VSGR)
3. Comprehensive ablations
4. Runtime/efficiency analysis
5. Qualitative + quantitative results

**Why**: Top venues expect thorough evaluation with cross-domain validation and SOTA comparisons.

---

## ⚠️ Risk Assessment

### Risks of Skipping HyperGLM

**Reviewer Concern**:
> "How does Orion compare to recent multimodal scene graph models like HyperGLM?"

**Mitigation**:
1. Emphasize annotation-free operation as key difference
2. Compare against annotation-free baselines (Heuristic, VoT)
3. Cite HyperGLM's numbers and note fundamental difference in approach
4. Highlight entity continuity as unique contribution (HyperGLM doesn't report this)

**Risk Level**: 🟡 **Medium** — May get reviewer pushback, but addressable

---

### Risks of Skipping VSGR

**Reviewer Concern**:
> "Results only on Action Genome — does this generalize to other domains?"

**Mitigation**:
1. Discuss design choices that promote generalization (annotation-free, domain-agnostic CIS)
2. Provide qualitative examples on diverse videos
3. Acknowledge limitation and propose future work

**Risk Level**: 🟢 **Low** — Single-dataset evaluation is common and acceptable

---

### Risks of Minimal Ablations

**Reviewer Concern**:
> "What's the contribution of each component?"

**Mitigation**:
1. Ensure you have at least 3 ablations (no-tracking, no-uplift, no-LLM)
2. Discuss each component's design rationale in related work
3. Provide qualitative examples showing failure modes without each component

**Risk Level**: 🔴 **High** if < 3 ablations — Ablations are expected for systems papers

---

## ✅ Final Recommendations

### Minimum for Acceptance
```
✅ Action Genome quantitative evaluation
✅ Heuristic baseline
✅ LLM-Only VoT baseline  
✅ Triplet F1, Entity Continuity, Causal F1
✅ 3 ablations (no-tracking, no-uplift, no-LLM)
```

### Recommended for Strong Paper
```
+ HyperGLM baseline (SOTA comparison)
+ Recall@K metrics (standard SGG metric)
+ Runtime/FPS analysis
+ Qualitative examples (success + failure cases)
+ Error breakdown by predicate type
```

### Ideal for Top-Tier Venue
```
+ VSGR/ASPIRe cross-domain validation
+ 4+ ablations including prompt variants
+ Captioning metrics (if applicable)
+ Comprehensive analysis section
```

---

## 🎯 Action Plan (Start Today)

### Step 1: Make Decisions (30 min)
- [ ] Confirm dataset scope (Action Genome only vs. + VSGR)
- [ ] Decide on HyperGLM (implement now vs. defer)
- [ ] Choose metrics (minimum vs. full set)
- [ ] Select timeline scenario (A, B, or C)

### Step 2: Setup Phase (Week 1, Day 1-2)
- [ ] Download Action Genome dataset and annotations
- [ ] Verify existing baselines work (run on 1 sample video)
- [ ] Confirm Orion outputs required format for evaluation

### Step 3: Implementation Start (Week 1, Day 3-5)
- [ ] Extend `research/evaluation/metrics.py` with new functions
- [ ] Create `research/evaluation/matcher.py` (entity matching)
- [ ] Test full evaluation pipeline on 5 videos

### Step 4: Iterate (Weeks 2-4+)
- [ ] Follow phased implementation plan
- [ ] Run continuous validation (test on small subset before full run)
- [ ] Generate results incrementally

---

## 📞 Questions to Answer Before Starting

1. **What's your paper deadline?** (Conference submission date)
2. **Do you have access to GPU with 16GB+ VRAM?** (For HyperGLM LLM inference)
3. **Have you successfully downloaded Action Genome annotations?**
4. **Can you run existing baselines (Heuristic + VoT) successfully?**
5. **What's your priority: speed-to-submission vs. comprehensive evaluation?**

---

**Next Action**: Review this decision matrix, answer the questions above, and commit to a scenario (A, B, or C). Then proceed with implementation starting from Phase 1 of the roadmap.
