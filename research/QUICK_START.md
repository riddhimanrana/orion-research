# Orion Evaluation: Quick Start Guide

**Last Updated**: October 25, 2025

This is your 5-minute guide to understanding what's been created and how to start implementing.

---

## 📁 What's New

Four comprehensive planning documents have been created in `research/`:

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **RESEARCH_SUMMARY.md** | 📖 Overview + current status | 5 min |
| **DECISION_MATRIX.md** | 🎯 Strategic decisions + recommendations | 10 min |
| **EVALUATION_PLAN.md** | 📋 High-level strategy + expected results | 15 min |
| **IMPLEMENTATION_ROADMAP.md** | 🛠️ Detailed implementation guide | 30 min |

---

## 🚀 Start Here (30-Minute Onboarding)

### 1. Read the Summary (5 min)
```bash
cat research/RESEARCH_SUMMARY.md
```

**Key Takeaway**: You already have foundation code (Heuristic + VoT baselines exist!). Need to extend metrics, add HyperGLM (optional), and build unified evaluation runner.

---

### 2. Make Strategic Decisions (10 min)
```bash
cat research/DECISION_MATRIX.md
```

**Answer These Questions**:
- [ ] Paper deadline? (Conference vs. journal?)
- [ ] Dataset scope? (Action Genome only vs. + VSGR?)
- [ ] Baseline coverage? (Skip HyperGLM or implement it?)
- [ ] Timeline? (4 weeks minimum vs. 6 weeks comprehensive?)

**Decision Template**:
```
My Choices:
- Timeline Scenario: [A/B/C] (see DECISION_MATRIX.md)
- Datasets: [Action Genome only / + VSGR]
- Baselines: [Heuristic + VoT + Orion / + HyperGLM]
- Metrics: [Minimum (F1, Continuity, Causal) / + Recall@K / + Captioning]
```

---

### 3. Review Current Code (10 min)

Check what already exists:

```bash
# Review existing baselines
ls -lh research/baselines/
# vot_baseline.py (587 lines) ✅

ls -lh research/evaluation/
# heuristic_baseline.py (433 lines) ✅
# metrics.py (377 lines) ⚠️ needs extension
# ag_adapter.py ⚠️ needs verification

# Test existing baseline (if possible)
# python -m research.baselines.vot_baseline --video data/examples/video.mp4
```

---

### 4. Setup Your Environment (5 min)

```bash
# Ensure Orion is installed
pip install -e .

# Check dependencies for evaluation
pip install pandas matplotlib seaborn  # for analysis
pip install pycocoeval  # if using captioning metrics
pip install peft transformers  # if implementing HyperGLM

# Create results directory
mkdir -p research/results
```

---

## 📋 Implementation Checklist

Based on your chosen scenario, complete these tasks:

### Phase 1: Foundation (Week 1)
- [ ] **Day 1-2**: Extend `research/evaluation/metrics.py`
  - [ ] Add `compute_triplet_metrics()` function
  - [ ] Add `compute_entity_continuity()` function
  - [ ] Add `compute_recall_at_k()` function
  - [ ] Add `compute_causal_f1()` function

- [ ] **Day 2-3**: Create `research/evaluation/matcher.py`
  - [ ] Implement `EntityMatcher` class (IoU-based matching)
  - [ ] Implement `TripletMatcher` class
  - [ ] Test on sample data

- [ ] **Day 3-4**: Verify Action Genome loader
  - [ ] Download Action Genome dataset
  - [ ] Test `research/evaluation/ag_adapter.py`
  - [ ] Ensure GT entity + triplet loading works

- [ ] **Day 5**: Integration test
  - [ ] Run evaluation on 5 sample videos
  - [ ] Verify metrics computation end-to-end

### Phase 2: Baselines (Week 2)
- [ ] **Review existing Heuristic baseline**
  - [ ] Run on sample video
  - [ ] Ensure standard output format
  - [ ] Add CLI interface if missing

- [ ] **Review existing VoT baseline**
  - [ ] Run on sample video
  - [ ] Verify FastVLM → LLM pipeline
  - [ ] Standardize output format

- [ ] **[Optional] Start HyperGLM implementation**
  - [ ] Entity graph builder
  - [ ] Procedural graph (decide: trained vs. uniform prior)
  - [ ] HyperGraph sampler (Algorithm 1)
  - [ ] Visual encoder (CLIP + MLP)
  - [ ] LLM reasoning (Mistral + LoRA)
  - [ ] Integration + testing

### Phase 3: Ablations (Week 3-4)
- [ ] Add config flags to `orion/pipeline.py`
  - [ ] `--disable-tracking`
  - [ ] `--disable-semantic-uplift`
  - [ ] `--disable-llm`
  - [ ] `--disable-graph-reasoning` (if applicable)

- [ ] Create `research/ablations/runner.py`
  - [ ] Run all ablations systematically
  - [ ] Aggregate results

### Phase 4: Unified Runner (Week 4-5)
- [ ] Create `research/run_evaluation.py`
  - [ ] Orchestrate all baselines
  - [ ] Run on full dataset (50-100 videos)
  - [ ] Save results to JSON

- [ ] Create `research/analyze_results.py`
  - [ ] Generate comparison tables
  - [ ] Create plots (bar charts, etc.)
  - [ ] Generate LaTeX table for paper

### Phase 5: Paper Integration (Week 5-6)
- [ ] Write evaluation section (4.1-4.4)
- [ ] Create results table
- [ ] Add qualitative examples (figures)
- [ ] Error analysis

---

## 🎯 Immediate Next Steps (Today)

### Step 1: Choose Your Scenario (10 min)
Read `DECISION_MATRIX.md` and commit to:
- [ ] Scenario A (4 weeks, minimum viable)
- [ ] Scenario B (6 weeks, comprehensive)
- [ ] Scenario C (7-8 weeks, ideal)

### Step 2: Download Action Genome (30 min)
```bash
# Visit https://www.actiongenome.org/
# Download annotations and videos

# Expected structure:
# data/action_genome/
# ├── annotations/
# │   ├── train/
# │   ├── test/
# │   └── val/
# └── videos/
#     ├── video_001.mp4
#     └── ...
```

### Step 3: Test Existing Baselines (30 min)
```bash
# Test Heuristic baseline
python -m research.evaluation.heuristic_baseline \
    --video data/examples/video.mp4 \
    --output results/heuristic_test.json

# Test VoT baseline
python -m research.baselines.vot_baseline \
    --video data/examples/video.mp4 \
    --output results/vot_test.json

# Review outputs
cat results/heuristic_test.json
cat results/vot_test.json
```

### Step 4: Plan Your Week 1 (10 min)
Based on `IMPLEMENTATION_ROADMAP.md`, create your personal schedule:

```
Week 1 Plan:
- Mon: Extend metrics.py (triplet F1, entity continuity)
- Tue: Create matcher.py (entity matching logic)
- Wed: Verify AG adapter, test on 5 videos
- Thu: Integration testing, fix any issues
- Fri: Document progress, plan Week 2
```

---

## 📊 Expected Outputs (End State)

After completing implementation, you should have:

### 1. Results JSON Files
```json
// research/results/action_genome/orion_full.json
{
  "triplet_f1": 0.638,
  "precision": 0.671,
  "recall": 0.608,
  "entity_continuity": 0.915,
  "causal_f1": 0.612,
  "recall_at_10": 0.521,
  "recall_at_20": 0.689,
  "fps": 7.2,
  "num_videos": 100
}
```

### 2. Comparison Table (LaTeX)
```latex
\begin{table}[t]
\centering
\caption{Comparison on Action Genome test set}
\begin{tabular}{lcccc}
\toprule
Model & Triplet F1 & Causal F1 & Entity Cont. & FPS \\
\midrule
Heuristic & 31.2 & -- & 42.1 & 48 \\
LLM-Only (VoT) & 38.5 & 22.1 & -- & 5 \\
HyperGLM & 51.4 & 48.7 & -- & 2 \\
\textbf{Orion (ours)} & \textbf{63.8} & \textbf{61.2} & \textbf{91.5} & \textbf{7} \\
\bottomrule
\end{tabular}
\end{table}
```

### 3. Ablation Study Table
```
Ablation          | Triplet F1 | Causal F1 | Entity Cont.
------------------|------------|-----------|-------------
Orion (full)      | 63.8       | 61.2      | 91.5
- No tracking     | 56.1       | 42.8      | 63.4  ← Big drop!
- No uplift       | 44.0       | 33.7      | 58.2
- No LLM          | 39.9       | 25.6      | 72.0
```

### 4. Plots
- Bar chart: Baseline comparison (Triplet F1, Causal F1, Entity Continuity)
- Line chart: Ablation study (performance drop per component)
- Qualitative examples: Success/failure case visualizations

---

## 🆘 Troubleshooting

### "I don't know which scenario to choose"
**→** Go with **Scenario A (4 weeks)** if:
- Conference deadline in < 6 weeks
- You want quick results for initial submission
- You can add HyperGLM in camera-ready version

**→** Go with **Scenario B (6 weeks)** if:
- Journal submission or flexible deadline
- You want comprehensive SOTA comparison
- You have GPU resources for HyperGLM

---

### "Existing baselines don't run"
**→** Check these first:
```bash
# Verify Orion is installed
python -c "import orion; print(orion.__version__)"

# Check dependencies
python -c "import torch, transformers, clip"

# Test on minimal example
python -c "from orion import VideoPipeline; print('OK')"
```

---

### "Action Genome download is unclear"
**→** Resources:
- Official site: https://www.actiongenome.org/
- Paper: "Action Genome: Actions as Compositions of Spatio-temporal Scene Graphs" (CVPR 2020)
- GitHub: Search for "action-genome-dataset" (may have scripts)

---

### "I need help with implementation"
**→** Refer to:
1. `IMPLEMENTATION_ROADMAP.md` — Detailed code skeletons for each module
2. Existing code in `research/evaluation/` — Working examples
3. Orion core modules — Reference implementations (e.g., `orion/perception/types.py` for data structures)

---

## 📞 Critical Questions Before Starting

Before you begin Phase 1 implementation, answer:

1. [ ] **Have you read all 4 planning documents?** (At least summaries)
2. [ ] **Have you chosen a timeline scenario?** (A, B, or C)
3. [ ] **Can you run existing baselines on a sample video?**
4. [ ] **Do you have Action Genome dataset downloaded?**
5. [ ] **Do you have GPU access?** (Required for HyperGLM if implementing)
6. [ ] **What's your paper deadline?** (Determines urgency)

---

## ✅ Final Checklist (Before Implementation)

- [ ] Read `RESEARCH_SUMMARY.md` (5 min)
- [ ] Read `DECISION_MATRIX.md` and choose scenario (10 min)
- [ ] Download Action Genome dataset (30 min)
- [ ] Test existing baselines (30 min)
- [ ] Create Week 1 implementation plan (10 min)
- [ ] Set up research results directory structure
- [ ] Commit to timeline and communicate with co-authors

**Total Time**: ~90 minutes to be fully ready for implementation

---

## 🎉 You're Ready!

Once you've completed the checklist above, proceed to:

**→ `IMPLEMENTATION_ROADMAP.md` > Phase 1: Foundation (Week 1)**

Start with extending the metrics module (`research/evaluation/metrics.py`).

---

**Questions?** Review the detailed planning documents. They contain:
- Code skeletons for every module
- Expected input/output formats
- Troubleshooting guidance
- Timeline and effort estimates

**Good luck! 🚀**
