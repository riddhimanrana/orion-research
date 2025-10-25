# Orion Evaluation Implementation Roadmap

**Status**: Active Development  
**Last Updated**: October 25, 2025

This document provides step-by-step implementation guidance for the evaluation plan described in [`EVALUATION_PLAN.md`](./EVALUATION_PLAN.md).

---

## 🎯 Current Status Assessment

### ✅ Already Implemented
- **Heuristic Baseline**: `research/evaluation/heuristic_baseline.py` (433 lines)
- **VoT Baseline**: `research/baselines/vot_baseline.py` (587 lines)
- **Metrics Module**: `research/evaluation/metrics.py` (377 lines)
- **Action Genome Adapter**: `research/evaluation/ag_adapter.py`
- **Benchmark Runner**: `research/evaluation/benchmark_runner.py`

### ⚠️ Needs Refactoring/Extension
- **Metrics**: Current focus is on graph structure; need triplet F1, Recall@K, entity continuity
- **Dataset Loaders**: Need proper GT matching and evaluation protocols
- **Baselines**: VoT and Heuristic exist but may need updates for new evaluation protocol

### ❌ Not Yet Implemented
- **HyperGLM Baseline**: Full implementation from scratch (complex, ~1-2 weeks)
- **VSGR/ASPIRe Dataset Support**: New dataset loader needed
- **Ablation Framework**: Config-based ablation runner
- **Unified Experiment Orchestrator**: `research/run_evaluation.py`
- **Results Analysis Tools**: LaTeX table generation, plots

---

## 📋 Implementation Plan (Phased)

### Phase 1: Foundation (Week 1) — **START HERE**

#### 1.1 Extend Metrics Module
**File**: `research/evaluation/metrics.py`

Add these new metric functions:

```python
def compute_triplet_metrics(
    pred_triplets: List[Tuple[str, str, str]],
    gt_triplets: List[Tuple[str, str, str]],
    entity_map: Dict[str, str]
) -> Dict[str, float]:
    """
    Compute triplet-level P/R/F1
    
    Args:
        pred_triplets: List of (subj_id, pred, obj_id)
        gt_triplets: List of (gt_subj_id, gt_pred, gt_obj_id)
        entity_map: Mapping from pred entity IDs to GT entity IDs
    
    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    # Implementation: match triplets via entity_map
    pass


def compute_recall_at_k(
    pred_triplets: List[Tuple[str, str, str, float]],  # with scores
    gt_triplets: List[Tuple[str, str, str]],
    k_values: List[int] = [10, 20, 50]
) -> Dict[str, float]:
    """
    Compute Recall@K for scene graph generation
    
    Returns:
        {"R@10": float, "R@20": float, "R@50": float}
    """
    pass


def compute_entity_continuity(
    pred_tracks: List[EntityTrack],
    gt_tracks: List[EntityTrack],
    iou_thresh: float = 0.5
) -> float:
    """
    Compute percentage of entities correctly tracked across frames
    
    Returns:
        Continuity score (0.0 - 1.0)
    """
    pass


def compute_causal_f1(
    pred_causal_edges: List[Tuple[str, str]],
    gt_causal_edges: List[Tuple[str, str]]
) -> Dict[str, float]:
    """
    Compute P/R/F1 for causal relationships
    
    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    pass
```

**TODO**: Extend existing `GraphMetrics` class or create new `TripletMetrics` class

---

#### 1.2 Create Entity Matcher Module
**File**: `research/evaluation/matcher.py` (NEW)

```python
"""
Entity and Triplet Matching Logic
==================================

Handles IoU-based entity matching and triplet alignment for evaluation.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    
    def iou(self, other: 'BBox') -> float:
        """Compute intersection-over-union with another bbox"""
        # Standard IoU calculation
        pass


@dataclass
class Entity:
    """Represents a detected/tracked entity"""
    id: str
    class_label: str
    bbox: BBox
    frame_idx: int
    confidence: float = 1.0


class EntityMatcher:
    """Matches predicted entities to ground-truth entities"""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
    
    def match_frame(
        self,
        pred_entities: List[Entity],
        gt_entities: List[Entity]
    ) -> Dict[str, str]:
        """
        Match predicted entities to GT entities in a single frame
        
        Returns:
            Dict mapping pred_entity_id → gt_entity_id
        """
        # Hungarian algorithm or greedy matching based on IoU + class
        pass
    
    def match_video(
        self,
        pred_tracks: List[List[Entity]],  # per-frame entities
        gt_tracks: List[List[Entity]]
    ) -> Dict[str, str]:
        """
        Match entity tracks across entire video
        
        Returns:
            Global entity ID mapping
        """
        pass


class TripletMatcher:
    """Matches predicted triplets to ground-truth triplets"""
    
    def __init__(self, entity_map: Dict[str, str]):
        self.entity_map = entity_map
    
    def match(
        self,
        pred_triplet: Tuple[str, str, str],
        gt_triplets: List[Tuple[str, str, str]]
    ) -> bool:
        """
        Check if predicted triplet matches any GT triplet
        
        Args:
            pred_triplet: (subj_id, predicate, obj_id)
            gt_triplets: List of GT triplets
        
        Returns:
            True if match found (TP), False otherwise (FP)
        """
        subj_id, pred_label, obj_id = pred_triplet
        
        # Map predicted entity IDs to GT IDs
        gt_subj = self.entity_map.get(subj_id)
        gt_obj = self.entity_map.get(obj_id)
        
        if gt_subj is None or gt_obj is None:
            return False  # No entity match → FP
        
        # Check if (gt_subj, pred_label, gt_obj) in GT triplets
        return (gt_subj, pred_label, gt_obj) in gt_triplets
```

---

#### 1.3 Update Action Genome Adapter
**File**: `research/evaluation/ag_adapter.py`

**Check current implementation** and ensure it provides:
- Ground-truth triplet loading per frame
- Ground-truth entity tracks (bbox + class + ID)
- Temporal relation annotations (if available)

**Required Interface**:
```python
class ActionGenomeDataset:
    def __init__(self, data_dir: Path, split: str = "test"):
        pass
    
    def get_video_ids(self) -> List[str]:
        """Return all video IDs in this split"""
        pass
    
    def get_gt_entities(self, video_id: str) -> List[List[Entity]]:
        """Return per-frame GT entities for video"""
        pass
    
    def get_gt_triplets(self, video_id: str) -> List[List[Tuple[str, str, str]]]:
        """Return per-frame GT triplets"""
        pass
    
    def get_gt_causal_edges(self, video_id: str) -> List[Tuple[str, str]]:
        """Return causal edges if annotated"""
        pass
```

---

### Phase 2: Baselines Review & Update (Week 2)

#### 2.1 Review Existing Heuristic Baseline
**File**: `research/evaluation/heuristic_baseline.py`

**Action Items**:
1. Read through current implementation
2. Ensure it outputs in standard format:
   ```python
   {
       "entities": List[Entity],
       "triplets": List[Tuple[str, str, str, float]],  # with scores
       "causal_edges": List[Tuple[str, str]]
   }
   ```
3. Add CLI interface: `python -m research.evaluation.heuristic_baseline --video <path>`

---

#### 2.2 Review Existing VoT Baseline
**File**: `research/baselines/vot_baseline.py`

**Action Items**:
1. Verify it uses FastVLM for captioning
2. Ensure it extracts triplets via LLM prompt
3. Standardize output format (same as above)
4. Add evaluation mode (no training, pure inference)

---

#### 2.3 Create HyperGLM Baseline (Complex Task)
**Estimated Time**: 1-2 weeks  
**Location**: `research/baselines/hyperglm/`

**Sub-modules to implement**:

##### 2.3.1 Entity Graph Builder
**File**: `research/baselines/hyperglm/entity_graph.py`

```python
"""
Per-frame entity scene graph construction for HyperGLM
"""

from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn


@dataclass
class EntityNode:
    """Single entity in a frame"""
    bbox: Tuple[float, float, float, float]
    class_label: str
    visual_feature: torch.Tensor  # ROI feature from detector


@dataclass
class RelationEdge:
    """Relation between two entities"""
    subject_idx: int
    object_idx: int
    predicate_logits: torch.Tensor  # Raw scores for all predicates


class EntityGraphBuilder:
    """Builds entity scene graph for a single frame"""
    
    def __init__(self, relation_classifier: nn.Module):
        self.relation_classifier = relation_classifier
    
    def build(self, detections: List[EntityNode]) -> Tuple[List[EntityNode], List[RelationEdge]]:
        """
        Build entity graph G_t = (V^e_t, E^e_t)
        
        Args:
            detections: List of detected entities in frame
        
        Returns:
            (nodes, edges) where edges contain relation predictions
        """
        nodes = detections
        edges = []
        
        # Pairwise relation prediction
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                # Concatenate ROI features
                feat_i = nodes[i].visual_feature
                feat_j = nodes[j].visual_feature
                
                # Predict relation
                logits = self.relation_classifier(torch.cat([feat_i, feat_j]))
                edges.append(RelationEdge(i, j, logits))
        
        return nodes, edges
```

##### 2.3.2 Procedural Graph Builder
**File**: `research/baselines/hyperglm/procedural_graph.py`

```python
"""
Procedural graph modeling relation transitions (Equation 3-4 from paper)
"""

import numpy as np
from typing import Dict, List, Tuple


class ProceduralGraph:
    """
    Temporal relation transition model
    
    Computes w(r_m, r_n) = P(relation r_n at t+1 | relation r_m at t)
    """
    
    def __init__(self, num_relations: int):
        self.num_relations = num_relations
        self.transition_matrix = np.zeros((num_relations, num_relations))
    
    def train_from_annotations(self, video_annotations: List[Dict]):
        """
        Compute transition matrix from training videos
        
        Args:
            video_annotations: List of videos with frame-by-frame relation labels
        """
        counts = np.zeros((self.num_relations, self.num_relations))
        
        for video in video_annotations:
            relations = video["relations_per_frame"]  # List[List[int]]
            
            for t in range(len(relations) - 1):
                for r_m in relations[t]:
                    for r_n in relations[t+1]:
                        counts[r_m, r_n] += 1
        
        # Normalize to probabilities (Equation 4)
        self.transition_matrix = counts / (counts.sum(axis=1, keepdims=True) + 1e-8)
    
    def get_transition_prob(self, r_m: int, r_n: int) -> float:
        """Get probability of r_m → r_n transition"""
        return self.transition_matrix[r_m, r_n]
```

**⚠️ Critical Decision**: Do we train this from Action Genome training data, or use a uniform prior?  
- **With training**: Fair comparison to paper, better performance  
- **Uniform prior**: Simpler, but weakens HyperGLM performance

---

##### 2.3.3 HyperGraph Sampler (Algorithm 1)
**File**: `research/baselines/hyperglm/hypergraph.py`

```python
"""
HyperGraph construction and random-walk sampling (Algorithm 1 from paper)
"""

import random
from dataclasses import dataclass
from typing import List, Set, Tuple
import networkx as nx


@dataclass
class HyperEdge:
    """A hyperedge connecting multiple nodes"""
    nodes: Set[str]  # Set of node IDs
    edge_type: str  # "entity", "relation", "procedural"


class HyperGraph:
    """Unified hypergraph H = (V_H, E_H)"""
    
    def __init__(self):
        self.nodes = {}  # node_id → node data
        self.edges = []  # List of HyperEdge
    
    def add_entity_graph(self, frame_idx: int, entity_graph):
        """Add per-frame entity graph to hypergraph"""
        # Add entity nodes from this frame
        pass
    
    def add_procedural_edges(self, procedural_graph):
        """Add procedural relation transition edges"""
        pass
    
    def sample_hyperedges(self, N_w: int = 60, N_l: int = 7) -> List[HyperEdge]:
        """
        Random-walk sampling (Algorithm 1 from HyperGLM paper)
        
        Args:
            N_w: Number of random walks
            N_l: Walk length
        
        Returns:
            List of sampled hyperedges
        """
        sampled = []
        
        for _ in range(N_w):
            # Start from random node
            current = random.choice(list(self.nodes.keys()))
            walk_nodes = {current}
            
            for step in range(N_l):
                # Alternate: node → hyperedge → node
                # (Implementation per Algorithm 1 in paper)
                pass
            
            sampled.append(HyperEdge(walk_nodes, "sampled"))
        
        return sampled
```

---

##### 2.3.4 Visual Encoder (CLIP + MLP)
**File**: `research/baselines/hyperglm/visual_encoder.py`

```python
"""
CLIP-based visual encoding with MLP projection to LLM token space
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class VisualEncoder(nn.Module):
    """CLIP ViT-L-336 + MLP projector"""
    
    def __init__(self, llm_hidden_size: int = 4096):
        super().__init__()
        
        # Load CLIP
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        
        # MLP projector (2-layer as per paper)
        clip_dim = self.clip.config.vision_config.hidden_size  # 1024 for ViT-L
        self.projector = nn.Sequential(
            nn.Linear(clip_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
    
    def encode_frame(self, image) -> torch.Tensor:
        """
        Encode frame to LLM token space
        
        Returns:
            Tensor of shape (num_tokens, llm_hidden_size)
            Paper uses ~10 tokens: 1 CLS + 9 pooled
        """
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(**inputs)
        
        # Get CLS token + patch tokens (simplified; paper uses pooling)
        vision_tokens = vision_outputs.last_hidden_state  # (1, num_patches, 1024)
        
        # Project to LLM space
        projected = self.projector(vision_tokens)  # (1, num_patches, llm_dim)
        
        return projected.squeeze(0)
```

---

##### 2.3.5 LLM Reasoning Module
**File**: `research/baselines/hyperglm/llm_reasoning.py`

```python
"""
LLM-based reasoning with LoRA for scene graph tasks
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class LLMReasoner:
    """Mistral-7B-Instruct with LoRA adapters"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Apply LoRA (rank=128, scale=256 per paper)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=128,
            lora_alpha=256,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        self.llm = get_peft_model(self.llm, peft_config)
    
    def generate_triplets(
        self,
        visual_tokens: torch.Tensor,
        hypergraph_prompt: str,
        task: str = "sgg"
    ) -> str:
        """
        Generate scene graph triplets via LLM
        
        Args:
            visual_tokens: Projected CLIP embeddings (num_tokens, hidden_dim)
            hypergraph_prompt: Text description of sampled hyperedges
            task: "sgg" (generation) or "sga" (anticipation)
        
        Returns:
            LLM output text with triplets
        """
        # Build prompt
        if task == "sgg":
            query = "Generate all <subject, predicate, object> triplets for this frame."
        elif task == "sga":
            query = "Predict relations in the next frame."
        
        full_prompt = f"{hypergraph_prompt}\n\n{query}"
        
        # Tokenize text prompt
        text_inputs = self.tokenizer(full_prompt, return_tensors="pt")
        
        # Interleave visual tokens with text tokens (implementation varies)
        # Simplified: prepend visual tokens
        input_embeds = torch.cat([
            visual_tokens.unsqueeze(0),  # Add batch dim
            self.llm.get_input_embeddings()(text_inputs.input_ids)
        ], dim=1)
        
        # Generate
        outputs = self.llm.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=256,
            temperature=0.7
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

##### 2.3.6 Full Pipeline Orchestration
**File**: `research/baselines/hyperglm/inference.py`

```python
"""
Full HyperGLM inference pipeline
"""

from pathlib import Path
from typing import List, Tuple
import cv2

from .entity_graph import EntityGraphBuilder
from .procedural_graph import ProceduralGraph
from .hypergraph import HyperGraph
from .visual_encoder import VisualEncoder
from .llm_reasoning import LLMReasoner


class HyperGLMPipeline:
    """End-to-end HyperGLM inference"""
    
    def __init__(
        self,
        detector,  # Object detector (Faster R-CNN or YOLO11x adapter)
        procedural_graph: ProceduralGraph,
        device: str = "cuda"
    ):
        self.detector = detector
        self.procedural_graph = procedural_graph
        
        self.entity_graph_builder = EntityGraphBuilder(...)
        self.visual_encoder = VisualEncoder().to(device)
        self.llm_reasoner = LLMReasoner()
    
    def process_video(self, video_path: Path) -> Dict:
        """Run full HyperGLM pipeline on video"""
        
        # 1. Extract frames
        frames = self._extract_frames(video_path)
        
        # 2. Detect objects
        detections = [self.detector(f) for f in frames]
        
        # 3. Build entity graphs per frame
        entity_graphs = [
            self.entity_graph_builder.build(d) for d in detections
        ]
        
        # 4. Build unified hypergraph
        H = HyperGraph()
        for i, eg in enumerate(entity_graphs):
            H.add_entity_graph(i, eg)
        H.add_procedural_edges(self.procedural_graph)
        
        # 5. Sample hyperedges
        sampled_hyperedges = H.sample_hyperedges(N_w=60, N_l=7)
        
        # 6. Encode frames with CLIP
        visual_tokens = [self.visual_encoder.encode_frame(f) for f in frames]
        
        # 7. Build hypergraph text prompt
        hypergraph_prompt = self._hyperedges_to_text(sampled_hyperedges)
        
        # 8. LLM reasoning
        outputs = self.llm_reasoner.generate_triplets(
            visual_tokens=visual_tokens[-1],  # Last frame
            hypergraph_prompt=hypergraph_prompt,
            task="sgg"
        )
        
        # 9. Parse triplets
        triplets = self._parse_triplets(outputs)
        
        return {
            "entities": detections[-1],
            "triplets": triplets,
            "hypergraph": sampled_hyperedges
        }
    
    def _parse_triplets(self, llm_output: str) -> List[Tuple[str, str, str]]:
        """Parse LLM output into structured triplets"""
        # Use regex or simple parsing
        pass
```

---

**Estimated Implementation Time for HyperGLM**: 1-2 weeks  
**Complexity**: High (requires CLIP, LLM, LoRA, graph sampling)  
**Optional**: Can defer if time-constrained; focus on Heuristic + VoT + Orion first

---

### Phase 3: Dataset Integration (Week 3)

#### 3.1 Finalize Action Genome Support
**Files**: `research/datasets/action_genome.py`, `research/evaluation/ag_adapter.py`

**Tasks**:
1. Download Action Genome annotations
2. Verify GT triplet format
3. Test entity matching on 5 sample videos
4. Confirm metrics computation works end-to-end

---

#### 3.2 Add VSGR/ASPIRe Support (Optional)
**File**: `research/datasets/vsgr.py` (NEW)

```python
"""
VSGR (ASPIRe) Dataset Loader
"""

from pathlib import Path
from typing import List, Dict


class VSGRDataset:
    """VSGR benchmark dataset (ASPIRe subset)"""
    
    def __init__(self, data_dir: Path, subset: str = "aspire"):
        self.data_dir = data_dir
        self.subset = subset
        
        # Load annotations
        self.annotations = self._load_annotations()
    
    def _load_annotations(self) -> Dict:
        """Load VSGR/ASPIRe annotation files"""
        # Parse annotation format (depends on released format)
        pass
    
    def get_video_ids(self) -> List[str]:
        """Return all video IDs"""
        pass
    
    # Similar interface to ActionGenomeDataset
```

**Decision Point**: Only implement if Action Genome results insufficient or need cross-domain evaluation.

---

### Phase 4: Ablation Framework (Week 4)

#### 4.1 Add Config Flags to Orion
**Files**: Modify `orion/pipeline.py`, `orion/perception/config.py`, `orion/semantic/config.py`

**Add flags**:
```python
@dataclass
class PipelineConfig:
    # Existing fields...
    
    # Ablation flags
    disable_tracking: bool = False        # No entity ID linking
    disable_semantic_uplift: bool = False # YOLO classes only, no VLM
    disable_graph_reasoning: bool = False # No hypergraph/attention
    disable_llm: bool = False             # Visual→symbolic only
```

**Implement logic**:
```python
# In VideoPipeline.process_video()
if config.disable_tracking:
    # Skip entity tracking, treat each detection independently
    perception_result.entities = [obs for obs in perception_result.raw_observations]

if config.disable_semantic_uplift:
    # Skip VLM description generation
    for entity in perception_result.entities:
        entity.description = entity.object_class  # Use YOLO class only

# etc.
```

---

#### 4.2 Create Ablation Runner
**File**: `research/ablations/runner.py` (NEW)

```python
"""
Ablation Study Runner
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict

from orion import VideoPipeline, PipelineConfig


ABLATIONS = {
    "full": {},
    "no_tracking": {"disable_tracking": True},
    "no_uplift": {"disable_semantic_uplift": True},
    "no_llm": {"disable_llm": True},
    "no_graph": {"disable_graph_reasoning": True},
}


def run_ablation_study(
    dataset_name: str,
    video_ids: List[str],
    output_dir: Path
) -> pd.DataFrame:
    """
    Run all ablations systematically
    
    Returns:
        DataFrame with columns: [ablation, triplet_f1, causal_f1, entity_cont, ...]
    """
    results = []
    
    for ablation_name, config_mods in ABLATIONS.items():
        print(f"\n{'='*80}")
        print(f"Running ablation: {ablation_name}")
        print(f"{'='*80}")
        
        # Create config
        config = PipelineConfig(**config_mods)
        pipeline = VideoPipeline(config)
        
        # Run on all videos
        all_metrics = []
        for video_id in video_ids:
            video_path = get_video_path(dataset_name, video_id)
            
            # Run Orion
            orion_output = pipeline.process_video(str(video_path))
            
            # Evaluate against GT
            metrics = evaluate_against_gt(dataset_name, video_id, orion_output)
            all_metrics.append(metrics)
        
        # Aggregate
        avg_metrics = aggregate_metrics(all_metrics)
        avg_metrics["ablation"] = ablation_name
        results.append(avg_metrics)
    
    return pd.DataFrame(results)
```

---

### Phase 5: Unified Experiment Runner (Week 5)

#### 5.1 Create Main Experiment Script
**File**: `research/run_evaluation.py` (NEW)

```python
"""
Unified Evaluation Runner

Orchestrates all baselines, datasets, and ablations.

Usage:
    python research/run_evaluation.py --dataset action_genome --output results/
    python research/run_evaluation.py --dataset vsgr --baseline hyperglm
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from research.datasets.action_genome import ActionGenomeDataset
from research.baselines.heuristic import HeuristicBaseline
from research.baselines.vot_baseline import VOTBaseline
from research.baselines.hyperglm import HyperGLMPipeline
from research.ablations.runner import run_ablation_study
from orion import VideoPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["action_genome", "vsgr"], required=True)
    parser.add_argument("--baselines", nargs="+", default=["all"],
                       choices=["heuristic", "vot", "hyperglm", "orion", "all"])
    parser.add_argument("--ablations", action="store_true", help="Run ablation studies")
    parser.add_argument("--output", type=Path, default=Path("research/results"))
    parser.add_argument("--max-videos", type=int, help="Limit number of videos")
    
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset == "action_genome":
        dataset = ActionGenomeDataset(Path("data/action_genome"), split="test")
    else:
        dataset = VSGRDataset(Path("data/vsgr_aspire"))
    
    video_ids = dataset.get_video_ids()[:args.max_videos]
    
    # Run baselines
    results = {}
    
    if "heuristic" in args.baselines or "all" in args.baselines:
        print("\n" + "="*80)
        print("BASELINE: Heuristic")
        print("="*80)
        results["heuristic"] = run_heuristic_baseline(dataset, video_ids)
    
    if "vot" in args.baselines or "all" in args.baselines:
        print("\n" + "="*80)
        print("BASELINE: LLM-Only (VoT)")
        print("="*80)
        results["vot"] = run_vot_baseline(dataset, video_ids)
    
    if "hyperglm" in args.baselines or "all" in args.baselines:
        print("\n" + "="*80)
        print("BASELINE: HyperGLM")
        print("="*80)
        results["hyperglm"] = run_hyperglm_baseline(dataset, video_ids)
    
    if "orion" in args.baselines or "all" in args.baselines:
        print("\n" + "="*80)
        print("ORION (FULL)")
        print("="*80)
        results["orion_full"] = run_orion_baseline(dataset, video_ids)
    
    # Run ablations
    if args.ablations:
        print("\n" + "="*80)
        print("ABLATION STUDY")
        print("="*80)
        ablation_df = run_ablation_study(args.dataset, video_ids, args.output)
        ablation_df.to_csv(args.output / "ablations.csv", index=False)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output / f"results_{args.dataset}_{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Generate comparison table
    generate_comparison_table(results, args.output / "comparison.tex")


def run_heuristic_baseline(dataset, video_ids):
    """Run heuristic baseline on all videos"""
    from research.evaluation.heuristic_baseline import HeuristicBaseline
    
    baseline = HeuristicBaseline()
    all_metrics = []
    
    for video_id in video_ids:
        video_path = dataset.get_video_path(video_id)
        
        # Run baseline
        output = baseline.process(video_path)
        
        # Evaluate
        metrics = evaluate_output(dataset, video_id, output)
        all_metrics.append(metrics)
    
    return aggregate_metrics(all_metrics)


# Similar functions for other baselines...
```

---

#### 5.2 Create Results Analysis Module
**File**: `research/analyze_results.py` (NEW)

```python
"""
Results Analysis and Visualization

Generates LaTeX tables, plots, and qualitative examples.
"""

import pandas as pd
from pathlib import Path
from typing import Dict


def generate_comparison_table(results: Dict, output_path: Path):
    """
    Generate LaTeX comparison table
    
    Args:
        results: Dict with keys = baseline names, values = metrics dict
        output_path: Where to save .tex file
    """
    df = pd.DataFrame(results).T
    
    # Format as LaTeX
    latex = df.to_latex(
        float_format="%.2f",
        column_format="l" + "c" * len(df.columns),
        caption="Comparison of Orion against baselines",
        label="tab:results"
    )
    
    output_path.write_text(latex)
    print(f"✅ LaTeX table saved to {output_path}")


def plot_metrics(results_csv: Path, output_dir: Path):
    """Generate bar charts comparing baselines"""
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(results_csv)
    
    metrics = ["triplet_f1", "causal_f1", "entity_continuity"]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for ax, metric in zip(axes, metrics):
        df.plot(x="baseline", y=metric, kind="bar", ax=ax)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score")
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=300)
    print(f"✅ Plot saved to {output_dir / 'metrics_comparison.png'}")


def generate_qualitative_examples(dataset, video_ids: List[str], output_dir: Path):
    """Generate visual examples of scene graphs"""
    # Implementation: visualize top-3 best and top-3 worst predictions
    pass
```

---

## 🎯 Quick Start Commands

Once implemented, you should be able to run:

```bash
# Full evaluation on Action Genome (all baselines)
python research/run_evaluation.py \
    --dataset action_genome \
    --baselines all \
    --ablations \
    --output results/action_genome/

# Only Orion vs Heuristic (fast test)
python research/run_evaluation.py \
    --dataset action_genome \
    --baselines orion heuristic \
    --max-videos 10

# Generate analysis
python research/analyze_results.py \
    --results results/action_genome/results_*.json \
    --output results/action_genome/analysis/
```

---

## ⚠️ Critical Decisions Needed

1. **HyperGLM Procedural Graph**: Train from AG training data or use uniform prior?
2. **VSGR Dataset**: Implement now or defer until Action Genome done?
3. **Evaluation Metrics**: Focus on triplet F1 + entity continuity, or also add captioning metrics?
4. **Compute Budget**: Can we run Mistral-7B for HyperGLM baseline?

---

## 📅 Timeline Summary

| Week | Phase | Deliverables |
|------|-------|-------------|
| 1 | Foundation | Metrics module, Entity matcher, AG adapter verified |
| 2 | Baselines | Heuristic + VoT reviewed/updated |
| 3 | Dataset | Action Genome fully integrated, sample evaluation run |
| 4 | Ablations | Ablation framework + runner |
| 5 | Integration | Unified experiment runner, results on 50+ videos |
| 6 | HyperGLM (opt) | HyperGLM baseline implementation |
| 7 | Analysis | LaTeX tables, plots, qualitative examples for paper |

**Total**: 5-7 weeks (depending on HyperGLM inclusion)

---

**Next Action**: Review this roadmap and confirm which components to prioritize. Then start with Phase 1.1 (extending metrics module).
