# Orion Deep Research Report: Optimization and Improvement (January 2026)

## 1. Executive Summary
This research concludes an end-to-end audit of the Orion v2 Precision Pipeline across multiple environments. While the "Precision Pipeline" (v yoloworld-precision) achieved a 92% reduction in false positives compared to the coarse baseline, a Gemini-powered audit reveals critical systemic failures in **temporal identity consistency**, **NMS redundancy**, and **spatial-semantic alignment**.

---

## 2. Perception Engine Audit (Stages 1-2)

### 2.1 Detection: YOLO-World v2
**Current Performance:**
- **Recall Issues**: Fails to capture prominent objects (e.g., central windows, furniture) unless explicitly prompted.
- **Hallucinations**: "Door" and "Person" boxes often encompass the entire frame (70%+ area) on architectural surfaces.
- **Ontology Gap**: "Notebook" is misclassified as "box" or "laptop," leading to downstream reasoning failures.

**Research Recommendations:**
- **Class-Specific Area Constraints**: Implement `max_area_ratio` per class (e.g., `person` < 50%, `laptop` < 30%) to kill global hallucinations.
- **Negative Sample Augmentation**: Introduce "Architectural Background" as a negative class to suppress wall/railing detections.
- **Open-Vocab Refinement**: Use a hierarchical prompting strategy: `[Object] -> [Specific Type]` (e.g., `Container -> Spiral Notebook`).

### 2.2 Re-ID: V-JEPA2 vs. CLIP
**Current Setup**: Using V-JEPA2 (1024-dim) for temporal track matching and CLIP (512-dim) for semantic labeling.
**Justification**:
- **Why V-JEPA2?**: Unlike CLIP (2D), V-JEPA2 is a video-native 3D-aware encoder. It captures temporal persistence and angular invariance, crucial for hand-object interactions and camera pans.
- **Why CLIP?**: Currently used as a legacy semantic labeling bridge.
**Better Alternatives**:
- **SigLIP (Meta/Google)**: Replaces CLIP with better zero-shot accuracy and lower latency.
- **DINOv2 (Meta)**: Superior for dense spatial features; if integrated with V-JEPA2, could resolve the "featureless wall" matching error.

---

## 3. Tracking and Memory Audit (Stage 2-3)

### 3.1 Temporal Tracking & Identity Drift
**Critical Failure**: Gemini audit found **100% ID switches** in high-motion sequences.
- **Root Cause**: The current tracker relies too heavily on appearance similarity without enough spatial "gating."
- **Evidence**: Track 34 jumped from a "hand" to a "desk" 1400 frames later because the wall background had a similar neutral color embedding.

**Proposed Solution (Motion + Spatial Gating)**:
- **Mahalanobis Distance**: Replace IoU with a Kalman-filtered spatial gate. Disallow matches that exceed a `max_pixel_distance` (e.g., 15% of frame width).
- **Strict Re-ID Thresholding**: Increase the V-JEPA2 cosine threshold from **0.65 to 0.82**.

### 3.2 Clustering (HDBSCAN vs. Cosine)
**Performance**:
- HDBSCAN currently creates redundant entities (e.g., 3-4 tracks for one notebook).
- **Redundancy Fix**: Implement **Class-Agnostic NMS** at the track level. If Track A (Box) and Track B (Laptop) overlap >80% over 5 frames, merge them into the more confident identity.

**Stage 2 vs Stage 3 Conflicts**:
There is currently a disconnect between "Perception Entities" (Phase 1C) and "Memory Objects" (Phase 2).
- **HDBSCAN (Phase 1C)**: Clusters observations into local entities.
- **Cosine Matcher (Phase 2)**: Re-clusters those entities into memory objects.
- **Research**: This dual-stage clustering introduces jitter. A better approach is **Hierarchical Cluster Merging**: use HDBSCAN for local temporal density and the Cosine Matcher with a **V-JEPA2 centroid** for global long-term identity.

---

## 4. Scene Graph & CIS Research (Stage 4)

### 4.1 Causal Influence Scoring (CIS) Formula Analysis
**Current Formula**: $CIS = 0.3T + 0.44S + 0.21M + 0.05Se + H$
- **Weakness**: $S$ (Spatial) and $M$ (Motion) are currently computed in 2D or semi-3D, leading to errors in "HELD_BY" relations when hands are merely in the foreground.
- **Optimization**:
    - **Depth Gating**: Apply a hard $Z$ (depth) disparity filter. If $|Z_a - Z_b| > 200mm$, the CIS score should be zeroed (prevents background-foreground hallucinations).
    - **Interaction Bonus (H)**: Increase weight of `hand_keypoint` proximity. Only trigger "HELD_BY" if $dist(hand, obj) < 50mm$ in 3D space.

### 4.2 Spatial Relations (ON, NEAR, INSIDE)
**Issue**: Current heuristic scene graphs show only 0.4 edges/frame.
- **Research**: The "ON" predicate fails because it requires exact 2D vertical stacking.
- **Fix**: Move to **3D AABB Volumetric Overlap**. Check if the 3D bounding box of $A$ is effectively resting on the surface plane of $B$.

### 4.3 CIS Integration Failures (Stage 4)
**Bug Identification**:
Runtime execution of `build_research_scene_graph` with `include_cis_edges=True` fails with an `AttributeError: '_EntityProxy' object has no attribute 'average_embedding'`.
- **Root Cause**: The abstraction layer in `scene_graph.py` does not correctly map memory object prototype embeddings to the CIS scorer's expected interface.
- **Systemic Impact**: Causal influence scoring is currently non-functional in the main pipeline.

---

## 5. Semantic Filtering & Ontology Validation

### 5.1 Justification: CLIP vs. Alternatives
Current Orion v2 uses **CLIP** for candidate labeling and **V-JEPA2** for Re-ID.
- **Why CLIP?**: 
    1. **Speed**: Lightweight ViT-B/32 is faster than larger VLMs.
    2. **Zero-Shot**: Established baseline for mapping boxes to natural language.
- **Why it is Lacking**: CLIP fails on "grained" distinctions (e.g., notebook vs. box) and has zero 3D awareness.
- **Better Options**:
    - **SigLIP**: Provides 10-15% better ZS accuracy for the same latency.
    - **Florence-2 (Microsoft)**: Much stronger at dense grounding (finding small objects like "barcode").
    - **V-JEPA2 (Meta)**: We should explore using V-JEPA2 for *both* labeling and Re-ID to maintain feature space consistency.

### 5.2 FastVLM + Sentence Transformers
- **Performance**: FastVLM (0.5B) is adequate for scene type classification but hallucinates details (e.g., repeating "Honor" or describing a bottle as a "foot").
- **Fix**: Implement a **VLM Voting** mechanism: ask the VLM to describe the crop 3 times and use Sentence Transformers to keep only the consensus (centroid) description.


---

## 6. Conclusion & Roadmap (Stages 1-4)

### Phase 1 (Optimization):
1. **Enable Class-Agnostic NMS** to eliminate duplicate tracks.
2. **Implement Spatial Gating** in `EnhancedTracker` to stop identity drift.
3. **Set Max-Area Filters** on `person` and `door` classes.

### Phase 2 (Architecture Improvement):
1. **Integrate 3D Depth** fully into the Scene Graph builder.
2. **Upgrade VLM Provider**: Fix the Keras dependency and implement repetition penalties in `FastVLM`.
3. **Replace CLIP** with **SigLIP** for candidate labeling accuracy.

### Phase 3 (Advanced Relations):
1. **Volumetric Spatial Predicates**: Range-based 3D checks for `NEAR` and `ON`.
2. **Temporal Edge Stability**: Implement a "cooling period" where an edge must persist for 5 frames before being committed to the graph.
