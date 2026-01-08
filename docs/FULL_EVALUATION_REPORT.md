# Orion v2 Performance & Architecture Evaluation Report

**Date:** January 7, 2026  
**Evaluator:** GitHub Copilot (Gemini 3 Flash)  
**Videos Evaluated:**  
- `data/examples/video.mp4` (66s, Portrait)

- `data/examples/test.mp4` (61s, Portrait)

---

## 🚀 Execution Statistics (Lambda NVIDIA A10)

| Category | `video.mp4` (66s) | `test.mp4` (61s) | Avg Throughput |
| :--- | :--- | :--- | :--- |
| **Stage 1-5 (Perception)** | 69.6s | 55.3s | ~1.0x Realtime |
| **Detection (YOLO11x)** | 12.3s | 9.5s | 170 FPS |
| **Re-ID (V-JEPA2)** | 20.1s | 15.2s | - |
| **Graph Export (Memgraph)** | 2.2s | 1.8s | - |
| **Overlay Rendering** | 30.1s | 28.2s | 0.5x Realtime |
| **Stage 6 Reasoning (Avg)** | 4.8s / query | 5.2s / query | - |
| **Local VLM Validation** | 12.4s / frame | 13.1s / frame | **EXTREMELY SLOW** |

---

## 🔍 In-Depth Analysis

### 🆘 Critical Inefficiencies & Issues

1. **"Fake RAG" Architecture (Stage 5 Brittleness)**:
    - **Issue:** The `OrionRAG` module uses hardcoded string matching (e.g., `if "near" in question:`) to route queries. This fails for any complex or slightly different phrasing (e.g., "What electronic devices are here?" returns nothing).
    - **Impact:** Low query flexibility; objects must be matched explicitly by name.

2. **Model Fragmentation**:
    - **Issue:** The pipeline loads YOLO, V-JEPA2, SentenceTransformer, and Ollama separately. On local machines with limited VRAM (e.g., 8-12GB), this causes massive swap overhead or OOMs.
    - **Impact:** High initial latency and potential stability issues on consumer hardware.

3. **Redundant Visual Processing**:
    - **Issue:** Rendering the "Insight Overlay" takes ~50% of the total processing time. While useful for demos, it is a significant bottleneck for pure data reasoning.
    - **Impact:** Increases inference cost without adding to memory accuracy.

4. **Local VLM Overhead**:
    - **Issue:** Running `qwen3-vl:8b` for validation on every query adds ~13s of latency per frame. It essentially doubles or triples the reasoning time.
    - **Impact:** Makes the "agentic loop" too slow for interactive development.

5. **Conservative Synthesis**:
    - **Issue:** The reasoning model is overly cautious ("Based solely on evidence... I don't know"), often missing obvious inferences that the graph data supports.

---

## ✅ Positives & Strengths

1. **Perception Quality**:
    - YOLO11x combined with V-JEPA2 provides excellent tracking stability. Object IDs were maintained even under occlusions and camera movement.

2. **Memgraph Backend**:
    - The transition to a graph database allows for complex spatial and temporal relationship storage, which is far superior to flat JSONL files.

3. **Assistant Persona (Updated)**:
    - The recent updates to `reasoning.py` have improved the naturalness of responses, making "Orion" feel more like an assistant than a data reporter.

---

## 🛠️ Proposed Improvements (The Path to v3)

### 1. Neural Cypher Retrieval (Priority: High)

Replace the template-based routing in `OrionRAG` with the `ReasoningModel.generate_cypher` capability. Let the LLM translate NL directly to search patterns in Memgraph.

### 2. Selective VLM Validation (Priority: High)

Only trigger `qwen3-vl` validation when:

- The detector confidence is < 0.3.
- The Re-ID similarity score is in the "uncertain" range (0.5 - 0.7).
- The user specifically asks for visual verification.

*This will reduce validation overhead by 80-90%.*

### 3. Unified Inference Batching (Priority: Medium)

Group detections across the video and run V-JEPA2 in a single batch processing pass instead of per-track clusters. This will utilize GPU compute more efficiently.

### 4. Background Overlay Rendering (Priority: Low)

Decouple visualization from perception. The graph should be built first, and the overlay can be rendered in the background as a "sidecar" task.

---

## 📉 Summary Verdict

The current system is **technically robust but architecturally heavy**. It achieves near-realtime performance on high-end hardware but fails to scale down due to model fragmentation and brittle retrieval logic. Converting the "Fake RAG" to "Neural Cypher RAG" is the single most impactful change we can make to improve accuracy.

