# Ego4D Benchmarks: Research Notes + Orion Mapping

> Status: Ego4D S3 access is currently blocked in our Lambda environment (HTTP 403). This document focuses on (a) what the benchmarks measure, (b) the publicly available baseline repositories / model families, and (c) how Orion maps onto each benchmark so we can keep building and testing on non-Ego4D data in the meantime.

## Benchmarks at a glance

| Benchmark | What it tests | Primary outputs | “If we had to build it in Orion” (1-liner) |
| --- | --- | --- | --- |
| Episodic Memory | Making *past* video queryable | Temporal window(s), and for visual queries: spatio-temporal object tracks (+ optional 3D displacement) | Build a query-time retrieval layer over Orion’s long-term tracks + scene memory |
| Hands & Objects | Understanding and localizing object state changes | PNR keyframe, object-of-change bbox, state-change label | Add state-change events on top of tracked objects + hand/tool context |
| Forecasting | Predicting *future* interactions and actions | Future object(s), verb(s), time-to-contact; future action sequence; trajectories; hands | Add intent/anticipation heads over track histories + scene graph dynamics |
| AV Diarization | Who spoke, when, and where (multi-modal) | Face tracks, active speaker labels, diarization segments, transcripts | Integrate audio pipeline (VAD/ASR) + face tracking + ASD |
| Social Interactions | Whether someone is looking/talking to the wearer | Per-face binary LAM/TTM labels over time | Add gaze/attention + directed-speech classification on top of AVD |

## Primary sources

### Documentation pages

- Benchmarks overview (index): <https://ego4d-data.org/docs/benchmarks/overview/>
- Episodic Memory: <https://ego4d-data.org/docs/benchmarks/episodic-memory/>
- Forecasting: <https://ego4d-data.org/docs/benchmarks/forecasting/>
- Hands & Objects: <https://ego4d-data.org/docs/benchmarks/hands-and-objects/>
- AV Diarization: <https://ego4d-data.org/docs/benchmarks/av-diarization/>
- Social Interactions: <https://ego4d-data.org/docs/benchmarks/social/>
- FHO Overview (annotation + taxonomy): <https://ego4d-data.org/docs/tutorials/FHO_Overview/>
- Challenge hub (EvalAI links): <https://ego4d-data.org/docs/challenge/>

### Benchmark repositories (baselines + evaluation)

- Episodic Memory repo: <https://github.com/EGO4D/episodic-memory>
- Forecasting repo: <https://github.com/EGO4D/forecasting>
- Hands & Objects repo: <https://github.com/EGO4D/hands-and-objects>
- AV Diarization repo: <https://github.com/EGO4D/audio-visual>
- Social repo: <https://github.com/EGO4D/social-interactions>

### Core paper

- Ego4D paper (CVPR’22): <https://arxiv.org/abs/2110.07058>

---

## 1) Episodic Memory (NLQ / VQ2D / VQ3D / MQ)

### Task definition (Episodic Memory)

From the Ego4D docs, Episodic Memory asks: given an egocentric video and a query, *localize where the answer can be seen*.

- **Natural Language Queries (NLQ)**: query in text → output temporal window.
- **Visual Queries (VQ)**:
  - **VQ2D**: query is an image crop of an object instance → output the last occurrence of that instance as a **response track** (temporal + 2D bbox per frame).
  - **VQ3D**: additionally output a **3D displacement vector** from query frame camera center to the object in 3D.
- **Moments Queries (MQ)**: query is an activity label (“moment”) → output all temporal windows where it occurs.

Official page: <https://ego4d-data.org/docs/benchmarks/episodic-memory/>

### Existing baselines & model families

#### VQ2D

The `VQ2D` README explicitly reports multiple baselines and metrics (validation):

- README (raw): <https://github.com/EGO4D/episodic-memory/raw/refs/heads/main/VQ2D/README.md>
- It frames the pipeline as:
  1) precompute per-frame bbox proposals + query similarity (expensive), then
  2) peak detection + bidirectional tracking to form the response track.

**Reported metrics in the README:**

- `stAP@0.25`, `stAP`, `tAP@0.25`, `tAP`, `recall%`, `success%`.

**Reported validation results (from the README):**

| Method | stAP@0.25 | stAP | tAP@0.25 | tAP | recall% | success% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SiamRCNN | 0.153 | 0.058 | 0.225 | 0.134 | 32.919 | 43.244 |
| Improved Baselines | 0.195 | 0.078 | 0.258 | 0.157 | 37.882 | 47.903 |
| Negative Frames Matter | 0.189 | 0.075 | 0.255 | 0.154 | 37.666 | 47.681 |

The README also points to prior “challenge winner” code: <https://github.com/facebookresearch/vq2d_cvpr>

**Interpretation (what works in practice):**

- Strong per-frame object proposals are necessary but not sufficient.
- The temporal “last seen” aspect is often handled by peak detection over similarity scores and *track stitching*, not by a single end-to-end network.

#### NLQ

In the `episodic-memory` repo, NLQ baselines are organized under `NLQ/`:

- VSLNet-based baseline: <https://github.com/EGO4D/episodic-memory/tree/main/NLQ/VSLNet>
- 2D-TAN baseline: <https://github.com/EGO4D/episodic-memory/tree/main/NLQ/2D-TAN>
VSLNet’s README shows the official evaluation script usage with thresholds (e.g., `--thresholds 0.3 0.5` and `--topK 1 3 5`), implying the common NLQ metric family:

- Recall@K at temporal IoU thresholds (e.g., 0.3, 0.5)

(Exact headline numbers are not consistently surfaced in the GitHub rendered snippets we fetched, but the repo includes runnable evaluation code.)

#### VQ3D

VQ3D is an extension that introduces:

- camera pose estimation (COLMAP / SfM),
- depth estimation,
- 3D displacement evaluation.

Baseline workflow is documented here:

- <https://github.com/EGO4D/episodic-memory/blob/main/VQ3D/README.md>

#### MQ

MQ baseline code lives here:

- <https://github.com/EGO4D/episodic-memory/blob/main/MQ/README.md>

The MQ baseline repo references an underlying temporal localization backbone (VSGN). It’s best viewed as “temporal action localization over long egocentric video”, not object tracking.

### Orion mapping (Episodic Memory)

#### What Orion already has that’s directly useful

- Long-lived object tracks + embeddings (Phase 1 + Phase 2 Re-ID)
- Scene graph export per frame (relations can become query-time features)
- Depth (DepthAnything) + early SLAM integration hooks

#### How Orion can attack Episodic Memory tasks

- **VQ2D (strong fit):**
  - Use Orion’s track gallery embeddings as the “proposal set” (instead of generic proposals).
  - For a query crop, embed with the same embedding model and nearest-neighbor over track gallery.
  - Return the best-matching track segment(s) as the response track.
- **VQ3D (good fit if SLAM stabilizes):**
  - Once camera poses are available (from SLAM), combine depth + pose + bbox center ray to estimate 3D point and compute displacement.
- **NLQ / MQ (needs new capability):**
  - Add a video-language index:
    - per-window captions (MLX-VLM / other VLM) → text embeddings
    - retrieve candidate windows → refine temporal boundaries
  - MQ likely needs an action recognition/temporal localization model or a VLM-driven classifier over a fixed taxonomy.

---

## 2) Hands & Objects (state change)

### Task definition (Hands & Objects)

The benchmark targets **object state changes** and defines three tasks:

1) **Point-of-no-return (PNR) temporal localization**: keyframe when the state change begins.
2) **State change object detection (SCOD)**: bbox of the object undergoing state change (given pre/PNR/post frames).
3) **Object state change classification (OSCC)**: classify whether a state change happened (and/or its type, depending on setup).

Official page: <https://ego4d-data.org/docs/benchmarks/hands-and-objects/>

### Annotation notes (useful for modeling)

From the official benchmark description, annotations for state changes include:

- three key frames per interaction: **pre**, **PNR**, **post**,
- bounding boxes for **hands**, **tools**, and **objects** in each of those frames,
- labels for **state change types** (e.g., remove/burn/…), plus action verbs and object nouns.

The broader Forecasting Hands & Objects (FHO) documentation provides additional context on how these clips are sampled from narrations (notably camera-wearer actions prefixed with `#C`) and reports dataset-scale stats for train+val such as:

- `# nouns = 508`, `# verbs = 119`
- `# unique object instances = 348,267`
- `# ground truth bounding boxes = 1,975,137`

Source: <https://ego4d-data.org/docs/tutorials/FHO_Overview/>

### Baselines & reference code (Hands & Objects)

- Benchmark repo: <https://github.com/EGO4D/hands-and-objects>

Within it:

- Temporal localization + classification baselines (I3D-ResNet, BiLSTM, BMN, SlowFast+Perceiver):
  - <https://github.com/EGO4D/hands-and-objects/blob/main/state-change-localization-classification/README.md>
- Object-of-change detection baselines (CenterBox, CenterNet, DETR, FasterRCNN):
  - <https://github.com/EGO4D/hands-and-objects/blob/main/state-change-object-detection-baselines/README.md>

### Orion mapping (Hands & Objects)

**What Orion has:** robust object tracks + per-track embeddings + (optional) depth + per-frame graph.

**What’s missing:** explicit modeling of *state change events* (temporal segmentation + “object-of-change”).

Concrete Orion extensions that align well:

- Add a new “event layer” (Phase 3+) that detects candidate state-change moments by:
  - strong appearance change in a tracked object (embedding delta + color/shape shifts),
  - hand/tool proximity patterns (hands near object then move away),
  - object split/merge or sudden size change.
- Once candidate PNR is found, use Gemini/VLM to label “what changed” and to sanity-check false positives.

---

## 3) Forecasting (future intent)

### Task definition (Forecasting)

The Forecasting benchmark includes four tasks:

1) locomotion prediction (future ground-plane trajectories),
2) hand movement prediction,
3) short-term object interaction anticipation (future active objects + verb + time-to-contact),
4) long-term action anticipation (future action sequence).

Official page: <https://ego4d-data.org/docs/benchmarks/forecasting/>

### Annotation notes (why baselines look the way they do)

The official docs describe a narration-driven annotation pipeline that identifies object interactions, assigns verb+noun labels, and defines **contact** and **precondition** frames. Bounding boxes are annotated for active objects (and hands) at multiple offsets before the interaction; ground-truth ego-trajectories are obtained via SfM.

Sources:

- Forecasting benchmark page: <https://ego4d-data.org/docs/benchmarks/forecasting/>
- FHO overview: <https://ego4d-data.org/docs/tutorials/FHO_Overview/>

### Baselines & reference code (Forecasting)

- Benchmark repo: <https://github.com/EGO4D/forecasting>

Short-term anticipation documentation notes:

- baseline uses Faster R-CNN object detector + SlowFast for verb + time-to-contact
- “new baseline code” is referenced as: <https://github.com/fpv-iplab/stillfast>

Source: <https://github.com/EGO4D/forecasting/blob/main/SHORT_TERM_ANTICIPATION.md>

Long-term anticipation uses egocentric recognition backbones (SlowFast / MViT) + aggregation modules and heads.

Source: <https://github.com/EGO4D/forecasting/blob/main/LONG_TERM_ANTICIPATION.md>

### Orion mapping (Forecasting)

Orion’s likely path here is not “beat Ego4D forecasting baselines tomorrow”, but:

- Represent the world state compactly (tracks + relations + recent interactions).
- Train lightweight heads to predict:
  - next interacted object track(s),
  - interaction verb distribution,
  - time-to-contact as regression.

If training data is unavailable, a pragmatic interim is **VLM-assisted forecasting**:

- Prompt a VLM with the last N seconds + tracked objects and ask for a structured guess:
  - {object, verb, TTC}.
- Use this as a *teacher* to bootstrap a smaller student model on your own data.

---

## 4) AV Diarization

### Task definition (AV Diarization)

The AVD benchmark defines 4 tasks:

1) face localization + tracking,
2) active speaker detection (including the camera wearer as an “invisible” speaker),
3) diarization segments per speaker,
4) transcription (English in this version).

Official page: <https://ego4d-data.org/docs/benchmarks/av-diarization/>

### Dataset scale (as described in the baseline repo)

The AVD baseline repo notes (high level):

- overall **>750 hours** of conversational data in v1,
- ~**50 hours** annotated to support benchmark tasks,
- **572 clips** (each **5 minutes**), with train/val/test splits.

Source: <https://github.com/EGO4D/audio-visual>

### Baselines & reference code (AV Diarization)

- Repo: <https://github.com/EGO4D/audio-visual>

It includes separate components for tracking, active speaker detection, diarization, and ASR.

- Tracking README describes TrackEval-based MOT evaluation: <https://github.com/EGO4D/audio-visual/blob/main/tracking/README.md>
- Audio-visual diarization baseline orchestration: <https://github.com/EGO4D/audio-visual/blob/main/diarization/audio-visual/README.md>
- Transcription baseline uses an ESPnet model zoo ASR model (oracle segmentation in the README): <https://github.com/EGO4D/audio-visual/blob/main/transcription/README.md>

### Orion mapping (AV Diarization)

Today Orion is *vision-first*. To play in AVD we’d need an audio stack:

- VAD (voice activity detection)
- ASR (e.g., Whisper / local ASR depending on deployment constraints)
- Audio-visual active speaker detection (ASD)

Vision side:

- face detection + tracking can re-use Orion’s tracking framework (with a face detector and face embeddings).

Once AVD exists, it becomes the substrate for the Social benchmark.

---

## 5) Social Interactions (LAM / TTM)

### Task definition (Social)

- **LAM (Looking at me):** per visible tracked face, classify whether they are looking at the wearer.
- **TTM (Talking to me):** per visible tracked face, classify whether they are talking to the wearer (builds on speaker status labels).

Official page: <https://ego4d-data.org/docs/benchmarks/social/>

### Baseline repository and reported baseline performance

The `social-interactions` repo has separate branches for LAM and TTM:

- LAM branch: <https://github.com/EGO4D/social-interactions/tree/lam>
  - Reported validation: `mAP: 78.07%`, `ACC: 87.97%`
  - If initialized from pretrained Gaze360: `mAP: 79.90%`, `ACC: 91.78%`
  - Reported test: `mAP: 66.07%`, `ACC: 75.38%`
- TTM branch: <https://github.com/EGO4D/social-interactions/tree/ttm>
  - Reported validation: `mAP: 52.85%`, `ACC: 60.24%`
  - Reported test: `mAP: 53.88%`, `ACC: 54.33%`

### Orion mapping (Social)

This is a good “Phase 3+” benchmark for Orion once:

- faces are treated as first-class tracked entities,
- (optional) audio integration exists.

Implementation ideas:

- **LAM:** gaze estimation + head pose + “camera-centric attention” classifier.
- **TTM:** combine active speaker detection with a directedness classifier (is speech directed at wearer vs someone else). Likely needs both audio cues and gaze/pose cues.

---

## Practical testing while Ego4D access is blocked

Even without Ego4D ground truth, we can run *proxy* evaluations that still de-risk Orion’s architecture.

### A) Episodic-memory-style VQ2D on any long video

1) Run Orion to produce tracks + embeddings (`python -m orion.cli.run_showcase ...`).
2) Pick a query crop from a frame early in the video.
3) Use Orion’s embedding index to retrieve the last matching track segment.
4) Use Gemini to validate the returned track qualitatively (is it really the same instance?)
   - this is in spirit aligned with VQ2D’s “where/when did I last see this?” objective.

### B) NLQ-style retrieval via captions

1) Chunk a video into fixed windows (e.g., 4–8s).
2) Caption each chunk with a VLM; embed captions.
3) For a natural language query, retrieve top windows.
4) Optionally refine boundaries by scanning around the top window.

### C) Hands & Objects sanity checks on your own clips

Record short clips where a clear state change occurs (open/close, cut, pour, remove).

- Verify whether Orion can:
  - identify the object undergoing change,
  - detect the key moment,
  - produce a stable track,
  - produce a reasonable natural-language description.

### D) Social interaction mini-set

Record simple conversation videos.

- Track faces
- Estimate gaze
- Detect “talking” using VAD + mouth motion (cheap heuristic)

These tests won’t produce official leaderboard numbers, but they will flush out integration problems (tracking IDs, time sync, serialization, model latency) early.

---

## Suggested Orion roadmap to cover these benchmarks

1) **Make VQ2D-style retrieval first-class** (best ROI): query crop → retrieve track(s) → export “response track” JSON.
2) **Stabilize 3D displacement**: SLAM pose + depth → 3D point + displacement vectors (VQ3D-ish).
3) **Add event/state-change layer**: detect candidate state changes; attach to tracks; optionally verify with VLM.
4) **Add audio stack** (if AVD/Social is a priority): VAD + ASR + ASD + diarization.
5) **Forecasting heads**: once track histories + events exist, forecasting becomes much easier.

