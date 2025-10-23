HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation
Trong-Thuan Nguyen1† , Pha Nguyen1† , Jackson Cothren1 , Alper Yilmaz2 , Khoa Luu1
1
1

2

University of Arkansas

Ohio State University

{thuann, panguyen, jcothre, khoaluu}@uark.edu

2

yilmaz.15@osu.edu

arXiv:2411.18042v2 [cs.CV] 31 Mar 2025

uark-cviu.github.io/projects/HyperGLM

Figure 1. Our HyperGLM framework supports Video Scene Graph Generation, Anticipation, and Reasoning. HyperGLM constructs scene graphs from
observed video frames and predicts relationships in unseen frames by leveraging a unified hypergraph for temporal modeling and comprehensive understanding.

Abstract

1. Introduction

Multimodal LLMs have advanced vision-language tasks but
still struggle with understanding video scenes. To bridge this
gap, Video Scene Graph Generation (VidSGG) has emerged
to capture multi-object relationships across video frames.
However, prior methods rely on pairwise connections, limiting their ability to handle complex multi-object interactions
and reasoning. To this end, we propose the Multimodal
Large Language Models (LLMs) on a Scene HyperGraph
(HyperGLM), promoting reasoning about multi-way interactions and higher-order relationships. Our approach uniquely
integrates entity scene graphs, which capture spatial relationships between objects, with a procedural graph that models
their causal transitions, forming a unified HyperGraph. Significantly, HyperGLM enables reasoning by injecting this
unified HyperGraph into LLMs. Additionally, we introduce a
new Video Scene Graph Reasoning (VSGR) dataset featuring
1.9M frames from third-person, egocentric, and drone views
and support five tasks. Empirically, HyperGLM consistently
outperforms state-of-the-art methods, effectively modeling
and reasoning complex relationships in diverse scenes.

In recent years, Multimodal Large Language Models
(LLMs) [2, 10] have set new benchmarks, with visionlanguage models excelling in diverse multimodal tasks. However, fully understanding dynamic video scenes remains a
significant challenge for applications like autonomous driving, intelligent surveillance, human-object interaction, and
multimedia analysis. Towards this goal, Video Scene Graph
Generation (VidSGG) [18, 52] has emerged as a critical task
for capturing multi-object relationships across video frames.
In particular, VidSGG enables high-level tasks such as event
forecasting [36, 43, 45], video captioning [24, 38, 40], and
video question answering [20, 30, 32, 41] by constructing
detailed representations of entities and their interactions.
However, prior VidSGG methods and datasets have limited capacity for comprehensive video understanding. Traditional scene graph-based methods [4, 34, 35, 42] only model
pairwise object relationships within single frames, making
it challenging to capture higher-order relationships and temporal dependencies in real-world scenarios. Additionally,
existing benchmark datasets [18, 34, 35, 52] focus is primarily confined to Scene Graph Generation (SGG) and Scene

† Equal contribution.

1

Graph Anticipation (SGA) tasks, lacking annotations for
reasoning tasks such as Video Question Answering (VQA),
Video Captioning (VC), and Relation Reasoning (RR).
In this paper, we propose the Multimodal LLMs on a
Scene HyperGraph (HyperGLM) approach to promote reasoning about multi-way interactions and higher-order relationships through a unified HyperGraph and LLMs, as illustrated in Fig. 1. To achieve this goal, we uniquely incorporate
entity scene graphs, which capture spatial relationships between objects, with a procedural graph that models their
causal transitions across video frames, forming a unified
HyperGraph, as shown in Fig. 2. Significantly, our HyperGLM approach allows reasoning by injecting this unified
HyperGraph into LLMs. In addition, we introduce a novel
Video Scene Graph Reasoning (VSGR) dataset, comprising 1.9 million video frames surpassing existing benchmark
datasets [18, 34, 35, 47, 52] in scale and annotation depth.
Specifically, our VSGR dataset includes videos from thirdperson, egocentric, and drone perspectives. It supports five
tasks: Scene Graph Generation, Scene Graph Anticipation,
Video Question Answering, Video Captioning, and Relation
Reasoning. Notably, our VSGR dataset introduces a new
Relation Reasoning task, setting it apart from existing video
scene graph datasets, as shown in the comparison in Table 1.
Contributions of this Work. This work presents three contributions to advance Video Scene Graph Generation. First,
we introduce Multimodal LLMs on a Scene HyperGraph,
leveraging hyperedges and LLMs for reasoning about multiway interactions and higher-order relationships. Second, we
develop a new Video Scene Graph Reasoning dataset, surpassing existing scale and annotation depth benchmarks. Our
VSGR dataset primarily supports five tasks within diverse
video scenes. Finally, the proposed HyperGLM consistently
outperforms state-of-the-art methods across all five tasks.

over, spatial-temporal methods [4, 34–36, 42] effectively
capture dynamic object relationships in videos. Recently,
Large Language Models [23] have been utilized to enhance
triplet extraction and alignment in weakly supervised SGG.

2.2. HyperGraphs
HyperGraphs have been adopted in computer vision to model
complex multimodal data and capture higher-order relationships. Unlike traditional graph-based approaches, HyperGraphs connect multiple nodes through hyperedges, enabling
multi-way relationships. They enhance Graph Neural Networks (GNNs) [12, 54] by allowing the modeling of more
sophisticated interactions. Recent advancements, such as HyperGraph Convolution [1] and HyperGraph Attention [22],
have further improved GNNs by capturing relationships beyond simple pairwise connections. Therefore, HyperGraphbased models effectively handle temporal dependencies and
complex interactions, significantly boosting performance in
tasks like accident anticipation [43], group activity recognition [28, 55], and video question answering [44, 46, 50].

2.3. Discussion
Limitations in Prior Methods. The methods introduced in
Section 2.1, based on Progressive Feature Fusion [39, 51],
Batch-Progressive Transformer Encoding [6, 13, 17, 25,
26], Spatial-Temporal Context Integration [4, 34, 42], and
Memory-Guided Temporal Consistency [7, 33, 35], have advanced VidSGG. However, these methods struggle to model
higher-order relationships and complex temporal dynamics. Specifically, Progressive Feature Fusion and BatchProgressive Transformer Encoding are limited in capturing
long-term temporal dependencies, with the former lacking
long-term context due to frame-by-frame processing and the
latter only addressing short-term dependencies. Similarly,
Spatial-Temporal Context Integration and Memory-Guided
Temporal Consistency inadequately represent multi-object interactions across video frames and insufficiently capture the
temporal evolution required for higher-order relationships.
Advantages of Our Approach. Our HyperGLM approach,
Multimodal LLMs on a Scene HyperGraph, promotes reasoning about multi-way interactions and high-order relationships. As illustrated in Fig. 2, HyperGLM enhances
the model’s ability to interpret complex relationships and
anticipate intricate video dynamics. Towards this goal, we
uniquely integrate entity scene graphs, which is introduced
in Sec. 3 to capture spatial interactions between objects
in each frame and a procedural graph, which is presented
in Sec. 4.1 to model their causal evolution. In our unified approach, hyperedges connect multiple nodes to capture higherorder relationships distinguished from traditional pairwise
methods [4, 34, 35, 42]. In addition, the procedural graph
enables multi-step transitions for anticipating future interactions or relationships, and reduces bias by generalizing

2. Related Work
In this section, we review advances in scene graph generation
and hypergraph applications in computer vision, then discuss
existing limitations and the advantages of our approach.

2.1. Scene Graph Generation
Scene Graph Generation has significantly advanced with
transformer-based models [6, 13, 17, 25, 26, 51] that have
become benchmarks due to their efficiency and state-ofthe-art performance. Recent work [3, 21, 33] focuses on
reducing bias and enhancing mean recall for rare predicates by integrating external knowledge and applying unbiased contextual augmentation, particularly in dynamic video
contexts. Open-vocabulary methods [15, 27] supported by
vision-language models handle unseen object and relationship classes, improving generalization. Additionally, generative models (e.g., diffusion-based methods [9, 53]) leverage
scene graphs for efficient image and scene synthesis. More2

Figure 2. (a) To model the temporal transition, a simple approach can be
using two scene graphs Gt and Gt+1 . (b) Another procedure graph can
present this temporal modeling. (c) Our unified HyperGraph in Fig. 2c
integrates both entity scene graph to capture spatial relationships and the
procedural graph to model the temporal evolution. HyperEdge represents
person sitting on couch, holding, then playing guitar, whereas holding →
playing describes a chain of interactions. HyperGraph is presented in 3D.

Figure 3. Our HyperGLM framework comprises an image encoder, MLP
projector, temporal aggregator, unified HyperGraph, and language model.
It processes video frames by encoding each frame with the image encoder
and MLP, extracting spatio-temporal features through image patch grids to
generate N spatial tokens per frame. The temporal aggregator compresses
the T × N embeddings over time. The MLP projector then transforms
these visual embeddings into the language feature space as frame tokens,
interleaved with language tokens, and fed into the Large Language Models.

infrequent relationship categories. Furthermore, our HyperGLM approach leverages fundamental mathematical properties: permutation equivariance ensures that the HyperGraph
structure remains consistent under any permutation of node
labels, and invariance to hyperedge order preserves semantic
meanings regardless of the node visit order during random
walks which is defined in Alg. 1. The theoretical foundations
and mathematical properties are detailed in the Appendices.

4. Our Proposed HyperGLM Approach
In this section, we present our approach, which incorporates
a unified HyperGraph into the LLMs, as illustrated in Fig. 3.

4.1. Video Scene (Hyper)Graphs
Traditional VidSGG methods [4, 34, 42] construct scene
graphs Gt to represent scene entities and their relationships
as in Fig. 2a. However, these graph-based approaches do not
capture higher-order relationships with temporal dependencies for video understanding [36]. To model this property,
we propose a novel HyperGraph-based framework that constructs a unified HyperGraph H = (VH , EH ), representing
spatial relationships within individual frames and temporal
transitions across frames, illustrated in Fig. 4. HyperGraph
extends traditional graph structures by allowing hyperedges
to connect more than two nodes, making them particularly
effective for modeling these higher-order relationships. To
leverage this property, our unified HyperGraph (see Fig. 2c)
integrates entity scene graphs Gt for each frame, capturing
spatial interactions as introduced in Sec. 3, along with a
procedural graph P that models their causal transitions, as
detailed in Sec. 4.1. We unify these components using a
random-walk HyperGraph construction, which is defined
in Alg. 1 to capture higher-order connectivity patterns and
structural semantics, thereby approximating subgraph matching between the entity scene graphs and the procedural graph.
This integration allows our approach to model current subjects’ interactions and anticipate their future relationships.
Procedural Graph Construction. SGA objective is to predict the set of relationships EG in the next frame based on
the current set of relationships. To model the temporal evolution of causal relationships between objects across video
frames, we introduce a procedural graph P = (VP , EP ),
serving as the temporal counterpart to the entity scene graphs
Gt as shown in Fig. 2b. In particular, the procedural graph P
can vary across datasets, modeling relationship transitions.
Toward this goal, we model and denote the set of vertices
in P as VP = {rtij ∈ VG | 1 ≤ t ≤ T }̸= , representing
distinct relationship categories. The set of edges in P, i.e.,

3. Problem Formulation
In this section, we define two tasks, including Scene Graph
Generation (SGG) and Scene Graph Anticipation (SGA).
Graph vertex sequence is represented as {VG | 1 ≤ t ≤ T },
where each set of vertex VG contains object features, bounding boxes, and categories. The scene graph for each frame t,
denoted by Gt = VG , EG = {rij | 1 ≤ i < j ≤ |VG |} ,
consists of all pairwise relationships between objects, with
rij representing the relationship category between vi and vj .
We aim to develop a process pθ : (VG × VG ) → rij to
predict the relationship rij between each object pair (vti , vtj )
in VG . We define task-specific queries QSGG , and QSGA to
direct our unified model to perform on each specific task. The
objective for each task is to minimize the negative likelihood
of the predicted scene graph Gt to the truth predicate set
Yt = {y ij | 1 ≤ i < j ≤ |VG |} on categories indexed by k.
Scene Graph Generation (SGG) generates the scene graph
for each frame t from t = 1 to t = T , which is defined as:
 \label {eq:sgg_formulation} \small \min _\theta \mathbb {E}_{\mathbf {G}_t, Y_t} \Big [ - \sum _{(v_t^{i, j})} \sum _{k} \big (y_k^{ij} \log p_\theta (r^{ij} \mid \mathbf {Q}_{\text {SGG}})_k\big ) \Big ] 

(1)

Scene Graph Anticipation (SGA) anticipates the scene
graphs for future frames, generating predictions Gt+n ,
where n denotes the anticipation horizon, formulated as:

 \label {eq:sga_formulation} \small \min _\theta \mathbb {E}_{\keyword {\mathbf {G}_{\leq t}}, Y_t} \Big [ - \sum _{(v_{t\keyword {+n}}^{i, j})} \sum _{k} \big (y_k^{ij} \log p_\theta (r^{ij} \mid \mathbf {Q}_{\text {SGA}})_k\big ) \Big ] 

(2)

In Eqns. (1) and (2), t indexes the current frame. While SGG
predicts the scene graph at frame t, SGA reads scene graphs
up to frame t to forecast the graph in future frame t + n.
3

Figure 4. Our Video Scene HyperGraph, including entity graphs and a procedural graph, as defined in Sec. 4.1. Blue nodes represent entities, while
green nodes denote relationships. The entity graph captures spatial relationships (subject ⊸ relationship ⊸ object), whereas the procedural graph models
relationship transitions (→). Hyperedges are visualized as polygons, encapsulating interactions through chains of relationships. For instance, a hyperedge
illustrates a person picking up, holding, opening, and reading a book while sitting on a couch. HyperGraph is presented in 3D, see Supplementary video.

EP = {(rm , rn )} represent possible causal transitions between these relationships, where an edge (rm , rn ) indicates
that relationship rm causally lead to relationship rn . We
quantify causal transitions by calculating transition probabilities w(rm , rn ) via their observed frequencies as in Eqn. (3).

  w(r_m, r_n) = \frac {\sum \limits _{t=1}^{T - 1} \sum \limits _{r^{ij}_{t, t+1}} \mathbbm {1}\left ( r_t^{ij} = r_m \land r_{t+1}^{ij} = r_n \right )}{ \sum \limits _{t=1}^{T - 1} \sum \limits _{r^{ij}_{t, t+1}} \mathbbm {1}\left ( r_t^{ij} = r_m \right )} \label {eq:transition_weights} 

representation combining the entity scene graphs with the
procedural graph. This unified structure merges spatial and
temporal relationships into a single graph, enabling the use of
conventional graph algorithms while preserving the complex
interactions captured by the HyperGraph. Mathematically, a
unified HyperGraph H = (VH , EH ) is defined as in Eqn. (6).

(3)

  \mathcal {H} = \left ( \bigcup _{t=1}^{T} V_{\mathbf {G}_t} \cup V_{\mathbf {P}}, \quad \bigcup _{t=1}^{T} E_{\mathbf {G}_t} \cup E_{\mathbf {P}} \right ) \label {eq:unified_HyperGraph} 

(6)

where VH includes all entity nodes vti from each Gt and the
relationship type nodes VP . The hyperedge set EH includes
pairwise relationships EGt within each Gt , capturing spatial relationships and temporal transition edges EP from P,
modeling the evolution of causal relationships across frames.

where 1(·) is the indicator function that counts transitions
from relationship rm at current frame t to rn at next frame
t + 1. Next, self-loops w(rm , rm ) are removed from the
graph, and these probabilities are normalized as in Eqn. (4).
  \sum _{r_n \in E_{\mathbf {P}}} w(r_m, r_n) = 1 \quad \text {for all } r_m \in E_{\mathbf {P}} \label {eq:transition_norm} 

Random-walk Algorithm. We employ the random walks
algorithm outlined in Alg. 1 to sample representative substructures from the unified HyperGraph H, which captures
connectivity patterns and mitigates the NP-hardness of exact
subgraph matching. Specifically, these walks alternate between nodes and hyperedges, preserving the multi-node connections intrinsic to hyperedges and capturing the complexity of multi-object relationships and their transition across
video frames. In each walk, a hyperedge hi aggregates the
visited nodes, thereby encapsulating higher-order relationships. For example, in the entity scene graph Gt , a walk
might traverse from a “person” to a “cup” via the “holding”,
resulting in the hyperedge hi = {person, cup}. Similarly, within the procedural graph P, a walk might transition through a sequence of interactions such as “holding”,
“placing”, and “releasing”, forming the hyperedge hi =
{holding, placing, releasing}. Therefore, we generate sampled hyperedges Esampled = {hi | i = 1, . . . , Nw }
as in L25 of Alg. 1 by conducting multiple random walks.

(4)
By leveraging these probabilities, the procedural graph
P enhances the prediction of future relationships. For each
relationship rtij in frame t, the most probable relationship
ij
rt+1
in the next frame t + 1 is determined as in Eqn. (5).
  r_{t+1}^{ij} = \arg \max _{r_n} P(r_n \mid r_t^{ij}, v^{i, j}) \label {eq:transition_probs} 

(5)

where P (rn | rtij , v i,j ) = w(rtij , rn ) × v i,j is the probability of transitioning from relationship rtij to rn , looking at
object features. This allows the model to anticipate future
interactions based on established temporal patterns.
HyperGraph Construction. To incorporate spatial relationships between objects in each frame and their causal
temporal transitions, we construct a unified HyperGraph H
that integrates the entity scene graphs Gt and the procedural graph P. Specifically, a HyperGraph is an augmented
4

Table 1. Comparisons of video scene graph datasets. SGG, SGA, VQA,
VC, and RR represent Scene Graph Generation, Scene Graph Anticipation,
Video Question Answering, Video Captioning, and Relation Reasoning.

Algorithm 1 Random-walk for HyperGraph Construction.
Require: H = (VH , EH ), Number of Walks Nw , Walk
Length Nl
′
Ensure: H′ = (VH , EH
)
1: Initialize Esampled ← ∅
2: for i = 1 to Nw do
3:
Select vstart ∈ VH uniformly at random
4:
Initialize walk sequence S ← [vstart ]
5:
for j = 1 to Nl do
\triangleright  Node to HyperEdge
6:
if j is odd then
7:
vcurrent ← S[j]
8:
Evcurrent ← {h ∈ EH | vcurrent ∈ h}
9:
if Evcurrent ̸= ∅ then
10:
Select hj ∈ Evcurrent
11:
Append hj to S
12:
end if
\triangleright  HyperEdge to Node
13:
else
14:
hcurrent ← S[j]
15:
Vhcurrent ← hcurrent
16:
Select vj ∈ Vhcurrent
17:
Append vj to S
18:
end if
19:
end for
20:
hi ← {v ∈ S | v ∈ VH } \triangleright  Form new HyperEdge
21:
if hi ∈
/ EH ∪ Esampled then
22:
Esampled ← Esampled ∪ {hi }
23:
end if
24: end for
′
← EH ∪ Esampled
25: EH
′
)
26: return H′ = (VH , EH

#Frames

SportsHHI [47]
Action Genome [18]
AeroEye [35]
ASPIRe [34]
PVSG [52]
VSGR (Ours)

11.4K
234.3K
261.5K
1.6M
153K
1.9M

SGG

SGA

Tasks
VQA

VC

RR

Bbox

✓
✓
✓
✓
✓
✓

✗
✓
✗
✗
✗
✓

✗
✗
✗
✗
✓
✓

✗
✗
✗
✗
✓
✓

✗
✗
✗
✗
✗
✓

✓
✓
✓
✓
✓
✓

Annotations
Relation Text
✓
✓
✓
✓
✓
✓

✗
✗
✗
✗
✓
✓

subsequent interactions by identifying new relationships and
adapting to evolving contexts using transition probabilities
ij
in P, anticipating future relationships rt+n
and improving
conversational flow. Finally, the hypothesis allows the model
to propose and test conjectures about the underlying themes
or intentions, synthesizing information from previous steps
and leveraging H to explain patterns and forecast results.

5. Video Scene Graph Reasoning Dataset
Limitations of Current Datasets. Existing benchmarks
(see Table 1) primarily support SGG and SGA, limiting their
applicability to reasoning tasks. They suffer from inadequate
support for VQA, VC, and RR, shallow annotations that fail
to capture intricate object interactions, insufficient modeling of temporal dynamics, and poor multimodal integration.
Thus, our VSGR dataset supports SGG, SGA, VQA, VC,
and RR, enabling the reasoning capabilities of LLMs.

5.1. Dataset Construction
Data Acquisition Stage. We source videos from the ASPIRe [34] and AeroEye [35] datasets. While the ASPIRe
dataset offers diverse, richly annotated videos emphasizing
dynamic interactions and temporal changes, the AeroEye
dataset offers drone-captured footage across various scenes.
Comprehension Tasks via Question-Answering. We introduce tasks that leverage fine-grained relationships from
scene graphs, extending Scene Graph Generation to focus
on relation understanding and subject/object interpretation
using <subject, relation, object> triplets and
leverage GPT-4/GPT-3.5 model for language generation.
Video Captioning (VC). We generate 82,532 videocaption pairs, resulting in about 22 captions per video. Our
process involves (1) extracting triplets from cropped video
frames focusing on foreground objects, (2) generating background descriptions based on these triplets, and (3) combining the foreground triplets and background descriptions
to produce captions. The average length is 893 characters,
surpassing the PVSG [52] dataset in quantity and detail.
Video Question Answering (VQA). We develop 74,856
question-answer pairs by designing questions that explore
diverse relationships, selecting subject and object categories
to verify specific relationships, or choosing triplets to assess
their uniqueness. This results in an average of approximately
20 questions per video, which exceeds existing benchmarks

4.2. Multimodal LLMs on HyperGraphs
Formulation. Given an input video V and a task query Q,
we aim to generate a target answer sequence A of length L.
Especially, by modeling a HyperGraph H, the target answer
A is generated via a process illustrated in Fig. 3, defined as:
  p(\mathbf {A} | \keywordtri {\mathbf {V}}, \mathbf {Q}, \keywordtwo {\mathcal {H}}) = \prod _{i=0}^{L-1} p(x_i | \keywordtri {\mathbf {V}}, \mathbf {Q}, \keywordtwo {\mathcal {H}}, x_{<i})~\label {eq:MLLM_with_G} 

Dataset

(7)

where x<i denotes the preceding token sequence, and A =
[x0 , . . . , xi , . . . , xL−1 ] is the sequence of answer tokens reasoning video V by question Q and HyperGraph H in Fig. 4.
As illustrated in Fig. 5, our process begins with the generation, constructing an entity scene graph Gt = (Vte , EG )
for each frame t to capture detected objects and their relationships. Relationship anticipation employs the procedural
graph P to model temporal evolution and predict future interactions. The model then engages in reasoning using the
HyperGraph H and verification through video captioning to
ensure contextual relevance and accuracy. It refines understanding through clarification, generating answers by reasoning over the video and H. In scene forecasting, it predicts
5

Figure 5. An example of the diversified context within the streaming dialog in our VSGR dataset. Best viewed in color and zooming in.

5.2. Dataset Comparison

such as the MSRVTT-QA [49] and MSVD-QA [48] datasets.
Relation Reasoning (RR). Using the annotated scene
graphs, we produce 61,120 relation reasoning tasks by selecting partial information as an incomplete input. Each task
requires the model to deduce relationships among entities
and identify the categories of the subject and object. With
an average of approximately 16 tasks per video, our VSGR
dataset provides a substantial collection for evaluating models’ abilities in relational reasoning and scene understanding.
Question and Answer Validation. To ensure the quality and
complexity of questions, we implement a rigorous validation
process that combines generation by LLMs with human refinement. Initially, GPT-4/GPT-3.5 generates queries based
on the <subject, relation, object> triplets extracted from the videos. Human annotators are trained with
specific guidelines that emphasize clarity, relevance, and appropriate challenge levels, then review and refine questions.
They enhance the questions by ensuring they are directly
answerable from the video content and require careful reasoning about the depicted interactions and relationships.
We apply strict filtering criteria to improve the quality
of the dataset. First, we eliminate questions that do not require video context and can be answered using general world
knowledge, ensuring that models must rely on visual information from the videos. Second, we exclude questions that
LLMs can answer correctly, increasing the challenge and
utility of the dataset in evaluating advanced reasoning abilities. Finally, independent annotators perform a second round
to review and evaluate the quality of the refined questions.

As reported in Table 1, our VSGR dataset represents a substantial advancement in video scene graph benchmarks. Our
dataset comprises 3,748 videos and 1,841,243 frames, surpassing existing datasets in scale. Unlike other datasets that
address only a limited subset of tasks, our dataset offers
comprehensive task coverage, facilitating multifaceted evaluations of LLMs. In addition, our ground truth enriches relation annotations with comprehensive textual descriptions, enabling sophisticated reasoning and relationship predictions,
as illustrated in Fig. 5. Additionally, our VSGR dataset
incorporates diverse viewpoints, including third-person, egocentric, and drone perspectives, enhancing its generalization.

6. Experiment Results
6.1. Implementation Details
Datasets. We leverage our VSGR dataset across five tasks.
In addition, we utilize the PVSG [52] dataset for the SGG
task and the Action Genome [18] dataset for the SGA task.
Model Configuration. We operate the CLIP-ViT-L-336 [8,
37] to encode each video frame into ten tokens (one CLS token and nine from 3×3 average pooling). These tokens
are fed into a two-layer MLP connector to the Mistral7B-Instruct [19] language model. For training, we apply
LoRA [16] to all linear layers with a rank of 128 and a scaling factor of 256, omitting vision-language alignment [31].
We train for two epochs with a batch size of 128 over 16
6

Table 3. Comparison (%) on the VSGR and PVSG datasets for the Scene
Graph Generation (SGG) task at Recall (R) and mean Recall (mR).
PVSG

Method
Transformer [52]
HIG [34]
CYCLO [35]
HyperGraph (Ours)
HyperGLM (Ours)

Figure 6. Comparison of Recall (R) and mean Recall (mR) at different
numbers of hyperedges for the SGG task on the VSGR dataset.

F

Method

R/mR@20

R/mR@50

R/mR@10

R/mR@20

R/mR@50

0.3

STTran+ [5]
DSGDetr+ [11]
STTran++ [5]
DSGDetr++ [11]
SceneSayerODE [36]
SceneSayerSDE [36]
HyperGraph (Ours)
HyperGLM (Ours)

13.9 / 3.5
14.3 / 3.6
15.4 / 6.2
16.8 / 8.4
23.3 / 13.3
25.9 / 15.6
26.5 / 14.8
27.5 / 15.8

21.6 / 7.3
21.8 / 7.6
27.2 / 14.1
29.0 / 16.7
32.5 / 20.1
35.7 / 23.1
36.2 / 22.1
37.0 / 24.5

40.8 / 20.3
41.3 / 21.2
48.6 / 32.2
48.9 / 32.3
45.1 / 33.0
47.4 / 37.1
49.3 / 37.2
50.0 / 38.0

12.0 / 4.0
12.5 / 4.2
14.0 / 6.5
14.5 / 7.0
18.0 / 9.5
19.5 / 11.0
18.8 / 9.3
19.0 / 10.0

19.0 / 8.0
19.5 / 8.3
22.0 / 11.0
22.5 / 12.0
26.0 / 15.0
27.5 / 17.0
27.2 / 16.3
28.0 / 16.5

35.7 / 18.0
36.0 / 19.0
39.0 / 25.1
40.0 / 26.0
42.0 / 30.0
44.0 / 33.5
42.5 / 27.4
43.0 / 27.5

0.5

STTran+ [5]
DSGDetr+ [11]
STTran++ [5]
DSGDetr++ [11]
SceneSayerODE [36]
SceneSayerSDE [36]
HyperGraph (Ours)
HyperGLM (Ours)

14.9 / 3.7
15.2 / 3.9
16.6 / 6.6
17.4 / 8.4
26.4 / 14.3
28.4 / 16.3
29.2 / 16.4
30.0 / 17.0

22.6 / 7.6
23.1 / 8.0
29.1 / 14.7
30.5 / 17.0
36.6 / 21.4
38.6 / 25.1
39.3 / 23.2
40.5 / 27.5

42.9 / 21.4
43.3 / 22.2
51.5 / 33.4
51.9 / 33.9
49.8 / 36.0
51.4 / 39.9
52.1 / 38.7
53.5 / 40.5

13.5 / 4.2
13.8 / 4.5
15.5 / 7.0
16.0 / 7.5
20.5 / 11.0
21.5 / 12.5
20.3 / 12.2
21.5 / 11.5

21.0 / 8.5
21.5 / 9.0
23.5 / 12.5
24.0 / 13.0
29.5 / 16.5
31.0 / 18.5
29.8 / 17.1
31.5 / 18.5

37.0 / 19.0
38.0 / 20.0
41.0 / 26.5
42.0 / 28.0
46.1 / 32.5
48.0 / 35.7
46.2 / 33.1
46.5 / 30.0

0.7

STTran+ [5]
DSGDetr+ [11]
STTran++ [5]
DSGDetr++ [11]
SceneSayerODE [36]
SceneSayerSDE [36]
HyperGraph (Ours)
HyperGLM (Ours)

16.6 / 4.2
16.8 / 4.3
19.0 / 7.7
19.8 / 9.5
32.1 / 16.5
33.3 / 18.1
34.3 / 19.2
35.7 / 19.5

25.1 / 8.5
25.3 / 8.8
32.8 / 17.1
34.1 / 19.2
42.8 / 24.4
44.0 / 27.3
45.2 / 25.3
46.1 / 30.0

47.2 / 24.0
47.4 / 24.7
56.8 / 36.8
56.7 / 37.2
55.6 / 39.6
56.4 / 44.4
57.2 / 42.1
58.2 / 44.0

15.0 / 5.0
15.5 / 5.2
17.0 / 8.0
17.5 / 8.5
23.5 / 13.0
24.5 / 14.5
22.3 / 13.2
25.1 / 13.5

24.0 / 10.5
24.5 / 11.0
27.0 / 14.0
28.0 / 15.0
33.5 / 19.0
35.7 / 21.0
34.2 / 19.3
35.5 / 21.5

41.0 / 21.5
42.0 / 22.0
45.0 / 29.0
46.1 / 30.0
51.0 / 36.0
53.0 / 38.0
50.4 / 35.3
51.0 / 33.5

R/mR@100

R/mR@20

R/mR@50

R/mR@100

4.0 / 1.8
4.6 / 1.9
5.8 / 2.0
6.5 / 2.2
7.5 / 2.8

4.4 / 1.9
4.9 / 2.1
6.1 / 2.2
7.0 / 2.4
8.1 / 3.7

4.9 / 2.0
5.4 / 2.2
6.7 / 2.3
7.5 / 2.6
8.5 / 3.9

25.7 / 6.3
23.8 / 5.7
29.4 / 7.1
31.6 / 7.8
35.8 / 9.2

34.5 / 6.5
31.1 / 5.9
36.4 / 7.7
38.8 / 8.3
42.3 / 10.1

43.5 / 7.0
40.4 / 6.9
47.7 / 7.7
50.3 / 8.5
54.7 / 10.4

Action Genome

Method
STTran+ [5]
DSGDetr+ [11]
STTran++ [5]
DSGDetr++ [11]
SceneSayerODE [36]
SceneSayerSDE [36]
HyperGraph (Ours)
HyperGLM (Ours)

VSGR

R/mR@10

R/mR@50

Table 4. Comparison (%) on the VSGR and Action Genome datasets for the
Scene Graph Anticipation (SGA) task at Recall (R) and mean Recall (mR).

Table 2. Comparison (%) on the VSGR and Action Genome datasets for the
Scene Graph Anticipation (SGA) task at varying video input fractions F .
Action Genome

VSGR

R/mR@20

VSGR

R/mR@10

R/mR@20

R/mR@50

R/mR@10

R/mR@20

R/mR@50

17.5 / 4.6
17.9 / 4.7
20.2 / 8.9
22.2 / 11.4
36.6 / 17.8
37.3 / 20.8
37.5 / 19.1
38.8 / 22.3

26.8 / 9.2
27.7 / 9.7
35.7 / 18.4
37.1 / 21.0
48.3 / 27.4
48.6 / 30.9
49.3 / 31.4
51.5 / 33.0

49.6 / 24.3
51.4 / 25.9
60.2 / 38.8
61.0 / 39.5
61.3 / 43.4
61.6 / 46.8
62.3 / 47.4
65.2 / 48.6

16.5 / 5.5
17.0 / 6.0
19.0 / 9.5
19.5 / 10.0
26.5 / 14.0
27.5 / 16.0
28.4 / 17.5
30.2 / 18.1

26.0 / 11.0
27.0 / 11.5
30.0 / 15.5
31.0 / 16.0
37.5 / 20.0
38.5 / 22.0
39.3 / 22.4
41.1 / 23.5

43.0 / 23.0
44.5 / 24.5
49.0 / 31.0
50.0 / 32.5
55.5 / 38.0
58.2 / 40.0
57.6 / 41.5
59.3 / 43.4

fraction, F, to 0.3, 0.5, and 0.7 for the SGA task. This adjustment allows the model to learn from varying observed
portions and predict the unseen segment. Table 2 indicates
that increasing the portion of the seen video improves performance, suggesting that additional visual context is beneficial.
In addition, Table 4 further confirms that the default setting
at a higher input fraction (F = 0.9) leads to optimal performance. We also present additional settings with varying
video input fractions for the SGA task in the Appendices.

6.3. Comparison with State-of-the-Arts
iterations on 4 × GPUs, taking approximately six hours.
Metrics. We evaluate the SGG and SGA tasks using the
Recall and mean Recall scores. In addition, we evaluate
the VQA and RR tasks by Accuracy, Precision, Recall, and
F1 scores. For the VC task, we utilize CIDEr, MENTOR,
ROUGE-L, and BLEU-4 scores to validate our performance.
Settings. For the SGG and SGA tasks, we adopt the evaluation settings based on [18, 36]. In particular, the model
is provided with raw video frames and must detect objects
using a pre-trained detector (i.e. Faster R-CNN) and predict
or anticipate their relationships. Especially for the SGA, we
set the initial video input fraction (F) to 0.9, following [36].

Table 3 demonstrates that HyperGLM significantly outperforms existing methods on the PVSG and VSGR datasets,
achieving the highest R@20 scores of 7.5% and 35.8%,
respectively. By leveraging hyperedges connecting multiple nodes within the HyperGraph, HyperGLM effectively
captures complex object interactions and spatial dependencies, transcending traditional pairwise methods to generate more accurate and detailed scene graph representations.
Notably, integrating a procedural graph within the HyperGraph reduces bias. Our approach enhances mean Recall,
reaching improvements of 2.8% on the PVSG dataset and
9.2% on the VSGR dataset, thereby addressing the longtail distribution challenges that have struggled in previous
methods [34, 35, 52]. Furthermore, the LLM enhances the
capacity of HyperGLM to infer and predict intricate relationships embedded within the unified HyperGraph, resulting
in improved performance compared to HyperGraph, which
shows a decrease of 4.2% at R@20 on the VSGR dataset.
As shown in Table 4, our HyperGLM approach outperforms existing SGA methods on the Action Genome and
VSGR datasets, achieving R@10 scores of 35.7% and 25.1%,
respectively. This improvement stems from integrating procedural graphs that model causal relationships within the
HyperGraph structure. In contrast, the SceneSayer [36]
method relies on NeuralODE and NeuralSDE to capture the
latent dynamics of object interaction evolution. Our procedural graphs enable multi-step transitions that explicitly

6.2. Ablation Study
Hypergraph Parameters. The number of walks (Nw ) and
walk length (Nl ) are introduced in Alg. 1, directly impacting
the capacity to capture high-order relationships by determining the number of hyperedges. Although more hyperedges increase relational diversity, they can also introduce
redundancy beyond an optimal point. Specifically, a higher
Nw broadens the range of sampled relationships, while a
moderate Nl balances depth. Our experiments demonstrate
that optimal performances are achieved with 60 hyperedges
(Nw = 60 and Nl = 7), shown in Fig. 6. Further experiments on the SGG task are provided in Table 3, and experiments on these parameters are included in the Appendices.
Video Input Fraction. We adjust the initial video input
7

Figure 7. Qualitative comparison of our HyperGLM approach versus SceneSayerSDE [36] for SGG and SGA. Red and green edge labels denote incorrect and
correct predictions, respectively. Our HyperGLM approach effectively captures the evolving interactions between person-1 and bicycle-1 or person-2 and
person-3 over time and anticipates interactions in unseen video frames, while SceneSayerSDE confuses to similar predicates. Best viewed in color.
Table 5. Comparison (%) on VSGR for the Video Question Answering.
Table 7. Comparison (%) on VSGR for the Relation Reasoning.

Method

Accuracy

Precision

Recall

F1 Score

Method

Accuracy

Precision

Recall

F1 Score

Video-ChatGPT [32]
Video-LLaVA-7B [30]
MovieChat [41]
Chat-UniVi-7B [20]
HyperGLM (Ours)

33.2
43.1
43.5
44.3
45.4

35.1
43.8
44.2
45.6
47.2

32.3
41.7
42.6
43.2
44.3

33.6
42.7
43.4
44.4
45.7

Video-LLaVA-7B [30]
MA-LMM [14]
LLaMA-VID-7B [29]
HyperGLM (Ours)

41.3
42.8
44.1
47.2

42.5
43.7
45.2
48.4

40.2
41.8
43.5
46.5

41.3
42.7
44.3
47.4

Table 6. Comparison (%) on VSGR for the Video Captioning.

Method

CIDEr

MENTOR

ROUGE-L

BLEU-4

MV-GPT [38]
CoCap [40]
UniVL + MELTR [24]
HyperGLM (Ours)

57.1
54.3
50.5
54.5

37.5
29.4
28.1
30.7

62.5
61.8
60.0
64.9

47.2
42.8
42.1
48.8

ently, achieving scores of 54.5% at CIDEr and 64.9% at
ROUGE-L. In RR task, the HyperGraph effectively manages
intricate dependencies between multiple objects, resulting in
precise relational inferences with an accuracy of 47.2%.

7. Conclusion

capture the temporal evolution of relationships by modeling
sequential interactions and their dependencies. In contrast,
NeuralODE and NeuralSDE primarily focus on continuoustime dynamics, which may limit their effectiveness in handling discrete, multi-step relational changes as illustrated
in Fig. 7b. Significantly, the HyperGraph is injected into
the LLM, allowing the model to capture complex temporal
patterns, resulting in better results than the HyperGraph.
In addition to the improvements in SGG and SGA shown
in Fig. 7, Tables 5, 6, and 7 demonstrate the significant improvements using our HyperGLM approach in the VQA,
VC, and RR tasks. In VQA, the hyperedges connect multiple objects, enabling HyperGLM to capture context-rich
interactions more effectively, reaching an accuracy of 45.4%.
For VC, the HyperGraph structure supports evolving object
connections over time, allowing our HyperGLM approach
to generate captions that describe the relationship between
objects and capture the temporal flow of interactions coher-

In this paper, we have introduced HyperGLM, a novel
VidSGG method that integrates scene hypergraph information into LLMs for context-aware and precise scene interpretation. Our approach effectively models complex interactions and higher-order relationships. It outperforms leading
methods on benchmarks, including PVSG, Action Genome,
and our newly collected VSGR dataset across five tasks.
Limitations. Although our HyperGLM approach effectively
models multi-way interactions, managing many objects and
their interactions can complicate relationship structures. As
the HyperGraph expands, essential relationships may become obscured, reducing the clarity of scene interpretation.
Acknowledgment. This material is based upon work supported by the National Science Foundation under Award
No. OIA-1946391. We also acknowledge the Arkansas
High-Performance Computing Center for providing GPUs.
8

References

2022. 2
[13] Zeeshan Hayder and Xuming He. Dsgg: Dense relation
transformer for an end-to-end scene graph generation. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 28317–28326, 2024. 2
[14] Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam
Lim. Ma-lmm: Memory-augmented large multimodal model
for long-term video understanding. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 13504–13514, 2024. 8
[15] Tao He, Lianli Gao, Jingkuan Song, and Yuan-Fang Li. Towards open-vocabulary scene graph generation with promptbased finetuning. In European Conference on Computer
Vision, pages 56–73. Springer, 2022. 2
[16] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.
LoRA: Low-rank adaptation of large language models. In
International Conference on Learning Representations, 2022.
6
[17] Jinbae Im, JeongYeon Nam, Nokyung Park, Hyungmin Lee,
and Seunghyun Park. Egtr: Extracting graph from transformer
for scene graph generation. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 24229–24238, 2024. 2
[18] Jingwei Ji, Ranjay Krishna, Li Fei-Fei, and Juan Carlos
Niebles. Action genome: Actions as compositions of spatiotemporal scene graphs. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 10236–10247, 2020. 1, 2, 5, 6, 7
[19] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch,
Chris Bamford, Devendra Singh Chaplot, Diego de las Casas,
Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile
Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825,
2023. 6
[20] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao,
and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 13700–
13710, 2024. 1, 8
[21] Tianlei Jin, Fangtai Guo, Qiwei Meng, Shiqiang Zhu, Xiangming Xi, Wen Wang, Zonghao Mu, and Wei Song. Fast
contextual scene graph generation with unbiased context augmentation. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 6302–6311,
2023. 2
[22] Eun-Sol Kim, Woo Young Kang, Kyoung-Woon On, Yu-Jung
Heo, and Byoung-Tak Zhang. Hypergraph attention networks
for multimodal learning. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages
14581–14590, 2020. 2
[23] Kibum Kim, Kanghoon Yoon, Jaehyeong Jeon, Yeonjun
In, Jinyoung Moon, Donghyun Kim, and Chanyoung Park.
Llm4sgg: Large language models for weakly supervised
scene graph generation. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 28306–28316, 2024. 2
[24] Dohwan Ko, Joonmyung Choi, Hyeong Kyu Choi, Kyoung-

[1] Song Bai, Feihu Zhang, and Philip HS Torr. Hypergraph
convolution and hypergraph attention. Pattern Recognition,
110:107637, 2021. 2
[2] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin,
Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing
Mao, and Mike Zheng Shou. Videollm-online: Online video
large language model for streaming video. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 18407–18418, 2024. 1
[3] Zhanwen Chen, Saed Rezayi, and Sheng Li. More knowledge,
less bias: Unbiasing scene graph generation with explicit
ontological adjustment. In Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision, pages
4023–4032, 2023. 2
[4] Yuren Cong, Wentong Liao, Hanno Ackermann, Bodo Rosenhahn, and Michael Ying Yang. Spatial-temporal transformer
for dynamic scene graph generation. In Proceedings of
the IEEE/CVF international conference on computer vision,
pages 16372–16382, 2021. 1, 2, 3
[5] Yuren Cong, Wentong Liao, H. Ackermann, M. Yang, and
B. Rosenhahn. Spatial-temporal transformer for dynamic
scene graph generation. IEEE International Conference on
Computer Vision, 2021. 7
[6] Yuren Cong, Michael Ying Yang, and Bodo Rosenhahn.
Reltr: Relation transformer for scene graph generation. IEEE
Transactions on Pattern Analysis and Machine Intelligence,
45(9):11169–11183, 2023. 2
[7] Youming Deng, Yansheng Li, Yongjun Zhang, Xiang Xiang, Jian Wang, Jingdong Chen, and Jiayi Ma. Hierarchical
memory learning for fine-grained scene graph generation. In
European Conference on Computer Vision, pages 266–283.
Springer, 2022. 2
[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is
worth 16x16 words: Transformers for image recognition at
scale. In International Conference on Learning Representations, 2021. 6
[9] Azade Farshad, Yousef Yeganeh, Yu Chi, Chengzhi Shen,
Böjrn Ommer, and Nassir Navab. Scenegenie: Scene graph
guided diffusion models for image synthesis. In Proceedings of the IEEE/CVF International Conference on Computer
Vision, pages 88–98, 2023. 2
[10] Hao Fei, Shengqiong Wu, Wei Ji, Hanwang Zhang, Meishan
Zhang, Mong-Li Lee, and Wynne Hsu. Video-of-thought:
Step-by-step video reasoning from perception to cognition.
In Forty-first International Conference on Machine Learning,
2024. 1
[11] Shengyu Feng, Hesham Mostafa, Marcel Nassar, Somdeb
Majumdar, and Subarna Tripathi. Exploiting long-term dependencies for generating dynamic scene graphs. IEEE Workshop/Winter Conference on Applications of Computer Vision,
2021. 7
[12] Yue Gao, Yifan Feng, Shuyi Ji, and Rongrong Ji. Hgnn+:
General hypergraph neural networks. IEEE Transactions on
Pattern Analysis and Machine Intelligence, 45(3):3181–3199,

9

Woon On, Byungseok Roh, and Hyunwoo J Kim. Meltr: Meta
loss transformer for learning to fine-tune video foundation
models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20105–20115,
2023. 1, 8
[25] Sanjoy Kundu and Sathyanarayanan N Aakur. Is-ggt: Iterative scene graph generation with generative transformers.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 6292–6301, 2023. 2
[26] Rongjie Li, Songyang Zhang, and Xuming He. Sgtr: End-toend scene graph generation with transformer. In proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pages 19486–19496, 2022. 2
[27] Rongjie Li, Songyang Zhang, Dahua Lin, Kai Chen, and
Xuming He. From pixels to graphs: Open-vocabulary scene
graph generation with vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 28076–28086, 2024. 2
[28] Wanxin Li, Wei Xie, Zhigang Tu, Wei Wang, and Lianghao
Jin. Multi-hyperedge hypergraph for group activity recognition. In 2022 International Joint Conference on Neural
Networks (IJCNN), pages 01–07. IEEE, 2022. 2
[29] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In European
Conference on Computer Vision, pages 323–340. Springer,
2025. 8
[30] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng
Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint
arXiv:2311.10122, 2023. 1, 8
[31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
Visual instruction tuning. Advances in neural information
processing systems, 36, 2024. 6
[32] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and
Fahad Khan. Video-ChatGPT: Towards detailed video understanding via large vision and language models. In LunWei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages
12585–12602, Bangkok, Thailand, Aug. 2024. Association
for Computational Linguistics. 1, 8
[33] Sayak Nag, Kyle Min, Subarna Tripathi, and Amit K RoyChowdhury. Unbiased scene graph generation in videos. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22803–22813, 2023. 2
[34] Trong-Thuan Nguyen, Pha Nguyen, and Khoa Luu. Hig: Hierarchical interlacement graph approach to scene graph generation in video understanding. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
2024. 1, 2, 3, 5, 7
[35] Trong-Thuan Nguyen, Pha Nguyen, Li Xin, Cothren Jackson, Yilmaz Alper, and Khoa Luu. CYCLO: Cyclic graph
transformer approach to multi-object relationship modeling
in aerial videos. In The Thirty-eighth Annual Conference on
Neural Information Processing Systems, 2024. 1, 2, 5, 7
[36] Rohith Peddi, Saksham Singh, Parag Singla, Vibhav Gogate,
et al. Towards scene graph anticipation. In European Conference on Computer Vision. Springer, 2024. 1, 2, 3, 7, 8
[37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya

Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervision. In International conference on machine learning, pages
8748–8763. PMLR, 2021. 6
[38] Paul Hongsuck Seo, Arsha Nagrani, Anurag Arnab, and
Cordelia Schmid. End-to-end generative pretraining for multimodal video captioning. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 17959–17968, 2022. 1, 8
[39] Xindi Shang, Yicong Li, Junbin Xiao, Wei Ji, and Tat-Seng
Chua. Video visual relation detection via iterative inference.
In Proceedings of the 29th ACM international conference on
Multimedia, pages 3654–3663, 2021. 2
[40] Yaojie Shen, Xin Gu, Kai Xu, Heng Fan, Longyin Wen, and
Libo Zhang. Accurate and fast compressed video captioning.
In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 15558–15567, 2023. 1, 8
[41] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang,
Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian
Ye, Yanting Zhang, et al. Moviechat: From dense token to
sparse memory for long video understanding. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 18221–18232, 2024. 1, 8
[42] Yao Teng, Limin Wang, Zhifeng Li, and Gangshan Wu. Target
adaptive context aggregation for video scene graph generation.
In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 13688–13697, 2021. 1, 2, 3
[43] Nupur Thakur, PrasanthSai Gouripeddi, and Baoxin Li. Graph
(graph): A nested graph-based framework for early accident
anticipation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 7533–7541,
2024. 1, 2
[44] Aisha Urooj, Hilde Kuehne, Bo Wu, Kim Chheu, Walid Bousselham, Chuang Gan, Niels Lobo, and Mubarak Shah. Learning situation hyper-graphs for video question answering. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14879–14889, 2023. 2
[45] Jiahao Wang, Guo Chen, Yifei Huang, Limin Wang, and Tong
Lu. Memory-and-anticipation transformer for online action
understanding. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 13824–13835, 2023.
1
[46] Yanan Wang, Shuichiro Haruta, Donghuo Zeng, Julio Vizcarra, and Mori Kurokawa. Multi-object event graph representation learning for video question answering. In Meeting
on Image Recognition and Understanding, 2024. 2
[47] Tao Wu, Runyu He, Gangshan Wu, and Limin Wang.
Sportshhi: A dataset for human-human interaction detection
in sports videos. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024. 2, 5
[48] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang,
Xiangnan He, and Yueting Zhuang. Video question answering
via gradually refined attention over appearance and motion.
In Proceedings of the 25th ACM international conference on
Multimedia, pages 1645–1653, 2017. 6
[49] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large
video description dataset for bridging video and language. In
Proceedings of the IEEE conference on computer vision and

10

pattern recognition, pages 5288–5296, 2016. 6
[50] Zenan Xu, Wanjun Zhong, Qinliang Su, Zijing Ou, and
Fuwei Zhang. Modeling semantic composition with syntactic hypergraph for video question answering. arXiv preprint
arXiv:2205.06530, 2022. 2
[51] Jingkang Yang, Yi Zhe Ang, Zujin Guo, Kaiyang Zhou,
Wayne Zhang, and Ziwei Liu. Panoptic scene graph generation. In European Conference on Computer Vision, pages
178–196. Springer, 2022. 2
[52] Jingkang Yang, Wenxuan Peng, Xiangtai Li, Zujin Guo,
Liangyu Chen, Bo Li, Zheng Ma, Kaiyang Zhou, Wayne
Zhang, Chen Change Loy, et al. Panoptic video scene graph
generation. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 18675–
18685, 2023. 1, 2, 5, 6, 7
[53] Guangyao Zhai, Evin Pınar Örnek, Shun-Cheng Wu, Yan Di,
Federico Tombari, Nassir Navab, and Benjamin Busam. Commonscenes: Generating commonsense 3d indoor scenes with
scene graphs. Advances in Neural Information Processing
Systems, 36, 2024. 2
[54] Zizhao Zhang, Yifan Feng, Shihui Ying, and Yue Gao.
Deep hypergraph structure learning.
arXiv preprint
arXiv:2208.12547, 2022. 2
[55] Xiaolin Zhu, Dongli Wang, Jianxun Li, Rui Su, Qin Wan, and
Yan Zhou. Dynamical attention hypergraph convolutional
network for group activity recognition. IEEE Transactions
on Neural Networks and Learning Systems, 2024. 2

11

