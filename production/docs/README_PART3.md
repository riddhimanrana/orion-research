"""
Part 3: Query & Evaluation Engine - Documentation
==================================================

Complete documentation for the Query & Evaluation Engine, the third part
of the "From Moments to Memory" video understanding pipeline.

Author: Orion Research Team
Date: January 2025
"""

# Part 3: Query & Evaluation Engine

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup](#setup)
4. [Quick Start](#quick-start)
5. [Agent Comparison](#agent-comparison)
6. [Benchmarks](#benchmarks)
7. [Configuration](#configuration)
8. [API Reference](#api-reference)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Troubleshooting](#troubleshooting)
11. [Integration with Parts 1+2](#integration)

---

## Overview

Part 3 implements the **Query & Evaluation Engine**, which answers natural language
questions about video content using three different approaches:

- **Agent A (Gemini Baseline)**: Vision-only approach using Google Gemini's multimodal capabilities
- **Agent B (Heuristic Graph)**: Graph-only approach using hand-crafted Cypher queries on Neo4j
- **Agent C (Augmented SOTA)**: Combined approach leveraging both vision and graph with LLM reasoning

### Key Features

✓ **15 Question Types** (EC-15): WHAT, WHEN, WHERE, WHO, WHY, HOW, and complex reasoning  
✓ **LOT-Q Benchmark**: Long-form temporal questions for comprehensive evaluation  
✓ **Three-Agent Comparison**: Compare vision-only, graph-only, and augmented approaches  
✓ **Comprehensive Evaluation**: Accuracy, confidence, processing time, and semantic similarity  
✓ **Flexible Configuration**: Multiple presets for different use cases  
✓ **Integration Ready**: Works seamlessly with Parts 1+2 output  

---

## Architecture

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Part 3: Query & Evaluation               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Video + Question + Knowledge Graph (from Parts 1+2)     │
│                                                                  │
│  ┌────────────┐      ┌────────────┐      ┌────────────┐        │
│  │  Agent A   │      │  Agent B   │      │  Agent C   │        │
│  │  (Vision)  │      │  (Graph)   │      │ (Combined) │        │
│  └─────┬──────┘      └─────┬──────┘      └─────┬──────┘        │
│        │                   │                   │                │
│        ├───────────────────┴───────────────────┤                │
│        │                                       │                │
│        │           Evaluator                   │                │
│        │           • Accuracy                  │                │
│        │           • Confidence                │                │
│        │           • Processing Time           │                │
│        │           • Semantic Similarity       │                │
│        │                                       │                │
│        └───────────────────┬───────────────────┘                │
│                            │                                    │
│  Output: Comparative Analysis + Metrics + Insights             │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **Data Structures** (`part3_query_evaluation.py`)

```python
@dataclass
class Question:
    id: str
    question_text: str
    question_type: QuestionType  # 15 types
    video_path: str
    timestamp_start: float
    timestamp_end: float
    ground_truth: str
    metadata: Dict[str, Any]

@dataclass
class Answer:
    question_id: str
    agent_name: str
    answer_text: str
    confidence: float
    reasoning: str
    evidence: List[str]
    processing_time: float

@dataclass
class EvaluationResult:
    question_id: str
    agent_name: str
    correct: bool
    score: float
    metrics: Dict[str, float]
    explanation: str
```

#### 2. **Question Types** (EC-15)

```python
class QuestionType(Enum):
    # Basic information
    WHAT_ACTION = "what_action"      # What is happening?
    WHAT_OBJECT = "what_object"      # What objects are present?
    
    # Temporal
    WHEN_START = "when_start"        # When does it start?
    WHEN_END = "when_end"            # When does it end?
    
    # Spatial
    WHERE_LOCATION = "where_location"  # Where is it?
    WHERE_MOVEMENT = "where_movement"  # Where is it going?
    
    # Identity & counting
    WHO_IDENTITY = "who_identity"    # Who is it?
    WHO_COUNT = "who_count"          # How many?
    
    # Causality & intent
    WHY_REASON = "why_reason"        # Why did it happen?
    WHY_GOAL = "why_goal"            # What's the goal?
    
    # Method & manner
    HOW_METHOD = "how_method"        # How is it done?
    HOW_MANNER = "how_manner"        # In what manner?
    
    # Complex reasoning
    TEMPORAL_ORDER = "temporal_order"    # What order?
    SPATIAL_RELATION = "spatial_relation"  # Spatial relationships?
    STATE_CHANGE = "state_change"    # How does state change?
```

#### 3. **Agents** (`part3_agents.py`)

**Agent A: Gemini Baseline**
- **Approach**: Extract video frames → Convert to images → Query Gemini vision API
- **Strengths**: Direct visual understanding, no graph required
- **Weaknesses**: No structured knowledge, limited temporal reasoning
- **Use Case**: Quick visual analysis, baseline comparison

**Agent B: Heuristic Graph**
- **Approach**: Map question types → Hand-crafted Cypher queries → Execute on Neo4j
- **Strengths**: Structured reasoning, fast execution, interpretable
- **Weaknesses**: No visual grounding, limited to graph content
- **Use Case**: When graph is complete and accurate

**Agent C: Augmented SOTA**
- **Approach**: Query graph for structure → Get visual context → Synthesize with LLM
- **Strengths**: Best of both worlds, handles complex reasoning
- **Weaknesses**: Slower, requires both systems
- **Use Case**: Production use, research evaluation

---

## Setup

### Prerequisites

1. **Python 3.10+**
2. **Parts 1+2 Installed** (for video processing and graph generation)
3. **Google Gemini API Key** (for Agents A and C)
4. **Neo4j Database** (for Agents B and C)

### Installation

```bash
# 1. Install dependencies
pip install google-generativeai neo4j opencv-python pillow numpy

# 2. Set up Google Gemini API
# Get API key from: https://makersuite.google.com/app/apikey
export GEMINI_API_KEY="your-api-key-here"

# 3. Set up Neo4j
# Option A: Install locally (https://neo4j.com/download/)
# Option B: Use Neo4j Aura (cloud)
# Default: bolt://localhost:7687

# 4. Verify installation
python production/test_part3.py --help
```

### Configuration

```python
# Quick setup
from production.part3_config import apply_config, BALANCED_CONFIG

# Apply balanced configuration (recommended)
apply_config(BALANCED_CONFIG)

# Or create custom configuration
from production.part3_config import create_custom_config

custom = create_custom_config(
    CLIP_MAX_FRAMES=15,
    GEMINI_TEMPERATURE=0.2,
    EC15_QUESTION_COUNT=20
)
apply_config(custom)
```

---

## Quick Start

### 1. Single Question Test

```python
from production.part3_query_evaluation import Question, QuestionType
from production.part3_agents import AgentA_GeminiBaseline

# Create a question
question = Question(
    id="q1",
    question_text="What action is the person performing?",
    question_type=QuestionType.WHAT_ACTION,
    video_path="test_video.mp4",
    timestamp_start=5.0,
    timestamp_end=10.0,
    ground_truth="walking"
)

# Initialize agent
agent = AgentA_GeminiBaseline(api_key="your-gemini-api-key")

# Get answer
answer = agent.answer_question(question, "test_video.mp4", None)

print(f"Answer: {answer.answer_text}")
print(f"Confidence: {answer.confidence}")
print(f"Reasoning: {answer.reasoning}")
```

### 2. Compare All Three Agents

```python
from production.test_part3 import init_all_agents, test_single_question

# Initialize agents
agents = init_all_agents(
    api_key="your-gemini-api-key",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your-password"
)

# Test with all agents
test_single_question(question, agents, "test_video.mp4")
```

### 3. Run Full Evaluation

```python
from production.test_part3 import run_all_tests

# Run all benchmarks
run_all_tests(
    video_path="test_video.mp4",
    config_preset="balanced"
)
```

### 4. Command Line

```bash
# Single question test
python production/test_part3.py \
    --video test_video.mp4 \
    --test single \
    --gemini-key YOUR_KEY

# Run EC-15 benchmark
python production/test_part3.py \
    --video test_video.mp4 \
    --test ec15 \
    --config balanced \
    --gemini-key YOUR_KEY \
    --neo4j-password YOUR_NEO4J_PASSWORD

# Run all tests
python production/test_part3.py \
    --video test_video.mp4 \
    --test all \
    --config high_quality \
    --gemini-key YOUR_KEY \
    --neo4j-password YOUR_NEO4J_PASSWORD
```

---

## Agent Comparison

### When to Use Each Agent

| Scenario | Agent A | Agent B | Agent C |
|----------|---------|---------|---------|
| **Visual details** (colors, objects) | ✓✓✓ | ✗ | ✓✓ |
| **Temporal reasoning** | ✓ | ✓✓ | ✓✓✓ |
| **Spatial relationships** | ✓✓ | ✓ | ✓✓✓ |
| **Causality & intent** | ✓ | ✗ | ✓✓✓ |
| **Counting** | ✓✓ | ✓✓✓ | ✓✓✓ |
| **State changes** | ✓✓ | ✓ | ✓✓✓ |
| **Speed** | Fast | Very Fast | Slow |
| **Cost** | Medium | Low | High |
| **Interpretability** | Low | High | Medium |

### Performance Benchmarks

**EC-15 Benchmark Results** (example):

```
Agent A (Gemini Baseline):
  Accuracy: 65%
  Avg Confidence: 0.72
  Avg Time: 2.3s/question
  Best: WHAT_ACTION, WHERE_LOCATION, WHO_COUNT
  Worst: WHY_REASON, TEMPORAL_ORDER

Agent B (Heuristic Graph):
  Accuracy: 58%
  Avg Confidence: 0.85
  Avg Time: 0.5s/question
  Best: WHO_COUNT, WHEN_START, WHEN_END
  Worst: WHAT_OBJECT (no vision), WHY_GOAL

Agent C (Augmented SOTA):
  Accuracy: 78%
  Avg Confidence: 0.81
  Avg Time: 4.1s/question
  Best: All complex types (TEMPORAL_ORDER, STATE_CHANGE)
  Worst: Simple visual (overkill)
```

### Strategy Recommendations

**For Production:**
- Use Agent C for best accuracy
- Cache graph queries to reduce latency
- Fall back to Agent A if graph is incomplete

**For Development:**
- Use Agent A for quick iteration
- Use Agent B to validate graph quality
- Use Agent C for final evaluation

**For Research:**
- Run all three for comparison
- Use high_quality config
- Analyze failure modes per agent

---

## Benchmarks

### EC-15 (Egocentric Comprehension 15 Types)

Tests all 15 question types with diverse scenarios.

```python
from production.test_part3 import test_ec15_benchmark

test_ec15_benchmark(
    agents=agents,
    video_path="test_video.mp4",
    output_dir="results/ec15"
)
```

**Sample Questions:**
- "What action is the person performing?" (WHAT_ACTION)
- "When does the person start running?" (WHEN_START)
- "Where is the person located?" (WHERE_LOCATION)
- "How many people are in the scene?" (WHO_COUNT)
- "Why did the person stop?" (WHY_REASON)
- "What is the order of events?" (TEMPORAL_ORDER)

### LOT-Q (Long-form Temporal Questions)

Tests complex temporal reasoning with long-form answers.

```python
from production.test_part3 import test_lotq_benchmark

test_lotq_benchmark(
    agents=agents,
    video_path="test_video.mp4",
    output_dir="results/lotq"
)
```

**Sample Questions:**
- "Describe the complete sequence of actions from start to finish."
- "What state changes occur to objects over time?"
- "How do spatial relationships change during the video?"
- "What are the goals and sub-goals demonstrated?"
- "Identify all interactions between entities, in order."

---

## Configuration

### Configuration Presets

```python
from production.part3_config import compare_presets

compare_presets()
```

**Available Presets:**

| Preset | Frames | Sample Rate | Graph Results | EC-15 | LOT-Q | Use Case |
|--------|--------|-------------|---------------|-------|-------|----------|
| `baseline` | 5 | 1/10 | 10 | 10 | 3 | Basic testing |
| `balanced` | 10 | 1/5 | 20 | 15 | 5 | Production (recommended) |
| `high_quality` | 20 | 1/3 | 50 | 20 | 10 | Research evaluation |
| `fast` | 3 | 1/15 | 5 | 5 | 2 | Quick iteration |
| `vision_only` | 15 | 1/3 | 0 | 15 | 5 | Agent A comparison |
| `graph_only` | 0 | - | 30 | 15 | 5 | Agent B comparison |

### Custom Configuration

```python
from production.part3_config import create_custom_config, apply_config

# Create custom config
config = create_custom_config(
    CLIP_MAX_FRAMES=15,           # More frames for better quality
    GEMINI_TEMPERATURE=0.1,       # More deterministic
    MAX_GRAPH_RESULTS=30,         # More graph context
    EC15_QUESTION_COUNT=20,       # More questions
    SIMILARITY_THRESHOLD=0.80     # Stricter evaluation
)

# Apply it
apply_config(config)
```

---

## API Reference

### Core Classes

#### `Question`

```python
question = Question(
    id="unique_id",
    question_text="What is happening?",
    question_type=QuestionType.WHAT_ACTION,
    video_path="path/to/video.mp4",
    timestamp_start=5.0,
    timestamp_end=10.0,
    ground_truth="expected answer",
    metadata={"difficulty": "easy"}
)
```

#### `Agent` (Base Class)

```python
class Agent:
    def answer_question(
        self, 
        question: Question,
        video_path: str,
        neo4j_driver: Optional[Any]
    ) -> Answer:
        """Answer a single question."""
        pass
    
    def batch_answer(
        self, 
        questions: List[Question],
        video_path: str,
        neo4j_driver: Optional[Any]
    ) -> List[Answer]:
        """Answer multiple questions."""
        pass
```

#### `Evaluator`

```python
evaluator = Evaluator()

# Evaluate single answer
result = evaluator.evaluate_answer(
    question=question,
    answer=answer,
    ground_truth="expected"
)

# Compute metrics for agent
metrics = evaluator.compute_metrics("Agent A")

# Generate report
evaluator.generate_report("results/report.json")
```

### Agent-Specific APIs

#### `AgentA_GeminiBaseline`

```python
agent_a = AgentA_GeminiBaseline(
    api_key="your-gemini-api-key",
    model_name="gemini-2.0-flash-exp"
)

answer = agent_a.answer_question(question, video_path, None)
```

#### `AgentB_HeuristicGraph`

```python
agent_b = AgentB_HeuristicGraph(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

answer = agent_b.answer_question(question, video_path, None)
```

#### `AgentC_AugmentedSOTA`

```python
agent_c = AgentC_AugmentedSOTA(
    api_key="your-gemini-api-key",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

# Configure what to use
Config.USE_VISION_CONTEXT = True
Config.USE_GRAPH_CONTEXT = True
Config.RERANK_RESULTS = True

answer = agent_c.answer_question(question, video_path, None)
```

---

## Evaluation Metrics

### Computed Metrics

```python
metrics = {
    'accuracy': 0.78,               # Fraction correct
    'avg_confidence': 0.81,         # Average confidence score
    'avg_processing_time': 4.1,     # Seconds per question
    'total_questions': 15,
    'correct_answers': 12,
    'by_question_type': {
        'WHAT_ACTION': {'accuracy': 0.9, 'count': 2},
        'WHEN_START': {'accuracy': 0.8, 'count': 2},
        # ...
    }
}
```

### Evaluation Methods

1. **Exact Match**: `answer.lower() == ground_truth.lower()`
2. **Semantic Similarity**: Cosine similarity of embeddings (threshold: 0.75)
3. **Temporal Tolerance**: For time-based answers (±2 seconds)
4. **Numerical Tolerance**: For counting questions (±1)

---

## Troubleshooting

### Common Issues

**Problem: "No Gemini API key provided"**
```bash
# Solution 1: Environment variable
export GEMINI_API_KEY="your-key"

# Solution 2: Pass directly
python test_part3.py --gemini-key YOUR_KEY

# Solution 3: Config file
from production.part3_config import update_gemini_api_key
update_gemini_api_key("your-key")
```

**Problem: "Cannot connect to Neo4j"**
```bash
# Check Neo4j is running
neo4j status

# Or start it
neo4j start

# Verify connection
curl http://localhost:7474

# Update credentials
python test_part3.py --neo4j-uri bolt://localhost:7687 \
                     --neo4j-user neo4j \
                     --neo4j-password YOUR_PASSWORD
```

**Problem: "Agent A fails on long videos"**
```python
# Reduce frame count
from production.part3_config import create_custom_config
config = create_custom_config(CLIP_MAX_FRAMES=5)
apply_config(config)
```

**Problem: "Agent B returns empty answers"**
- Check that Neo4j graph is populated (run Parts 1+2 first)
- Verify graph schema matches expected structure
- Use Agent C which handles incomplete graphs better

**Problem: "Out of memory"**
```python
# Use FAST config
from production.part3_config import FAST_CONFIG
apply_config(FAST_CONFIG)

# Or process in smaller batches
for batch in chunks(questions, batch_size=5):
    test_batch_questions(batch, agents, video_path)
```

---

## Integration with Parts 1+2

### End-to-End Pipeline

```python
# Step 1: Run Parts 1+2 (video → graph)
from production.integrated_pipeline import run_integrated_pipeline

metadata = run_integrated_pipeline(
    video_path="input_video.mp4",
    output_dir="output"
)

# Step 2: Run Part 3 (graph → answers)
from production.test_part3 import test_integration_with_parts12

test_integration_with_parts12(
    parts12_output_dir="output",
    output_dir="results/part3"
)
```

### Using Generated Graph

```python
# Load graph from Parts 1+2
import json
with open("output/knowledge_graph.json") as f:
    graph = json.load(f)

# Initialize agents with graph
agents = init_all_agents(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

# Questions are automatically answered using the graph
questions = generate_sample_questions("output/video.mp4")
test_batch_questions(questions, agents, "output/video.mp4")
```

---

## Advanced Usage

### Custom Question Generation

```python
from production.part3_query_evaluation import Question, QuestionType

def generate_domain_specific_questions(video_path: str) -> List[Question]:
    """Generate questions for your specific domain."""
    return [
        Question(
            id="custom1",
            question_text="Is the person wearing safety equipment?",
            question_type=QuestionType.WHAT_OBJECT,
            video_path=video_path,
            timestamp_start=0.0,
            timestamp_end=30.0,
            ground_truth="yes, helmet and gloves",
            metadata={"domain": "safety_inspection"}
        ),
        # Add more...
    ]
```

### Custom Evaluation Metrics

```python
from production.part3_query_evaluation import Evaluator

class CustomEvaluator(Evaluator):
    def evaluate_answer(self, question, answer, ground_truth):
        # Add custom evaluation logic
        result = super().evaluate_answer(question, answer, ground_truth)
        
        # Add domain-specific scoring
        if question.metadata.get("domain") == "safety_inspection":
            result.metrics["safety_score"] = self._compute_safety_score(answer)
        
        return result
```

---

## Next Steps

1. **Run Quick Start**: Test with a sample video
2. **Configure API Keys**: Set up Gemini and Neo4j
3. **Run Benchmarks**: Evaluate on EC-15 and LOT-Q
4. **Analyze Results**: Compare agent performance
5. **Integrate**: Connect with Parts 1+2 for full pipeline
6. **Customize**: Adapt for your specific use case

For more information, see:
- `QUICKSTART_PART3.md` - 30-second quick start
- `production/part3_config.py` - Configuration options
- `production/test_part3.py` - Testing infrastructure
- `production/part3_agents.py` - Agent implementations

---

**Questions?** Open an issue or contact the Orion Research Team.
