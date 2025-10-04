# Part 3 Quick Start Guide

Get started with the Query & Evaluation Engine in under 30 seconds!

## 30-Second Quick Start

```bash
# 1. Install dependencies
pip install google-generativeai neo4j opencv-python pillow

# 2. Set API key
export GEMINI_API_KEY="your-key-here"

# 3. Run test
python production/test_part3.py --video test_video.mp4 --test single --gemini-key $GEMINI_API_KEY
```

Done! You've just tested Agent A (Gemini Baseline) on a single question.

---

## Step-by-Step Guide

### Step 1: Install Dependencies (2 minutes)

```bash
# Core dependencies
pip install google-generativeai neo4j opencv-python pillow numpy

# Optional: For enhanced evaluation
pip install sentence-transformers scikit-learn
```

### Step 2: Configure API Keys (1 minute)

**Option A: Environment Variables (Recommended)**

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export NEO4J_PASSWORD="your-neo4j-password"
```

**Option B: Python Configuration**

```python
from production.part3_config import update_gemini_api_key, update_neo4j_credentials

update_gemini_api_key("your-gemini-api-key")
update_neo4j_credentials(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your-password"
)
```

### Step 3: Test Single Agent (30 seconds)

```python
from production.part3_query_evaluation import Question, QuestionType
from production.part3_agents import AgentA_GeminiBaseline

# Create question
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
agent = AgentA_GeminiBaseline(api_key="your-key")

# Get answer
answer = agent.answer_question(question, "test_video.mp4", None)

print(f"Answer: {answer.answer_text}")
print(f"Confidence: {answer.confidence}")
```

### Step 4: Compare All Agents (2 minutes)

```python
from production.test_part3 import init_all_agents, test_single_question

# Initialize all agents
agents = init_all_agents(
    api_key="your-gemini-key",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="your-neo4j-password"
)

# Test with all agents
test_single_question(question, agents, "test_video.mp4")
```

### Step 5: Run Full Evaluation (5 minutes)

```bash
# Run EC-15 benchmark
python production/test_part3.py \
    --video test_video.mp4 \
    --test ec15 \
    --config balanced \
    --gemini-key $GEMINI_API_KEY \
    --neo4j-password $NEO4J_PASSWORD

# View results
cat test_results/ec15/evaluation_report.json
```

---

## Common Workflows

### Workflow 1: Vision-Only Testing (No Graph Required)

```python
from production.part3_config import apply_config, VISION_ONLY_CONFIG
from production.test_part3 import init_agent_a, test_question_types

# Configure for vision-only
apply_config(VISION_ONLY_CONFIG)

# Initialize Agent A
agent_a = init_agent_a(api_key="your-key")

# Test all question types
test_question_types([agent_a], "test_video.mp4")
```

**Use Case:** Quick testing, no Neo4j setup required

### Workflow 2: Graph-Only Testing (No Vision API)

```python
from production.part3_config import apply_config, GRAPH_ONLY_CONFIG
from production.test_part3 import init_agent_b, test_ec15_benchmark

# Configure for graph-only
apply_config(GRAPH_ONLY_CONFIG)

# Initialize Agent B
agent_b = init_agent_b(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="your-password"
)

# Run benchmark
test_ec15_benchmark([agent_b], "test_video.mp4")
```

**Use Case:** Validate graph quality, test Cypher queries

### Workflow 3: Full Pipeline with Parts 1+2

```bash
# Step 1: Run Parts 1+2 (video → graph)
python production/integrated_pipeline.py \
    --video input_video.mp4 \
    --output output

# Step 2: Run Part 3 (graph → answers)
python production/test_part3.py \
    --video output/video.mp4 \
    --test integration \
    --gemini-key $GEMINI_API_KEY \
    --neo4j-password $NEO4J_PASSWORD
```

**Use Case:** End-to-end testing, production deployment

---

## Configuration Quick Reference

### Presets

```python
from production.part3_config import apply_config, get_preset

# Fast (development)
apply_config(get_preset('fast'))

# Balanced (production)
apply_config(get_preset('balanced'))

# High Quality (research)
apply_config(get_preset('high_quality'))
```

### Custom Configuration

```python
from production.part3_config import create_custom_config

config = create_custom_config(
    CLIP_MAX_FRAMES=15,
    GEMINI_TEMPERATURE=0.2,
    EC15_QUESTION_COUNT=20
)
apply_config(config)
```

---

## CLI Reference

```bash
# Test single question
python production/test_part3.py --test single --video VIDEO_PATH

# Test all question types
python production/test_part3.py --test types --video VIDEO_PATH

# Run EC-15 benchmark
python production/test_part3.py --test ec15 --video VIDEO_PATH

# Run LOT-Q benchmark
python production/test_part3.py --test lotq --video VIDEO_PATH

# Test integration with Parts 1+2
python production/test_part3.py --test integration

# Run all tests
python production/test_part3.py --test all --video VIDEO_PATH

# Options:
#   --config {baseline,balanced,high_quality,fast}
#   --gemini-key YOUR_KEY
#   --neo4j-uri bolt://localhost:7687
#   --neo4j-user neo4j
#   --neo4j-password YOUR_PASSWORD
```

---

## Troubleshooting

### Error: "No Gemini API key provided"

```bash
# Set environment variable
export GEMINI_API_KEY="your-key"

# Or pass as argument
python test_part3.py --gemini-key YOUR_KEY
```

### Error: "Cannot connect to Neo4j"

```bash
# Check Neo4j status
neo4j status

# Start Neo4j
neo4j start

# Or use Docker
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
```

### Error: "Video file not found"

```bash
# Use absolute path
python test_part3.py --video /full/path/to/video.mp4

# Or use example video
python test_part3.py --video data/examples/example1.mp4
```

### Error: "Out of memory"

```python
# Use FAST config
from production.part3_config import FAST_CONFIG
apply_config(FAST_CONFIG)
```

---

## Next Steps

1. ✅ **Completed Quick Start**
2. 📚 **Read Full Documentation**: `README_PART3.md`
3. 🔬 **Run Benchmarks**: Test on EC-15 and LOT-Q
4. 🔧 **Customize**: Adapt for your use case
5. 🚀 **Integrate**: Connect with Parts 1+2

---

## Example Output

```
Testing Question: q1
================================================================================
Question: What action is the person performing?
Type: what_action
Ground Truth: walking
Timestamp: 5.0s - 10.0s

Agent A (Gemini Baseline)
--------------------------------------------------------------------------------
Answer: The person is walking along a path in a park.
Confidence: 0.87
Processing Time: 2.3s
Reasoning: Based on visual analysis of frames 5-10, the person's gait and...

Agent B (Heuristic Graph)
--------------------------------------------------------------------------------
Answer: walking
Confidence: 0.95
Processing Time: 0.5s
Reasoning: Retrieved from graph: (person)-[:PERFORMED]->(walk_event)

Agent C (Augmented SOTA)
--------------------------------------------------------------------------------
Answer: The person is walking towards the fountain.
Confidence: 0.91
Processing Time: 4.1s
Reasoning: Visual context shows walking motion, graph confirms walk_event...

================================================================================
```

---

**Ready to dive deeper?** See `README_PART3.md` for comprehensive documentation.
