# Stage 6 Implementation: LLM Reasoning

## Overview

Stage 6 adds natural language reasoning capabilities to Orion using Ollama for local LLM inference. This enables conversational Q&A about video content with context-aware answers.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: LLM REASONING                                                     │
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │  User Question  │────▶│   Intent Parse  │────▶│  Cypher Query   │       │
│  │  (Natural Lang) │     │   (Rule-based)  │     │  (Memgraph)     │       │
│  └─────────────────┘     └─────────────────┘     └────────┬────────┘       │
│                                                           │                 │
│                                                           ▼                 │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │  Final Answer   │◀────│  LLM Synthesis  │◀────│   Evidence      │       │
│  │  (Streaming)    │     │  (Ollama)       │     │   (JSON)        │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. ReasoningModel (`orion/query/reasoning.py`)

Ollama-based LLM wrapper for:
- **Cypher Generation**: NL → Cypher query translation
- **Answer Synthesis**: Evidence → Natural language response
- **Streaming**: Token-by-token output for real-time interaction
- **Conversation Tracking**: Context memory for follow-up questions

```python
from orion.query.reasoning import ReasoningModel, ReasoningConfig

config = ReasoningConfig(model="qwen2.5:14b-instruct-q8_0")
model = ReasoningModel(config)

# Generate Cypher
cypher = model.generate_cypher("What objects were near the laptop?")

# Synthesize answer
answer = model.synthesize_answer(
    question="What did the person interact with?",
    evidence=[{"object": "book", "holder": "person", "count": 77}]
)
```

### 2. OrionRAG (`orion/query/rag_v2.py`)

Enhanced RAG interface combining Stage 5 retrieval with Stage 6 reasoning:

- **Query Routing**: Maps NL questions to appropriate handlers
- **LLM Integration**: Optional LLM synthesis for richer answers
- **Vector Search**: Similarity queries using V-JEPA2 embeddings
- **Streaming**: Real-time answer generation

```python
from orion.query import OrionRAG

rag = OrionRAG(
    host="127.0.0.1",
    port=7687,
    enable_llm=True,
    llm_model="qwen2.5:14b-instruct-q8_0",
)

# Template-based answer (fast)
result = rag.query("What objects are in the video?", use_llm=False)

# LLM-synthesized answer (richer)
result = rag.query("Describe the interactions", use_llm=True)

# Streaming response
for token in rag.stream_query("What did the person do?"):
    print(token, end="")
```

### 3. Query CLI (`orion/cli/run_query.py`)

Interactive query interface:

```bash
# Interactive mode
python -m orion.cli.run_query --episode my_episode

# Single question
python -m orion.cli.run_query --episode my_episode -q "What objects are visible?"

# Without LLM (template answers)
python -m orion.cli.run_query --episode my_episode --no-llm
```

## Query Types

| Type | Example Question | Handler |
|------|------------------|---------|
| `all_objects` | "What objects are in the video?" | List all entity classes |
| `object_location` | "Where did the book appear?" | Temporal appearance range |
| `spatial_near` | "What was near the laptop?" | NEAR relationship query |
| `interactions` | "What did the person interact with?" | HELD_BY relationship query |
| `temporal` | "What happened at 25s?" | Frame-based lookup |
| `similarity` | "Find objects similar to the person" | V-JEPA2 embedding search |
| `object_info` | "Tell me about the laptop" | General entity query |

## Evaluation Scripts

### Full Pipeline Evaluation

```bash
python scripts/eval_full_pipeline.py \
    --video data/examples/video.mp4 \
    --episode eval_001
```

Tests all 6 stages with Gemini validation.

### Conversational Evaluation

```bash
python scripts/eval_conversation.py \
    --video data/examples/video.mp4 \
    --episode eval_001
```

Tests back-and-forth Q&A with Gemini-based verdict scoring.

### Stage 6 Unit Tests

```bash
python scripts/test_stage6.py --host 127.0.0.1
```

Verifies Ollama, Memgraph, and RAG components.

## Configuration

### Ollama Setup (Lambda AI)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start server
ollama serve &

# Pull model (14B recommended for reasoning quality)
ollama pull qwen2.5:14b-instruct-q8_0
```

### Recommended Models

| Model | Size | Quality | Speed | Use Case |
|-------|------|---------|-------|----------|
| `qwen2.5:14b-instruct-q8_0` | ~14GB | High | Medium | Default reasoning |
| `qwen2.5:7b-instruct-q8_0` | ~7GB | Good | Fast | Resource-constrained |
| `gemma3:4b` | ~4GB | Fair | Very Fast | Quick tests |

### ReasoningConfig Options

```python
@dataclass
class ReasoningConfig:
    model: str = "qwen2.5:14b-instruct-q8_0"
    base_url: str = "http://localhost:11434"
    
    temperature_cypher: float = 0.0   # Deterministic Cypher
    temperature_synthesis: float = 0.3  # Slightly creative answers
    max_tokens: int = 1024
    max_conversation_history: int = 10
```

## Metrics & Evaluation

### Key Metrics

- **Query Latency**: Time from question to answer (target: <2s with LLM)
- **Accuracy Score**: Gemini-validated answer correctness (target: >80%)
- **Hallucination Rate**: False claims per conversation (target: 0)
- **Confidence**: Self-reported retrieval confidence (0-1)

### Sample Results (Stage 5 Evaluation)

```
Episode: stage5_eval_001
Video: data/examples/video.mp4

Pipeline Metrics:
- Entities: 10
- Frames: 169
- Observations: 1,890
- Spatial Relations (NEAR): 142
- Interactions (HELD_BY): 77

Query Performance:
- Avg Latency: 450ms (template) / 2,100ms (LLM)
- Accuracy: 80% (Gemini-validated)
- Hallucinations: 0
```

## Troubleshooting

### Ollama Connection Failed

```
Error: Failed to connect to Ollama at http://localhost:11434
```

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve &

# Pull the model
ollama pull qwen2.5:14b-instruct-q8_0
```

### Memgraph No Data

```
Warning: No entities found in Memgraph!
```

**Solution:**
```bash
# Run perception pipeline with Memgraph export
python -m orion.cli.run_showcase \
    --episode my_episode \
    --video my_video.mp4 \
    --memgraph
```

### Slow Responses

If LLM responses are slow (>5s):

1. Use a smaller model: `--model qwen2.5:7b-instruct-q8_0`
2. Disable LLM for exploratory queries: `--no-llm`
3. Check GPU utilization on Lambda instance

## Future Work

- [ ] Multi-turn reasoning with explicit context injection
- [ ] Cypher query caching for repeated patterns
- [ ] Confidence calibration for answer reliability
- [ ] Multi-episode cross-reference queries
- [ ] Voice input/output integration
