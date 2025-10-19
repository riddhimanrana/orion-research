# Orion Configuration, Deployment & Integration Guide

## Configuration System Overview

### Three-Tier Architecture

```
┌─────────────────────────────────────────────┐
│ Environment Variables (.env file)           │
│ - ORION_NEO4J_PASSWORD                      │
│ - ORION_NEO4J_URI (optional)                │
│ - ORION_OLLAMA_BASE_URL (optional)          │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ ConfigManager Singleton                     │
│ - Lazy loads config on first access         │
│ - Resolves environment variables            │
│ - Validates required parameters             │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ OrionConfig Instance                        │
│ - detection: DetectionConfig                │
│ - embedding: EmbeddingConfig                │
│ - clustering: ClusteringConfig              │
│ - description: DescriptionConfig            │
│ - neo4j: Neo4jConfig                        │
│ - ollama: OllamaConfig                      │
└─────────────────────────────────────────────┘
```

### Configuration Files

#### 1. orion/config.py (508 lines)

**Purpose**: Define all configuration dataclasses and preset factories

**Key Dataclasses**:

```python
@dataclass
class DetectionConfig:
    model_variant: str                  # "11x", "11m", "11s", "11n"
    confidence_threshold: float         # 0.45
    nms_threshold: float               # 0.45
    image_size: int                    # 640
    device: str                        # "cpu", "cuda", "mps"

@dataclass
class EmbeddingConfig:
    model_name: str                    # "openai/clip-vit-base-patch32"
    embedding_dimension: int           # 512, 1024, 2048
    batch_size: int                    # 32
    device: str                        # "cpu", "cuda", "mps"

@dataclass
class ClusteringConfig:
    min_cluster_size: int              # 10
    min_samples: int                   # 5
    metric: str                        # "cosine"
    cluster_selection_epsilon: float   # 0.5

@dataclass
class DescriptionConfig:
    model_name: str                    # "fastvlm-0.5b" or similar
    max_tokens: int                    # 128
    temperature: float                 # 0.7

@dataclass
class Neo4jConfig:
    uri: str                           # "bolt://localhost:7687"
    user: str                          # "neo4j"
    password: str                      # From env var

@dataclass
class OllamaConfig:
    base_url: str                      # "http://localhost:11434"
    model: str                         # "mistral", "llama2", etc.
    timeout: int                       # 30 seconds

@dataclass
class OrionConfig:
    detection: DetectionConfig
    embedding: EmbeddingConfig
    clustering: ClusteringConfig
    description: DescriptionConfig
    neo4j: Neo4jConfig
    ollama: OllamaConfig
    temporal_window_size: int
    state_change_threshold: float
    spatial_distance_threshold: float
```

**Preset Functions**:

```python
def get_fast_config() -> OrionConfig:
    """Minimal latency, lower accuracy"""
    return OrionConfig(
        detection=DetectionConfig(model_variant="11n"),      # Smallest model
        embedding=EmbeddingConfig(embedding_dimension=512),  # Smallest embeddings
        clustering=ClusteringConfig(min_cluster_size=15),    # Larger clusters
        ...
    )

def get_balanced_config() -> OrionConfig:
    """Production-recommended balance"""
    return OrionConfig(
        detection=DetectionConfig(model_variant="11m"),
        embedding=EmbeddingConfig(embedding_dimension=1024),
        clustering=ClusteringConfig(min_cluster_size=10),
        ...
    )

def get_accurate_config() -> OrionConfig:
    """Maximum accuracy, higher resource usage"""
    return OrionConfig(
        detection=DetectionConfig(model_variant="11x"),      # Largest model
        embedding=EmbeddingConfig(embedding_dimension=2048), # Largest embeddings
        clustering=ClusteringConfig(min_cluster_size=5),     # Smaller clusters
        ...
    )
```

#### 2. orion/config_manager.py (240 lines)

**Purpose**: Singleton credential/config management with environment variable loading

```python
from orion.config import OrionConfig, get_balanced_config

class ConfigManager:
    _instance: Optional['ConfigManager'] = None
    _config: Optional[OrionConfig] = None
    
    @classmethod
    def get_config(cls) -> OrionConfig:
        """Get singleton config instance with lazy loading"""
        if cls._instance is None:
            cls._instance = ConfigManager()
        if cls._config is None:
            cls._config = cls._instance._load_config()
        return cls._config
    
    def _load_config(self) -> OrionConfig:
        """Load configuration from environment variables"""
        
        # 1. Load base config
        config = get_balanced_config()  # Default preset
        
        # 2. Override with environment variables
        neo4j_password = os.getenv("ORION_NEO4J_PASSWORD")
        if neo4j_password:
            config.neo4j.password = neo4j_password
        
        neo4j_uri = os.getenv("ORION_NEO4J_URI")
        if neo4j_uri:
            config.neo4j.uri = neo4j_uri
        
        ollama_url = os.getenv("ORION_OLLAMA_BASE_URL")
        if ollama_url:
            config.ollama.base_url = ollama_url
        
        # 3. Load from JSON config file if present
        config_file = Path.cwd() / "orion_config.json"
        if config_file.exists():
            self._load_from_json(config_file, config)
        
        return config
    
    def _load_from_json(self, path: Path, config: OrionConfig):
        """Merge JSON file configuration"""
        with open(path) as f:
            json_config = json.load(f)
        
        # Update dataclass fields from JSON
        if 'detection' in json_config:
            config.detection = DetectionConfig(**json_config['detection'])
        # ... similarly for other configs
```

### Environment Variable Setup

Create `.env` file in project root:

```bash
# Required
ORION_NEO4J_PASSWORD=your_secure_password_here

# Optional - defaults provided in config.py
ORION_NEO4J_URI=bolt://localhost:7687
ORION_OLLAMA_BASE_URL=http://localhost:11434
```

Load in Python:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Usage Patterns

#### Pattern 1: Using Default Config

```python
from orion.config_manager import ConfigManager

config = ConfigManager.get_config()
detection_confidence = config.detection.confidence_threshold
neo4j_uri = config.neo4j.uri
```

#### Pattern 2: Using Presets

```python
from orion.config import get_fast_config, get_balanced_config, get_accurate_config

# For quick testing
fast_config = get_fast_config()

# For production
prod_config = get_balanced_config()

# For research/maximum quality
accurate_config = get_accurate_config()
```

#### Pattern 3: Custom Configuration

```python
from orion.config import OrionConfig, DetectionConfig, EmbeddingConfig
from orion.config_manager import ConfigManager

custom = OrionConfig(
    detection=DetectionConfig(model_variant="11x", confidence_threshold=0.6),
    embedding=EmbeddingConfig(embedding_dimension=2048),
    # ... other configs
)

# This custom config is used directly in pipeline calls
result = run_pipeline(video_path, config=custom)
```

---

## Component Integration Guide

### 1. Perception Engine Integration

```python
from orion.perception_engine import PerceptionEngine
from orion.config_manager import ConfigManager

config = ConfigManager.get_config()
engine = PerceptionEngine(config.detection, config.embedding)

# Process video
import cv2
video = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    result = engine.process_frame(frame)
    # result.detections: List[Detection]
    # result.embeddings: np.ndarray [N, embedding_dim]
    # result.processing_time_ms: float
```

### 2. Tracking Engine Integration

```python
from orion.tracking_engine import TrackingEngine
from orion.config_manager import ConfigManager

config = ConfigManager.get_config()
tracker = TrackingEngine(config.temporal_window_size)

# Update with detections from perception
assignments = tracker.update_tracks(
    frame_idx=0,
    detections=result.detections,
    embeddings=result.embeddings
)

# assignments: List[(entity_id, Detection)]
for entity_id, detection in assignments:
    entity = tracker.entities[entity_id]
    # Track entity properties
```

### 3. Semantic Uplift Integration

```python
from orion.semantic_uplift import SemanticUplift
from orion.config_manager import ConfigManager

config = ConfigManager.get_config()
uplift = SemanticUplift(config)

# Process entity trajectories (after full video analysis)
semantic_data = uplift.process_entities(
    entities=tracker.entities,
    entity_trajectories=entity_trajectories
)

# semantic_data contains:
# - events: List[Event]
# - state_changes: List[StateChange]
# - spatial_relationships: List[SpatialRelationship]
```

### 4. Knowledge Graph Integration

```python
from orion.knowledge_graph import KnowledgeGraphBuilder
from orion.config_manager import ConfigManager

config = ConfigManager.get_config()

# KnowledgeGraphBuilder automatically uses ConfigManager if no manager provided
kg_builder = KnowledgeGraphBuilder()

# Or explicitly provide Neo4jManager
from orion.neo4j_manager import Neo4jManager
neo4j_manager = Neo4jManager(
    uri=config.neo4j.uri,
    user=config.neo4j.user,
    password=config.neo4j.password
)
kg_builder = KnowledgeGraphBuilder(neo4j_manager)

# Build graph
success = kg_builder.build_graph(semantic_data)
```

---

## Deployment Scenarios

### Scenario 1: Local Development

```bash
# 1. Start Neo4j (Docker)
docker run --name neo4j -p 7687:7687 -p 7474:7474 \
    -e NEO4J_AUTH=neo4j/test_password \
    neo4j:latest

# 2. Start Ollama (for LLM event composition)
ollama serve  # Defaults to localhost:11434

# 3. Create .env
cat > .env << EOF
ORION_NEO4J_PASSWORD=test_password
ORION_NEO4J_URI=bolt://localhost:7687
ORION_OLLAMA_BASE_URL=http://localhost:11434
EOF

# 4. Run pipeline
python scripts/test_complete_pipeline.py video.mp4
```

### Scenario 2: Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY . .

# Set environment
ENV ORION_NEO4J_PASSWORD=${NEO4J_PASSWORD}
ENV ORION_NEO4J_URI=${NEO4J_URI}
ENV ORION_OLLAMA_BASE_URL=${OLLAMA_URL}

# Run pipeline
CMD ["python", "scripts/run_pipeline.py", "${VIDEO_PATH}"]
```

**Build & Run**:

```bash
docker build -t orion:latest .

docker run --rm \
    -e ORION_NEO4J_PASSWORD=secure_pass \
    -e ORION_NEO4J_URI=bolt://neo4j:7687 \
    -e ORION_OLLAMA_BASE_URL=http://ollama:11434 \
    -v /path/to/videos:/videos \
    orion:latest python scripts/run_pipeline.py /videos/video.mp4
```

### Scenario 3: Kubernetes Deployment

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: orion-config
data:
  orion_config.json: |
    {
      "detection": {
        "model_variant": "11m",
        "confidence_threshold": 0.45
      },
      "embedding": {
        "embedding_dimension": 1024
      },
      "clustering": {
        "min_cluster_size": 10
      }
    }

---
apiVersion: v1
kind: Secret
metadata:
  name: orion-secrets
type: Opaque
data:
  neo4j-password: base64_encoded_password

---
apiVersion: batch/v1
kind: Job
metadata:
  name: orion-pipeline-job
spec:
  template:
    spec:
      containers:
      - name: orion
        image: orion:latest
        env:
        - name: ORION_NEO4J_PASSWORD
          valueFrom:
            secretKeyRef:
              name: orion-secrets
              key: neo4j-password
        - name: ORION_NEO4J_URI
          value: "bolt://neo4j-service:7687"
        volumeMounts:
        - name: config
          mountPath: /app/orion_config.json
          subPath: orion_config.json
        - name: videos
          mountPath: /videos
      volumes:
      - name: config
        configMap:
          name: orion-config
      - name: videos
        persistentVolumeClaim:
          claimName: video-pvc
      restartPolicy: Never
```

---

## Performance Tuning Guide

### Memory Optimization

```python
# For low-memory environments (e.g., 8GB)
config = get_fast_config()  # Smaller models, lower embedding dim

# Batch processing reduces peak memory
BATCH_SIZE = 16  # Process 16 frames at a time instead of all

for i in range(0, total_frames, BATCH_SIZE):
    batch_frames = frames[i:i+BATCH_SIZE]
    results = [engine.process_frame(f) for f in batch_frames]
```

### Speed Optimization

```python
# Use smaller YOLO model for faster detection
config = get_fast_config()

# Frame sampling: process every Nth frame
SAMPLE_RATE = 2  # Process every 2nd frame
for frame_idx in range(0, total_frames, SAMPLE_RATE):
    # Process only sampled frames
```

### Accuracy Optimization

```python
# Use larger models and smaller clustering thresholds
config = get_accurate_config()

# Increase temporal window for better state detection
config.temporal_window_size = 60  # Instead of default 30
```

---

## Testing & Validation

### Unit Tests

```bash
# Run specific test suite
python -m pytest tests/unit/ -v

# Run with coverage
python -m pytest tests/unit/ --cov=orion --cov-report=html
```

### Integration Tests

```bash
# Full pipeline test
python scripts/test_complete_pipeline.py test_video.mp4

# Specific component test
python scripts/test_tracking.py
python scripts/test_semantic_uplift.py
```

### Configuration Validation

```python
# Verify configuration can be loaded
from orion.config_manager import ConfigManager
config = ConfigManager.get_config()

# Verify Neo4j connection
from orion.neo4j_manager import Neo4jManager
manager = Neo4jManager(
    uri=config.neo4j.uri,
    user=config.neo4j.user,
    password=config.neo4j.password
)
manager.connect()
stats = manager.get_stats()
print(f"Connected to Neo4j: {stats}")
manager.close()
```

---

## Troubleshooting

### Common Issues

**Issue 1: Neo4j connection fails**
```
Solution: 
1. Check ORION_NEO4J_PASSWORD is set correctly
2. Verify Neo4j service is running: curl bolt://localhost:7687
3. Check firewall rules if remote Neo4j
```

**Issue 2: Out of memory during embedding generation**
```
Solution:
1. Use get_fast_config() to reduce embedding dimension
2. Reduce batch_size in EmbeddingConfig
3. Process video in chunks with separate Neo4j sessions
```

**Issue 3: Low detection accuracy**
```
Solution:
1. Increase detection model size: "11x" instead of "11n"
2. Lower confidence_threshold from 0.45 to 0.30
3. Check video quality/lighting
```

**Issue 4: Slow semantic uplift processing**
```
Solution:
1. Increase HDBSCAN min_cluster_size to reduce computation
2. Use smaller temporal_window_size (15 instead of 30 frames)
3. Reduce number of state change detection dimensions
```

---

## Production Checklist

- [ ] Environment variables properly configured and secured
- [ ] Neo4j database initialized with appropriate indexes
- [ ] Ollama service deployed and tested
- [ ] Configuration preset selected based on hardware (fast/balanced/accurate)
- [ ] Model weights downloaded and cached
- [ ] Video input validation implemented
- [ ] Error handling and logging configured
- [ ] Neo4j queries tested and optimized
- [ ] Performance tested with representative video dataset
- [ ] Monitoring and alerting set up
- [ ] Database backup strategy implemented
- [ ] Documentation updated with deployment details

This guide enables complete integration of all Orion components with secure credential management and flexible deployment options.
