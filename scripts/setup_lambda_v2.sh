#!/bin/bash
# Orion v2 Lambda Setup Script
# Run this on a fresh Lambda instance with A10 GPU

set -e

echo "=========================================="
echo "  ORION V2 - Lambda Environment Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "\n${GREEN}[STEP]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check GPU
print_step "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv || {
    print_error "No GPU detected!"
    exit 1
}

# System info
print_step "System Info"
echo "Hostname: $(hostname)"
echo "Python: $(python3 --version)"
echo "Disk space:"
df -h / | tail -1

# Navigate to repo
REPO_DIR="${HOME}/orion-core-fs/orion-research"
if [ ! -d "$REPO_DIR" ]; then
    print_step "Cloning Orion repository..."
    mkdir -p "${HOME}/orion-core-fs"
    cd "${HOME}/orion-core-fs"
    git clone https://github.com/riddhimanrana/orion-research.git
fi

cd "$REPO_DIR"
print_step "Working in: $(pwd)"

# Create virtual environment
print_step "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA
print_step "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
print_step "Installing Orion dependencies..."
pip install -e .

# Install additional dependencies for v2
print_step "Installing v2 dependencies..."
pip install accelerate transformers>=4.45.0 sentence-transformers

# Install V-JEPA2 dependencies (video processing)
print_step "Installing video processing dependencies..."
pip install torchcodec av

# Install Memgraph driver
print_step "Installing Memgraph Python driver..."
pip install gqlalchemy neo4j

# Install Ultralytics for YOLO-World
print_step "Installing Ultralytics (YOLO-World)..."
pip install ultralytics>=8.3.0

# Verify installations
print_step "Verifying installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import sentence_transformers; print(f'Sentence Transformers installed')"
python3 -c "from ultralytics import YOLOWorld; print('YOLO-World available')"

# Pre-download models
print_step "Pre-downloading models (this may take a while)..."

# Download YOLO-World
print_step "Downloading YOLO-World v2 (x-large)..."
python3 -c "
from ultralytics import YOLOWorld
model = YOLOWorld('yolov8x-worldv2.pt')
print('YOLO-World downloaded')
"

# Download sentence transformer
print_step "Downloading all-mpnet-base-v2..."
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print('Sentence transformer downloaded')
"

# Download V-JEPA2 (this is large ~2GB)
print_step "Downloading V-JEPA2 encoder..."
python3 -c "
from transformers import AutoModel, AutoVideoProcessor
import torch

processor = AutoVideoProcessor.from_pretrained('facebook/vjepa2-vitl-fpc64-256')
model = AutoModel.from_pretrained(
    'facebook/vjepa2-vitl-fpc64-256',
    torch_dtype=torch.float16,
    device_map='auto',
    attn_implementation='sdpa'
)
print('V-JEPA2 downloaded and loaded')
"

# Set up Docker for Memgraph
print_step "Setting up Memgraph (Docker)..."
if command -v docker &> /dev/null; then
    # Check if memgraph is already running
    if docker ps | grep -q memgraph; then
        print_warn "Memgraph container already running"
    else
        docker run -d --name memgraph \
            -p 7687:7687 -p 7444:7444 \
            -v memgraph_data:/var/lib/memgraph \
            memgraph/memgraph-platform || print_warn "Memgraph may already exist, trying to start..."
        docker start memgraph 2>/dev/null || true
    fi
    
    # Wait for Memgraph to be ready
    sleep 5
    
    # Test connection
    python3 -c "
from neo4j import GraphDatabase
try:
    driver = GraphDatabase.driver('bolt://localhost:7687')
    driver.verify_connectivity()
    print('Memgraph connection successful!')
    driver.close()
except Exception as e:
    print(f'Memgraph connection failed: {e}')
    print('You may need to wait a moment and try again')
"
else
    print_warn "Docker not available. Install Docker to use Memgraph."
fi

# Set up Ollama
print_step "Setting up Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama server in background
print_step "Starting Ollama server..."
pkill ollama || true  # Kill any existing
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 3

# Pull model (14B fits in 24GB A10)
print_step "Pulling Qwen2.5-14B model (this will take a while)..."
ollama pull qwen2.5:14b-instruct-q8_0 || print_warn "Ollama pull failed, you may need to run this manually"

# Create activation script
print_step "Creating activation script..."
cat > "${REPO_DIR}/activate_orion.sh" << 'EOF'
#!/bin/bash
# Source this file to activate Orion v2 environment
export ORION_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ORION_HOME}/venv/bin/activate"
export PYTHONPATH="${ORION_HOME}:${PYTHONPATH}"
export TRANSFORMERS_CACHE="${ORION_HOME}/.cache/huggingface"
export HF_HOME="${ORION_HOME}/.cache/huggingface"

# Suppress TensorFlow warnings
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export TF_CPP_MIN_LOG_LEVEL=3

echo "Orion v2 environment activated"
echo "  ORION_HOME: ${ORION_HOME}"
echo "  Python: $(which python)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
EOF
chmod +x "${REPO_DIR}/activate_orion.sh"

# Summary
echo ""
echo "=========================================="
echo "  ORION V2 SETUP COMPLETE"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source ${REPO_DIR}/activate_orion.sh"
echo ""
echo "To run Orion:"
echo "  orion init --episode test --video /path/to/video.mp4"
echo "  orion analyze --episode test"
echo ""
echo "Services:"
echo "  - Memgraph: bolt://localhost:7687"
echo "  - Ollama: http://localhost:11434"
echo ""
echo "Quick test:"
echo "  python -c \"from orion import __version__; print(f'Orion {__version__}')\""
echo ""
