#!/usr/bin/env bash
# Lambda AI Setup for Stage 6 LLM Reasoning
# Run this after SSH into Lambda instance

set -e

echo "============================================================"
echo "ORION STAGE 6 SETUP - Lambda AI"
echo "============================================================"

# 1. Clone/Update repo
echo ""
echo "[1/5] Setting up Orion repository..."
if [ -d ~/orion-research ]; then
    cd ~/orion-research
    git pull
    echo "  ✓ Repository updated"
else
    cd ~
    git clone https://github.com/your-repo/orion-research.git
    cd ~/orion-research
    echo "  ✓ Repository cloned"
fi

# 2. Install Python dependencies
echo ""
echo "[2/5] Installing Python dependencies..."
pip install -q neo4j httpx sentence-transformers
echo "  ✓ Python packages installed"

# 3. Start Memgraph with Docker
echo ""
echo "[3/5] Starting Memgraph..."
if command -v docker &> /dev/null; then
    # Check if Memgraph is already running
    if docker ps | grep -q orion-memgraph; then
        echo "  ✓ Memgraph already running"
    else
        docker compose up -d memgraph
        echo "  Waiting for Memgraph to be ready..."
        sleep 10
        echo "  ✓ Memgraph started"
    fi
else
    echo "  ⚠ Docker not found - skipping Memgraph setup"
    echo "    Install Docker or use managed Memgraph instance"
fi

# 4. Install and start Ollama
echo ""
echo "[4/5] Setting up Ollama..."
if command -v ollama &> /dev/null; then
    echo "  ✓ Ollama already installed"
else
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  ✓ Ollama installed"
fi

# Check if Ollama server is running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✓ Ollama server already running"
else
    echo "  Starting Ollama server..."
    ollama serve &
    sleep 3
    echo "  ✓ Ollama server started"
fi

# Pull the recommended model
echo ""
echo "[5/5] Pulling Ollama model..."
MODEL="qwen2.5:14b-instruct-q8_0"
if ollama list | grep -q "$MODEL"; then
    echo "  ✓ Model $MODEL already available"
else
    echo "  Pulling $MODEL (this may take a few minutes)..."
    ollama pull $MODEL
    echo "  ✓ Model downloaded"
fi

# Verify setup
echo ""
echo "============================================================"
echo "VERIFICATION"
echo "============================================================"

# Check Memgraph
echo ""
echo "Memgraph:"
if docker ps | grep -q orion-memgraph; then
    echo "  ✓ Container running"
    echo "  ✓ Port 7687 (Bolt)"
    echo "  ✓ Port 3000 (Lab UI)"
else
    echo "  ✗ Not running"
fi

# Check Ollama
echo ""
echo "Ollama:"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✓ Server running at http://localhost:11434"
    MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; m=json.load(sys.stdin).get('models',[]); print(', '.join(x['name'] for x in m))" 2>/dev/null || echo "error")
    echo "  ✓ Models: $MODELS"
else
    echo "  ✗ Server not running"
fi

echo ""
echo "============================================================"
echo "READY TO RUN"
echo "============================================================"
echo ""
echo "# Process video with all 6 stages:"
echo "python -m orion.cli.run_showcase \\"
echo "    --episode stage6_eval \\"
echo "    --video data/examples/video.mp4 \\"
echo "    --memgraph"
echo ""
echo "# Run Stage 6 evaluation:"
echo "python scripts/eval_full_pipeline.py \\"
echo "    --video data/examples/video.mp4 \\"
echo "    --episode stage6_eval"
echo ""
echo "# Interactive query mode:"
echo "python -m orion.cli.run_query --episode stage6_eval"
echo ""
echo "# Conversational eval with Gemini:"
echo "GEMINI_API_KEY=your_key python scripts/eval_conversation.py \\"
echo "    --video data/examples/video.mp4 \\"
echo "    --episode stage6_eval"
echo ""
echo "============================================================"
