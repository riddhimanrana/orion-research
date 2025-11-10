#!/bin/bash
# Quick start script for Memgraph integration

echo "🚀 Memgraph Integration Quick Start"
echo "===================================="
echo

# Check if Docker is running
if ! docker ps &> /dev/null; then
    echo "❌ Docker is not running"
    echo "   Please start Docker Desktop and run this script again"
    exit 1
fi

echo "✓ Docker is running"

# Check if Memgraph is already running
if docker ps | grep -q memgraph; then
    echo "✓ Memgraph is already running"
else
    echo "📦 Starting Memgraph..."
    cd memgraph-platform 2>/dev/null || {
        echo "   Creating memgraph-platform directory..."
        mkdir -p memgraph-platform
        cd memgraph-platform
        curl -sSf "https://download.memgraph.com/memgraph-platform/docker-compose.yml" -o docker-compose.yml
    }
    
    docker compose up -d
    
    if [ $? -eq 0 ]; then
        echo "✓ Memgraph started"
        echo "   Lab UI: http://localhost:3000"
    else
        echo "❌ Failed to start Memgraph"
        exit 1
    fi
    
    cd ..
fi

# Check Python client
echo
echo "📦 Checking Python client..."
python -c "import mgclient" 2>/dev/null && echo "✓ pymgclient installed" || {
    echo "   Installing pymgclient..."
    pip install pymgclient
}

# Test connection
echo
echo "🔌 Testing connection..."
python -c "import mgclient; mgclient.connect(host='127.0.0.1', port=7687); print('✓ Successfully connected to Memgraph')" || {
    echo "❌ Connection failed"
    echo "   Make sure Memgraph is running: docker ps | grep memgraph"
    exit 1
}

echo
echo "✅ Setup complete!"
echo
echo "Next steps:"
echo "  1. Process a video with Memgraph export:"
echo "     python scripts/run_slam_complete.py \\"
echo "       --video data/examples/video.mp4 \\"
echo "       --yolo-model yolo11n \\"
echo "       --skip 50 \\"
echo "       --no-fastvlm \\"
echo "       --export-memgraph"
echo
echo "  2. Query the graph:"
echo "     python scripts/query_memgraph.py --interactive"
echo
echo "  3. Open Memgraph Lab (web UI):"
echo "     open http://localhost:3000"
echo
