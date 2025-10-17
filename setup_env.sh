#!/bin/bash
# Setup script for Orion Research with LLaVA

set -e  # Exit on error

echo "🚀 Setting up Orion Research environment..."

# Check if conda environment exists
if ! conda env list | grep -q "orion"; then
    echo "❌ Error: 'orion' conda environment not found"
    echo "Please create it first with:"
    echo "  conda create -n orion python=3.10"
    exit 1
fi

# Activate the environment
echo "📦 Activating orion environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orion

# Verify Python version
PYTHON_VERSION=$(python --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "⚠️  Warning: Expected Python 3.10, got $PYTHON_VERSION"
fi

# Install main project
echo "📦 Installing orion package..."
pip install -e .

# Install llava package
echo "📦 Installing llava package..."
pip install -e ./llava

# Verify installation
echo "✅ Verifying installation..."
python -c "from llava.utils import disable_torch_init; print('  ✓ LLaVA imported successfully')"
python -c "from orion.backends.torch_fastvlm import FastVLMTorchWrapper; print('  ✓ FastVLM imported successfully')"

echo ""
echo "✨ Setup complete!"
echo ""
echo "To use the environment:"
echo "  conda activate orion"
echo ""
echo "To test FastVLM:"
echo "  python scripts/test_fastvlm.py data/examples/example1.jpg 'describe the image'"
