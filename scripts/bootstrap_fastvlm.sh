#!/bin/bash
# Orion FastVLM Apple Silicon bootstrap script
# Downloads and sets up the official Apple FastVLM model for MLX backend

set -e
MODEL_DIR="$(dirname "$0")/../models"
MODEL_ZIP="llava-fastvithd_0.5b_stage3_llm.fp16.zip"
MODEL_URL="https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3_llm.fp16.zip"
MODEL_FOLDER="llava-fastvithd_0.5b_stage3_llm.fp16"

cd "$MODEL_DIR"

if [ -d "$MODEL_FOLDER" ]; then
    echo "[Orion] FastVLM model already exists: $MODEL_FOLDER"
    exit 0
fi

# Download
wget -q "$MODEL_URL" -O "$MODEL_ZIP"

# Unzip quietly, overwrite existing files
unzip -qq -o "$MODEL_ZIP"

# Remove zip file
rm "$MODEL_ZIP"

echo "[Orion] FastVLM model setup complete: $MODEL_FOLDER"
