#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

echo "▶ Starting Fastpitch+HiFi-GAN inference..."
echo "▶ Using fine-tuned version..."
echo "▶ Using Sundanese dataset..."
echo "▶ Project Root: $PROJECT_ROOT"

uv run python $PROJECT_ROOT/main.py \
  model_choice=fastpitch_hifigan \
  +method=pt 

echo "Inference complete."
