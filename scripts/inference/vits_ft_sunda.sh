#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

VITS_CKPT=$PROJECT_ROOT/log/VITS/checkpoints/vits_model.nemo

echo "▶ Starting VITS inference..."
echo "▶ Using fine-tuned version..."
echo "▶ Using Indonesian dataset..."
echo "▶ Project Root: $PROJECT_ROOT"

uv run python $PROJECT_ROOT/main.py \
  model_choice=vits \
  vits_config.load_method=checkpoint \
  vits_config.checkpoint_path=$VITS_CKPT

echo "Inference complete."
