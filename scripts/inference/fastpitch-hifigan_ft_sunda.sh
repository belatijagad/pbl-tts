#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

GENERATOR_CKPT=$PROJECT_ROOT/log/FastPitch/2025-11-19_15-21-37/checkpoints/FastPitch--val_loss=0.6710-epoch=199.ckpt
VOCODER_CKPT=$PROJECT_ROOT/log/HifiGan/2025-11-19_15-44-55/checkpoints/HifiGan--val_loss=41.2395-epoch=149.ckpt

echo "▶ Starting Fastpitch+HiFi-GAN inference..."
echo "▶ Using fine-tuned version..."
echo "▶ Using Sundanese dataset..."
echo "▶ Project Root: $PROJECT_ROOT"

uv run python $PROJECT_ROOT/main.py \
  model_choice=fastpitch_hifigan \
  components_config.generator.load_method=checkpoint \
  components_config.generator.checkpoint_path=\"$GENERATOR_CKPT\" \
  components_config.vocoder.load_method=checkpoint \
  components_config.vocoder.checkpoint_path=\"$VOCODER_CKPT\" \
  +method=ft 

echo "Inference complete."
