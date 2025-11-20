#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

GENERATOR_CKPT=$PROJECT_ROOT/log/FastPitch/2025-11-20_15-33-01-sunda/checkpoints/FastPitch--val_loss=2.1788-epoch=99.ckpt
VOCODER_CKPT=$PROJECT_ROOT/log/HifiGan/2025-11-20_15-42-17-sunda/checkpoints/HifiGan--val_loss=42.3759-epoch=99-last.ckpt

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
