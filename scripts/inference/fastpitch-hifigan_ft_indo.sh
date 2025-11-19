#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

DATASET_NAME=tts_indo
DATASET_REPO_ID=syarief-mulyadi/pbl-tts-dataset
DATASET_FILENAME=TODO

GENERATOR_CKPT=$PROJECT_ROOT/log/FastPitch/checkpoints/fastpitch_model.nemo
VOCODER_CKPT=$PROJECT_ROOT/log/HiFiGan/checkpoints/hifigan_model.nemo

echo "▶ Starting Fastpitch+HiFi-GAN inference..."
echo "▶ Using fine-tuned version..."
echo "▶ Using Indonesian dataset..."
echo "▶ Project Root: $PROJECT_ROOT"

uv run python $PROJECT_ROOT/main.py \
  dataset.data_name=$DATASET_NAME \
  dataset.repo_id=$DATASET_REPO_ID \
  dataset.filename=$DATASET_FILENAME \
  model_choice=fastpitch_hifigan \
  components_config.generator.load_method=checkpoint \
  components_config.generator.checkpoint_path=$GENERATOR_CKPT \
  components_config.vocoder.load_method=checkpoint \
  components_config.vocoder.checkpoint_path=$VOCODER_CKPT \
  +method=ft 

echo "Inference complete."
