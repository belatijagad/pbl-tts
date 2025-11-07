#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

DATASET_NAME=tts_indo
DATASET_REPO_ID=TODO
DATASET_FILENAME=TODO

echo "▶ Starting VITS inference..."
echo "▶ Using fine-tuned version..."
echo "▶ Using Indonesian dataset..."
echo "▶ Project Root: $PROJECT_ROOT"

uv run python $PROJECT_ROOT/main.py \
  dataset.data_name=$DATASET_NAME \
  dataset.repo_id=$DATASET_REPO_ID \
  dataset.filename=$DATASET_FILENAME \
  model_choice=vits

echo "Inference complete."
