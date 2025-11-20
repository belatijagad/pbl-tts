#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

DATASET_NAME=tts_indo
DATASET_REPO_ID=indonesian-nlp/librivox-indonesia
DATASET_FILENAME="data/audio_test.tgz"
DATASET_LANGUAGE=indonesian
DATASET_METADATA=metadata_test.csv

VITS_CKPT=$PROJECT_ROOT/log/VITS/checkpoints/vits_model.nemo

echo "▶ Starting VITS inference..."
echo "▶ Using fine-tuned version..."
echo "▶ Using Indonesian dataset..."
echo "▶ Project Root: $PROJECT_ROOT"

uv run python $PROJECT_ROOT/main.py \
  dataset.data_name="$DATASET_NAME" \
  dataset.repo_id="$DATASET_REPO_ID" \
  dataset.filename="$DATASET_FILENAME" \
  dataset.variant=indonesian \
  dataset.language="$DATASET_LANGUAGE" \
  dataset.metadata_filename="$DATASET_METADATA" \
  dataset.extracted_subdir=librivox-indonesia \
  dataset.split_subdir=test \
  dataset.split.train_size=140 \
  dataset.split.eval_size=10 \
  dataset.split.seed=42 \
  model_choice=vits \
  vits_config.load_method=checkpoint \
  vits_config.checkpoint_path=$VITS_CKPT \
  +method=ft 

echo "Inference complete."
