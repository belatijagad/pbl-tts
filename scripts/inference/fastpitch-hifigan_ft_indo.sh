#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

DATASET_NAME=tts_indo
DATASET_REPO_ID=indonesian-nlp/librivox-indonesia
DATASET_FILENAME="data/audio_test.tgz"
DATASET_LANGUAGE=indonesian
DATASET_METADATA=metadata_test.csv
TOKENIZER_DICT="$PROJECT_ROOT/finetune_utils/cmudict-0.7b_nv22.10"
TOKENIZER_HETERONYMS="$PROJECT_ROOT/finetune_utils/heteronyms-052722"

GENERATOR_CKPT=$PROJECT_ROOT/log/FastPitch/2025-11-20_16-02-44-indo/checkpoints/FastPitch--val_loss=2.5084-epoch=99-last.ckpt
VOCODER_CKPT=$PROJECT_ROOT/log/HifiGan/2025-11-20_15-15-52-indo/checkpoints/HifiGan--val_loss=44.9518-epoch=99.ckpt

echo "▶ Starting Fastpitch+HiFi-GAN inference..."
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
  tokenizer.type=english \
  tokenizer.phoneme_dict_path="$TOKENIZER_DICT" \
  tokenizer.heteronyms_path="$TOKENIZER_HETERONYMS" \
  model_choice=fastpitch_hifigan \
  components_config.generator.load_method=checkpoint \
  components_config.generator.checkpoint_path=\"$GENERATOR_CKPT\" \
  components_config.vocoder.load_method=checkpoint \
  components_config.vocoder.checkpoint_path=\"$VOCODER_CKPT\" \
  +method=ft 

echo "Inference complete."
