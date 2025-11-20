#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

LOG_DIR="$PROJECT_ROOT/log"
PRETRAINED_DIR="$PROJECT_ROOT/nemo_pretrained"
HIFIGAN_PATH="$PRETRAINED_DIR/tts_en_hifigan.nemo"
TRAIN_DATASET_PATH="$PROJECT_ROOT/data/tts_indo/train_manifest.json"
VAL_DATASET_PATH="$PROJECT_ROOT/data/tts_indo/val_manifest.json"
mkdir -p $LOG_DIR

echo "▶ Starting HiFi-GAN Finetuning (Indonesian)..."
echo "▶ Project Root: $PROJECT_ROOT"
echo "▶ Log Directory:  $LOG_DIR"

uv run python $PROJECT_ROOT/finetune_utils/hifigan_finetune.py --config-name=hifigan.yaml \
  ++trainer.max_epochs=500 \
  trainer.check_val_every_n_epoch=50 \
  model.train_ds.dataloader_params.batch_size=32 \
  model.validation_ds.dataloader_params.batch_size=32 \
  model.optim.lr=0.00001 \
  ~model.optim.sched \
  exp_manager.exp_dir=$LOG_DIR \
  +init_from_nemo_model=$HIFIGAN_PATH \
  model.train_ds.dataset.manifest_filepath=$TRAIN_DATASET_PATH \
  model.validation_ds.dataset.manifest_filepath=$VAL_DATASET_PATH \
  model/train_ds=train_ds_finetune \
  model/validation_ds=val_ds_finetune

echo "HiFi-GAN finetuning complete. Logs saved to $LOG_DIR."
