#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

LOG_DIR="$PROJECT_ROOT/log"
VITS_PATH="$PROJECT_ROOT/nemo_pretrained/tts_en_lj_vits.nemo"
TRAIN_DATASET_PATH="$PROJECT_ROOT/data/tts_sunda/train_manifest.json"
VAL_DATASET_PATH="$PROJECT_ROOT/data/tts_sunda/val_manifest.json"
mkdir -p $LOG_DIR

echo "▶ Starting VITS Finetuning..."
echo "▶ Project Root: $PROJECT_ROOT"
echo "▶ Log Directory:  $LOG_DIR"

uv run python $PROJECT_ROOT/finetune_utils/vits_finetune.py --config-name=vits.yaml \
  train_dataset=$TRAIN_DATASET_PATH \
  validation_datasets=$VAL_DATASET_PATH \
  trainer.max_epochs=500 \
  +trainer.check_val_every_n_epoch=50 \
  ++model.train_ds.batch_sampler.batch_size=32 \
  ++model.validation_ds.dataloader_params.batch_size=32 \
  phoneme_dict_path=$PROJECT_ROOT/finetune_utils/ipa_cmudict-0.7b_nv23.01.txt \
  heteronyms_path=$PROJECT_ROOT/finetune_utils/heteronyms-052722 \
  exp_manager.exp_dir=$LOG_DIR \
  model.sample_rate=22050 \
  +init_from_nemo_model=$VITS_PATH \
  trainer.accelerator=gpu \
  trainer.strategy=ddp_find_unused_parameters_true \
  trainer.devices=1

echo "VITS finetuning complete. Logs saved to $LOG_DIR."