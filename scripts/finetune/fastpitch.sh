#!/bin/bash
HYDRA_FULL_ERROR=1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

LOG_DIR="$PROJECT_ROOT/log"
PRETRAINED_DIR="$PROJECT_ROOT/nemo_pretrained"
DATASET_NAME=tts_indo

FAST_PITCH_PATH="$PRETRAINED_DIR/tts_en_fastpitch.nemo"
TRAIN_DATASET_PATH="$PROJECT_ROOT/data/$DATASET_NAME/train_manifest.json"
VAL_DATASET_PATH="$PROJECT_ROOT/data/$DATASET_NAME/val_manifest.json"
SUP_DATA_PATH="$PROJECT_ROOT/sup_data/fastpitch_sup_data"

mkdir -p $LOG_DIR

echo "▶ Starting Fastpitch Finetuning..."
echo "▶ Project Root: $PROJECT_ROOT"
echo "▶ Log Directory:  $LOG_DIR"

uv run python $PROJECT_ROOT/finetune_utils/fastpitch_finetune.py --config-name=fastpitch_align.yaml \
  ++trainer.max_epochs=1000 \
  trainer.check_val_every_n_epoch=50 \
  model.train_ds.dataloader_params.batch_size=128 \
  model.validation_ds.dataloader_params.batch_size=128 \
  phoneme_dict_path=$PROJECT_ROOT/finetune_utils/cmudict-0.7b_nv22.10 \
  heteronyms_path=$PROJECT_ROOT/finetune_utils/heteronyms-052722 \
  exp_manager.exp_dir=$LOG_DIR \
  +init_from_nemo_model=$FAST_PITCH_PATH \
  train_dataset=$TRAIN_DATASET_PATH \
  validation_datasets=$VAL_DATASET_PATH \
  sup_data_path=$SUP_DATA_PATH \
  model.n_speakers=1 model.pitch_mean=152.3 model.pitch_std=64.0 \
  model.pitch_fmin=30 model.pitch_fmax=512 model.optim.lr=2e-4 \
  ~model.optim.sched model.optim.name=adam trainer.devices=1 trainer.strategy=auto \
  +model.text_tokenizer.add_blank_at=true

echo "Fastpitch finetuning complete. Logs saved to $LOG_DIR."
