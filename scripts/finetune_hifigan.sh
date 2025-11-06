#!/bin/bash

uv run python ../finetune_utils/hifigan_finetune.py --config-name=hifigan.yaml \
  train_dataset=../data/train_manifest.json \
  validation_datasets=../data/val_manifest.json \
  ++trainer.max_epochs=100 \
  trainer.check_val_every_n_epoch=20 \
  model.train_ds.dataloader_params.batch_size=16 \
  model.validation_ds.dataloader_params.batch_size=16 \
  model.optim.lr=0.00001 \
  ~model.optim.sched \
  exp_manager.exp_dir=./log \
  +init_from_nemo_model={hifigan_path} \
  model/train_ds=train_ds_finetune \
  model/validation_ds=val_ds_finetune

echo "HiFi-GAN finetuning complete."