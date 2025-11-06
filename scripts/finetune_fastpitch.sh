#!/bin/bash

uv run python ../finetune_utils/fastpitch_finetune.py --config-name=../config/fastpitch_align.yaml \
  train_dataset=../data/train_manifest.json \
  validation_datasets=../data/val_manifest.json \
  ++trainer.max_epochs=100 \
  trainer.check_val_every_n_epoch=20 \
  model.train_ds.dataloader_params.batch_size=16 \
  model.validation_ds.dataloader_params.batch_size=16 \
  phoneme_dict_path=../finetune_utils/cmudict-0.7b_nv22.10 \
  heteronyms_path=../finetune_utils/heteronyms-052722 \
  exp_manager.exp_dir=./log \
  +init_from_nemo_model={fastpitch_path} \
  model.n_speakers=1 model.pitch_mean=152.3 model.pitch_std=64.0 \
  model.pitch_fmin=30 model.pitch_fmax=512 model.optim.lr=2e-4 \
  ~model.optim.sched model.optim.name=adam trainer.devices=1 trainer.strategy=auto \
  +model.text_tokenizer.add_blank_at=true

echo "Finetuning complete."
