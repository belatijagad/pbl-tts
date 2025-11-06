#!/bin/bash

uv run python ../finetune_utils/vits_finetune.py --config-name=../config/vits.yaml \
  train_dataset=../data/train_manifest.json \
  validation_datasets=../data/val_manifest.json \
  trainer.max_epochs=100 \
  +trainer.check_val_every_n_epoch=20 \
  ++model.train_ds.batch_sampler.batch_size=32 \
  ++model.validation_ds.dataloader_params.batch_size=16 \
  phoneme_dict_path=../finetune_utils/ipa_cmudict-0.7b_nv23.01.txt \
  heteronyms_path=../finetune_utils/heteronyms-052722 \
  exp_manager.exp_dir=./log \
  model.sample_rate=22050 \
  +init_from_nemo_model={vits_path} \
  trainer.accelerator=gpu \
  trainer.strategy=ddp_find_unused_parameters_true \
  trainer.devices=1

echo "VITS finetuning complete."
