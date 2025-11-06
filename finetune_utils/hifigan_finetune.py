# Modified NeMo HiFi-GAN Finetuned Example

from nemo.core.config import hydra_runner
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import HifiGanModel
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="../config/hifigan")
def main(cfg):
    # Trainer setup
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Model setup
    model = HifiGanModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)

    # Log and callback
    lr_logger = LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])

    # Start training
    trainer.fit(model)


if __name__ == "__main__":
    main()
