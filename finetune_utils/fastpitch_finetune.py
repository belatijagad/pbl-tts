# Modified Nemo FastPitch Finetuned example

from nemo.core.config import hydra_runner
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FastPitchModel
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import numpy as np
from metric import compute_mcd
import torch


class FastPitchWithMCD(FastPitchModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_mcds = []  # store MCD values during validation

    def validation_step(self, batch, batch_idx):
        outputs = super().validation_step(batch, batch_idx)
        gt_audio, gt_audio_len, text, *_ = batch

        gt_mels, gt_mel_lens = self.preprocessor(
            input_signal=gt_audio, length=gt_audio_len
        )
        gt_mels = gt_mels.detach().cpu().numpy()
        gt_mel_lens = gt_mel_lens.detach().cpu().numpy()

        for i, txt in enumerate(text):
            result = self.generate_spectrogram(tokens=txt.unsqueeze(0), speaker=None)
            pred_mel, pred_mel_len = (
                (result, result.shape[-1])
                if isinstance(result, torch.Tensor)
                else result
            )
            pred_mel = pred_mel.detach().cpu().numpy()[0]
            gt_mel = gt_mels[i, :, : gt_mel_lens[i]]
            self.val_mcds.append(compute_mcd(gt_mel, pred_mel))

        return outputs

    def on_validation_epoch_end(self):
        """Compute average MCD across all validation batches."""
        print(f"val_mcds: {self.val_mcds}")
        if len(self.val_mcds) > 0:
            avg_mcd = float(np.mean(self.val_mcds))
            std_mcd = float(np.std(self.val_mcds))
            self.log("val_avg_mcd_dtw", avg_mcd, prog_bar=True, sync_dist=True)
            self.log("val_std_mcd_dtw", std_mcd, prog_bar=True, sync_dist=True)
            self.val_mcds.clear()  # reset for next epoch
        super().on_validation_epoch_end()


# hydra_runner call
@hydra_runner(config_path="../conf")
def main(cfg):
    if hasattr(cfg.model.optim, "sched"):
        logging.warning(
            "You are using an optimizer scheduler while finetuning. Are you sure this is intended?"
        )
    if cfg.model.optim.lr > 1e-3 or cfg.model.optim.lr < 1e-5:
        logging.warning("The recommended learning rate for finetuning is 2e-4")

    # Trainer
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Model
    model = FastPitchWithMCD(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)

    # Log and callback
    lr_logger = LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])

    trainer.fit(model)


if __name__ == "__main__":
    main()
