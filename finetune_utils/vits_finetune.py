# Modified NeMo VITS Finetuned Example

from nemo.core.config import hydra_runner
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import VitsModel
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import numpy as np
from metric import compute_mcd


class VitsWithMCD(VitsModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_mcds = []  # store MCD values during validation

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)

        audio, audio_len, text, text_len = batch
        audio_pred, _, mask, *_ = self.net_g.infer(text, text_len, None, max_len=1000)
        audio_pred = audio_pred.squeeze()
        audio_pred_len = (
            mask.sum([1, 2]).long() * self._cfg.validation_ds.dataset.hop_length
        )

        tgt_mel, tgt_len = self.audio_to_melspec_processor(audio, audio_len)
        pred_mel, pred_len = self.audio_to_melspec_processor(audio_pred, audio_pred_len)

        tgt_mel, pred_mel = (
            tgt_mel.detach().cpu().numpy(),
            pred_mel.detach().cpu().numpy(),
        )
        tgt_len, pred_len = tgt_len.cpu().numpy(), pred_len.cpu().numpy()

        for i, (ref, pred) in enumerate(zip(tgt_mel, pred_mel)):
            ref, pred = ref[:, : tgt_len[i]], pred[:, : pred_len[i]]
            self.val_mcds.append(compute_mcd(ref, pred))

        return None

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


@hydra_runner(config_path="../conf")
def main(cfg):
    # Trainer setup
    trainer = pl.Trainer(use_distributed_sampler=False, **cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Model setup
    model = VitsWithMCD(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)

    # Log and callback
    lr_logger = LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])

    # Train
    trainer.fit(model)


if __name__ == "__main__":
    main()
