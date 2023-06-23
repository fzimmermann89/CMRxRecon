from abc import ABC
import torch
import pytorch_lightning as pl
from neptune.new.types import File as neptuneFile
from pytorch_lightning.loggers.neptune import NeptuneLogger


class TrainingMixin_xrss(ABC):
    def training_step(self, batch, batch_idx):
        gt = batch.pop("gt")
        ret = self(**batch)
        prediction, rss = ret["prediction"], ret["rss"]
        loss = torch.nn.functional.mse_loss(prediction, gt)
        rss_loss = torch.nn.functional.mse_loss(rss, gt)
        self.log("train_advantage", (rss_loss - loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("rss_loss", rss_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss


class ValidationMixin(ABC):
    def validation_step(self, batch, batch_idx):
        gt = batch.pop("gt")
        ret = self(**batch)
        prediction, rss = ret["prediction"], ret["rss"]
        loss = torch.nn.functional.mse_loss(prediction, gt)
        rss_loss = torch.nn.functional.mse_loss(rss, gt)
        self.log("val_advantage", (rss_loss - loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if batch_idx == 0:
            for logger in self.loggers:
                if isinstance(logger, NeptuneLogger):
                    # only for neptune logger, log the first image
                    img = prediction[0, 0, 0, :, :].detach().cpu().numpy()
                    img = img - img.min()
                    img = img / img.max()
                    logger.experiment["val/image"].log(neptuneFile.as_image(img))


class DefaultOptimizerMixin(ABC):
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.schedule:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches, pct_start=0.05, anneal_strategy="cos", cycle_momentum=True, div_factor=30, final_div_factor=1e3, verbose=False
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        else:
            return optimizer


class BaseModel(ValidationMixin, TrainingMixin_xrss, DefaultOptimizerMixin, pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
