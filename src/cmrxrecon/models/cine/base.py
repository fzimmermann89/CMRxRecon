from abc import ABC
from re import T
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from neptune.new.types import File as neptuneFile
from torch.nn import init


class TrainingMixin_xrss(pl.LightningModule, ABC):
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


class ValidationMixin(pl.LightningModule, ABC):
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


class CineModel(ValidationMixin, TrainingMixin_xrss, pl.LightningModule, ABC):
    def init(self):
        self.save_hyperparameters()
