from abc import ABC
import torch
import pytorch_lightning as pl
from neptune.new.types import File as neptuneFile
from pytorch_lightning.loggers.neptune import NeptuneLogger
from cmrxrecon.models.utils.ssim import ssim
import matplotlib.pyplot as plt
import random, string
from pathlib import Path
from datetime import datetime
import numpy as np


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
        ssim_value = ssim(gt / gt.max(), prediction / prediction.max())
        nrmse = torch.nn.functional.mse_loss(prediction / prediction.max(), gt / gt.max())

        self.log("val_advantage", (rss_loss - loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_nrmse", nrmse, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if batch_idx == 0:
            acceleration = int(batch["acceleration"].item())
            scalemin, scalemax = gt.min().item(), gt.max().item()

            rndpath = Path(
                f"{self.__class__.__name__}_Acc{acceleration}_{datetime.now().strftime('%m%d%H%M')}_{''.join(random.choices(string.ascii_uppercase + string.digits, k=4))}"
            )

            def scale(data):
                return (data[0, 0].detach().cpu().numpy() - scalemin) / (scalemax - scalemin)

            def plot(data, colorbar=False, cmap="gray"):
                fig, ax = plt.subplots(1, figsize=(6 + 2 * colorbar, 2.5), tight_layout=True)
                data = data[(data.ndim - 2) * (0,)]
                s = ax.matshow(data, cmap=cmap)
                if colorbar:
                    plt.colorbar(s, ax=ax)
                ax.axis("off")
                return fig

            def log(name, data, **kwargs):
                for logger in self.loggers:
                    if isinstance(logger, NeptuneLogger):
                        logger.experiment["val/" + name].log(neptuneFile.as_image(data))
                        break
                else:
                    rndpath.mkdir(exist_ok=True)
                    if isinstance(data, (torch.Tensor, np.ndarray)):
                        np.save(str(rndpath / name) + ".npy", data)
                    if isinstance(data, plt.Figure):
                        f = data
                    else:
                        f = plot(data, **kwargs)
                    imgfilename = str(rndpath / name) + ".png"
                    f.savefig(imgfilename)
                    print("no neptune logger. saved as", imgfilename)

            gt_img = scale(gt)
            pred_img = scale(prediction)
            xrss_img = scale(rss)
            error = np.abs(pred_img - gt_img)
            log("prediction", pred_img)
            log("gt", gt_img)
            log("xrss", xrss_img)
            log("error", error, cmap="viridis", colorbar=True)


class TestPredictMixin(ABC):
    def test_step(self, batch, batch_idx):
        ret = self(**batch)
        return ret["prediction"]

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        ret = self(**batch)
        return ret["prediction"]


class DefaultOptimizerMixin(ABC):
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.schedule:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                anneal_strategy="cos",
                cycle_momentum=True,
                div_factor=10,
                final_div_factor=1e2,
                verbose=False,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        else:
            return optimizer


class BaseModel(ValidationMixin, TrainingMixin_xrss, TestPredictMixin, DefaultOptimizerMixin, pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
