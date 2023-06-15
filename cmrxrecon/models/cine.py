import pytorch_lightning as pl
import torch
import einops

from cmrxrecon.nets.unet import Unet
from neptune.new.types import File as neptuneFile


class BasicUNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = Unet(
            dim=2.5,
            channels_in=21,
            channels_out=1,
            layer=4,
            filters=32,
            padding_mode="circular",
            feature_growth=lambda d: (1.5, 1.33, 1, 1, 1, 1)[d],
        )

    def forward(self, k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x0 = torch.fft.ifftn(k, dim=(-2, -1), norm="ortho")
        rss = x0.abs().square().sum(1, keepdim=True).sqrt()

        x0 = einops.rearrange(
            torch.view_as_real(x0),
            "b c z t x y r -> (b z) (r c) t x y",
        )
        norm = rss.max()
        x0 = torch.cat(
            (
                x0,
                einops.rearrange(rss, "b c z t x y -> (b z) c t x y"),
            ),
            1,
        ) * (1 / norm)
        rss = rss.squeeze(1)
        x_net = self.net(x0) * norm
        x_net = einops.rearrange(x_net, "(b z) c t x y -> b z (c t) x y", b=x0.shape[0], c=1)
        pred = rss + x_net
        return pred, rss

    def training_step(self, batch, batch_idx):
        k, mask, gt = batch
        prediction, rss = self(k, mask)
        loss = torch.nn.functional.mse_loss(prediction, gt)
        rss_loss = torch.nn.functional.mse_loss(rss, gt)
        self.log("advantage", (rss_loss - loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("rss_loss", rss_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        if batch_idx % 20 == 0:
            img = prediction[0, 0, 0, :, :].detach().cpu().numpy()
            img = img - img.min()
            img = img / img.max()
            self.logger.experiment["train/image"].log(neptuneFile.as_image(img))
        return loss

    def validation_step(self, batch, batch_idx):
        k, mask, gt = batch
        prediction = self(k, mask)
        loss = torch.nn.functional.mse_loss(prediction, gt)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
