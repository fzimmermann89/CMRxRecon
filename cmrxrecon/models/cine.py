import pytorch_lightning as pl
import torch
import einops

from cmrxrecon.nets.unet import Unet


class BasicUNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = Unet(
            dim=2.5, channels_in=20, channels_out=1, layer=4, filters=32, feature_growth=lambda d: (2, 1.5, 1, 1, 1, 1)[d]
        )

    def forward(self, k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x0 = torch.fft.ifftn(k, dim=(-2, -1))
        x0_reshaped = einops.rearrange(
            torch.view_as_real(x0),
            "b c z t x y r -> (b z) (r c) t x y",
        )
        x_net = self.net(x0_reshaped)
        pred = einops.rearrange(x_net, "(b z) c t x y -> b z (c t) x y", b=x0.shape[0], c=1)
        return pred

    def training_step(self, batch, batch_idx):
        k, mask, gt = batch
        prediction = self(k, mask)
        loss = torch.nn.functional.mse_loss(prediction, gt)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        k, mask, gt = batch
        prediction = self(k, mask)
        loss = torch.nn.functional.mse_loss(prediction, gt)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
