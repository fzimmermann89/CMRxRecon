import torch
import einops
from cmrxrecon.nets.unet import Unet
from . import CineModel


class BasicUNet(CineModel):
    def __init__(
        self,
        input_coils=False,
    ):
        super().__init__()
        self.net = Unet(
            dim=2.5,
            channels_in=1 + 20 * input_coils,
            channels_out=1,
            layer=4,
            filters=32,
            padding_mode="circular",
            residual="inner",
            feature_growth=lambda d: (1, 2, 1.5, 1.33, 1, 1)[d],
            activation="leakyrelu",
        )
        self.input_coils = input_coils
        with torch.no_grad():
            self.net.last[0].weight *= 1e-1
            self.net.last[0].bias.zero_()

    def forward(self, k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x0 = torch.fft.ifftn(k, dim=(-2, -1), norm="ortho")
        rss = x0.abs().square().sum(1, keepdim=True).sqrt()
        norm = rss.max()

        if self.input_coils:
            x0 = einops.rearrange(
                torch.view_as_real(x0),
                "b c z t x y r -> (b z) (r c) t x y",
            )
            x0 = torch.cat(
                (
                    x0,
                    einops.rearrange(rss, "b c z t x y -> (b z) c t x y"),
                ),
                1,
            ) * (1 / norm)
        else:
            x0 = einops.rearrange(rss, "b c z t x y -> (b z) c t x y") * (1 / norm)
        rss = rss.squeeze(1)
        x_net = self.net(x0)
        x_net = einops.rearrange(x_net, "(b z) c t x y -> b z (c t) x y", b=x0.shape[0], c=1)
        pred = torch.add(rss, x_net, alpha=norm)
        return pred, rss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=3e-3, total_steps=self.trainer.estimated_stepping_batches, pct_start=0.05, anneal_strategy="cos", cycle_momentum=True, div_factor=30, final_div_factor=1e3, verbose=False
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
