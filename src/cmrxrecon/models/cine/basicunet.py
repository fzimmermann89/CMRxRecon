import torch
import einops
from cmrxrecon.nets.unet import Unet
from . import CineModel


class BasicUNet(CineModel):
    def __init__(self):
        super().__init__()
        self.net = torch.compile(
            Unet(
                dim=2.5,
                channels_in=21,
                channels_out=1,
                layer=4,
                filters=32,
                padding_mode="circular",
                feature_growth=lambda d: (2, 1.5, 1, 1, 1, 1)[d],
            )
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

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-5)
