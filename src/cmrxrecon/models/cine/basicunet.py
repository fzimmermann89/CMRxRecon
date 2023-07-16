import torch
import einops
from cmrxrecon.nets.unet import Unet
from cmrxrecon.nets.unet_andreas import Unet as UnetA
from . import CineModel


class BasicUNet(CineModel):
    def __init__(self, input_coils=True, output_coils=False, lr=3e-3, weight_decay=1e-6, schedule=True):
        super().__init__()
        self.net = Unet(
            dim=2.5,
            channels_in=1 + 20 * input_coils,
            channels_out=20 if output_coils else 1,
            layer=4,
            filters=32,
            padding_mode="circular",
            residual="inner",
            norm="group8",
            feature_growth=lambda d: (1, 2, 1.5, 1.34, 1, 1)[d],
            activation="leakyrelu",
            change_filters_last=False,
        )
        self.input_coils = input_coils
        self.output_coils = output_coils
        with torch.no_grad():
            self.net.last[0].bias.zero_()
            self.net.last[0].weight *= 0.1

    def forward(self, k: torch.Tensor, mask: torch.Tensor, **other) -> dict:
        x0 = torch.fft.ifftn(k, dim=(-2, -1), norm="ortho")
        rss = x0.abs().square().sum(1, keepdim=True).sqrt()
        norm = 0.0001  # rss.max()
        # print("norm", norm)

        if self.input_coils:
            # input the coil reconstructions as channels alongsite the rss
            net_input = einops.rearrange(
                torch.view_as_real(x0),
                "b c z t x y r -> (b z) (r c) t x y",
            )
            net_input = torch.cat(
                (
                    net_input,
                    einops.rearrange(rss, "b c z t x y -> (b z) c t x y"),
                ),
                1,
            ) * (
                1 / norm
            )  # normalize the input
        else:
            # only input the rss as a channel
            net_input = einops.rearrange(rss, "b c z t x y -> (b z) c t x y") * (1 / norm)
        rss = rss.squeeze(1)
        x_net = self.net(net_input)
        if self.output_coils:
            x_net = einops.rearrange(x_net, "(b z) (c r) t x y -> b c z t x y r", b=x0.shape[0], c=10, r=2)
            pred = torch.add(torch.view_as_real(x0), x_net, alpha=1e-3)
            pred = torch.sum(pred.square(), dim=(-1, 1)).sqrt()
        else:
            x_net = einops.rearrange(x_net, "(b z) c t x y -> b z (c t) x y", b=x0.shape[0], c=1)

            pred = torch.add(rss, x_net, alpha=1e-3)  # alpha unnormalizes the output
        return dict(prediction=pred, rss=rss)
