import torch
import einops
from cmrxrecon.nets.unet import Unet
from cmrxrecon.nets.mlp import MLP
from typing import *
from . import CineModel
from cmrxrecon.models.utils.crop import crops_by_threshold, uncrop
from cmrxrecon.models.utils.ema import EMA
from cmrxrecon.models.utils import rss
from cmrxrecon.models.utils.mapper import Mapper
from cmrxrecon.models.utils.multicoildc import MultiCoilDCLayer


class CNNWrapper(torch.nn.Module):
    """Wrapper for CNN that performs complex to real conversion, reshaping and complex to real conversion back
    Parameters
    ----------
    net: CNN
    include_rss: if True, the rss is appended to the input
    """

    def __init__(self, net: torch.nn.Module, include_rss: bool = True, crop_threshold: float = 0.005):
        super().__init__()
        self.net = net
        self.include_rss = include_rss
        self.crop_threshold = crop_threshold

    def forward(self, x: torch.Tensor, *args, x_rss=None) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.crop_threshold > 0:
            crops = crops_by_threshold(x.detach(), [self.crop_threshold])
            xc = x[crops]
            if x_rss is not None:
                x_rss_c = x_rss[(crops[0], *crops[2:])]
        else:
            xc = x
            x_rss_c = x_rss
            crops = None

        net_input = einops.rearrange(torch.view_as_real(xc), "b c z t x y r -> (b z) (r c) t x y")
        if self.include_rss:
            if x_rss is None:
                x_rss_c = rss(xc)
            x_rss_c = einops.rearrange(x_rss_c, "b z t x y -> (b z) t x y").unsqueeze(1)
            net_input = torch.cat((net_input, x_rss_c), 1)
        x_net = self.net(net_input, *args)
        x_net = einops.rearrange(x_net, "(b z) (r c) t x y -> b c z t x y r", b=len(x), r=2).contiguous()
        x_net = torch.view_as_complex(x_net)
        x_net = uncrop(x_net, x.shape, crops)
        return x + x_net


class CascadeNN(torch.nn.Module):
    def __init__(self, input_rss=True, Nc: int = 10, T: int = 2, **kwargs):
        super().__init__()
        self.input_rss = input_rss
        embed_dim: int = 128
        net = Unet(
            dim=2.5,
            channels_in=input_rss + 2 * Nc,
            channels_out=2 * Nc,
            layer=3,
            filters=64,
            padding_mode="zeros",
            residual="inner",
            # residual=False,
            # norm=False,
            norm="group8",
            feature_growth=lambda d: (1, 2, 1.5, 1.34, 1, 1)[d],
            activation="leakyrelu",
            change_filters_last=False,
            emb_dim=embed_dim,
            downsample_dimensions=((-1, -2), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3)),
        )
        with torch.no_grad():
            net.last[0].bias.zero_()
        self.net = CNNWrapper(net, include_rss=input_rss)

        self.dc = torch.jit.script(MultiCoilDCLayer(Nc, embed_dim=embed_dim))  # torch.jit.script(

        self.embed_augment_channels = 6
        self.embed_axis_channels = 2
        self.embed_acceleration_map = Mapper([2, 4, 6, 8, 10, 12])
        self.embed_iter_map = Mapper([0, 1, 2, 3, 4, 5])
        self.T = T
        embed_input_channels = (
            self.embed_augment_channels
            + self.embed_axis_channels
            + self.embed_acceleration_map.out_dim
            + self.embed_iter_map.out_dim
        )

        self.embed_net = torch.compile(MLP([embed_input_channels, embed_dim, embed_dim]))

    def forward(self, k: torch.Tensor, mask: torch.Tensor, **other) -> dict:
        # get all the conditioning information or defaults
        augmentinfo = other.get("augmentinfo", torch.zeros(k.shape[0], self.embed_augment_channels, device=k.device)).float()
        acceleration = other.get("acceleration", torch.ones(k.shape[0], device=k.device)).float()[:, None]
        accelerationinfo = self.embed_acceleration_map(acceleration)
        axis = other.get("axis", torch.zeros(k.shape[0], device=k.device)).float()[:, None]
        axisinfo = torch.cat((axis, 1 - axis), dim=-1)
        static_info = torch.cat((augmentinfo, axisinfo, accelerationinfo), dim=-1)

        x0 = torch.fft.ifftn(k, dim=(-2, -1), norm="ortho")
        x_rss = rss(x0)

        x = [x0]
        for t in range(self.T):
            iteration_info = self.embed_iter_map(t * torch.ones(k.shape[0], 1, device=k.device))
            z = self.embed_net(torch.cat((static_info, iteration_info), dim=-1))
            x_net = self.net(x[-1], z, x_rss=None if t else x_rss)
            x_dc = self.dc(k, x_net, mask, z)
            # x.append(x_net)
            x.append(x_dc)

        pred = rss(x[-1])
        return dict(prediction=pred, rss=x_rss, xs=x)


class Cascade(CineModel):
    def __init__(self, input_rss=True, lr=5e-4, weight_decay=1e-5, schedule=True, Nc: int = 10, T: int = 3, **kwargs):
        super().__init__()
        self.net = CascadeNN(input_rss=input_rss, Nc=Nc, T=T, **kwargs)
        self.EMANorm = EMA(alpha=0.9, max_iter=100)

    def forward(self, k: torch.Tensor, mask: torch.Tensor, **other) -> dict:
        k = k * self.EMANorm.ema_unbiased
        ret = self.net(k, mask, **other)
        if not self.training:
            unnorm = 1 / self.EMANorm.ema_unbiased
            ret["prediction"] = ret["prediction"] * unnorm
            ret["rss"] = ret["rss"] * unnorm
            ret["xs"] = [x * unnorm for x in ret["xs"]]
        return ret

    def training_step(self, batch, batch_idx):
        gt = batch.pop("gt")
        norm = self.EMANorm(1 / gt.std())

        gt = gt * norm
        ret = self(**batch)
        prediction, x_rss = ret["prediction"], ret["rss"]
        loss = torch.nn.functional.mse_loss(prediction, gt)
        rss_loss = torch.nn.functional.mse_loss(x_rss, gt)
        if "xs" in ret and self.trainer.global_step < 5000:
            gready_loss = sum([torch.nn.functional.mse_loss(rss(x), gt) for x in ret["xs"]])
        else:
            gready_loss = 0.0
        self.log("train_advantage", (rss_loss - loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("rss_loss", rss_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        loss = 0.8 * loss + 0.2 * gready_loss
        return loss
