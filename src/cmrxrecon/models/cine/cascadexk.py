from typing import Any, Mapping
import torch
from einops import rearrange
from cmrxrecon.nets.unet import Unet
from cmrxrecon.nets.mlp import MLP
from typing import *
import torch._dynamo
from . import CineModel
from cmrxrecon.models.utils.crop import crops_by_threshold, uncrop
from cmrxrecon.models.utils.ema import EMA
from cmrxrecon.models.utils import rss
from cmrxrecon.models.utils.mapper import Mapper
from cmrxrecon.models.utils.multicoildc import MultiCoilDCLayer
import gc
from cmrxrecon.models.utils.ssim import ssim
import torch.nn.functional as F
from torch import view_as_real as c2r, view_as_complex as r2c


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

    def forward(self, x: torch.Tensor, *args, x_rss=None, **kwargs) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.crop_threshold > 0:
            crops = crops_by_threshold(x.detach(), [self.crop_threshold])
            xc = x[crops]
            if x_rss is not None:
                x_rss_c = x_rss[(crops[0], *crops[2:])]
        else:
            xc = x
            x_rss_c = x_rss
            crops = None
        net_input = rearrange(c2r(xc), "b c z t x y r -> (b z) (r c) t x y")
        if self.include_rss:
            if x_rss is None:
                x_rss_c = rss(xc)
            x_rss_c = rearrange(x_rss_c, "b z t x y -> (b z) t x y").unsqueeze(1)
            net_input = torch.cat((net_input, x_rss_c), 1)
        x_net, h = self.net(net_input, *args, **kwargs)
        x_net = rearrange(x_net, "(b z) (r c) t x y -> b c z t x y r", b=len(x), r=2).contiguous()
        x_net = r2c(x_net)
        x_net = uncrop(x_net, x.shape, crops)
        return x + x_net, h


class KCNNWrapper(torch.nn.Module):
    def __init__(self, net, input_k0: bool = True):
        super().__init__()
        self.net = net
        self.input_k0 = input_k0

    def prepare_k0(self, k0):
        if self.input_k0:
            return torch.fft.fftshift(torch.fft.ifft(k0, dim=-1, norm="ortho"), dim=-2)

    def forward(self, k, *args, prepared_k0=None, **kwargs):
        if prepared_k0 is not None:
            k_netinput = torch.concat((torch.fft.fftshift(k, dim=-2), prepared_k0), 1)
        else:
            k_netinput = torch.fft.fftshift(k, dim=-2)
        k_netinput = rearrange(c2r(k_netinput), "b c z t x y r-> (b z t) (r c) x y")
        k_net, h = self.net(k_netinput, *args, **kwargs)
        k_net = torch.fft.fftshift(
            rearrange(k_net, "(b z t) (r c) x y -> b c z t x y r", b=k.shape[0], z=k.shape[2], r=2), dim=-3
        )
        k_net = r2c(k_net)
        return k + k_net, h


class CascadeNet(torch.nn.Module):
    def __init__(
        self,
        unet_args=None,
        k_unet_args=None,
        input_rss=True,
        input_k0=True,
        Nc: int = 10,
        T: int = 2,
        embed_dim=192,
        crop_threshold: float = 0.005,
        lambda_init=1e-6,
        **kwargs,
    ):
        super().__init__()
        self.input_rss = input_rss
        embed_dim: int = 192
        if unet_args is None:
            unet_args = dict(
                dim=2,
                layer=3,
                filters=32,
                change_filters_last=False,
                feature_growth=(1, 2, 1, 1, 1, 1),
            )

        if k_unet_args is None:
            k_unet_args = dict(
                dim=2,
                layer=1,
                filters=32,
                change_filters_last=False,
                feature_growth=(1, 1.5, 1, 1, 1, 1),
            )

        net = Unet(channels_in=input_rss + 2 * Nc, channels_out=2 * Nc, emb_dim=embed_dim // 3, **unet_args)
        knet = Unet(channels_in=2 * (Nc + Nc * input_k0), channels_out=2 * Nc, emb_dim=embed_dim // 3, **k_unet_args)

        with torch.no_grad():
            net.last[0].bias.zero_()
            knet.last[0].bias.zero_()
            knet.last[0].weight *= 0.01

        self.net = CNNWrapper(net, include_rss=input_rss, crop_threshold=crop_threshold)
        self.knet = KCNNWrapper(knet, input_k0=input_k0)

        self.dc = torch.jit.script(
            MultiCoilDCLayer(Nc, lambda_init=lambda_init, embed_dim=embed_dim // 3, input_nn_k=(True, False))
        )

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
        self.input_k0 = input_k0

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

        xi = x0
        hk = None
        hx = None
        xs = []
        ks = []
        prepared_k0 = self.knet.prepare_k0(k)
        for t in range(self.T):
            iteration_info = self.embed_iter_map(t * torch.ones(k.shape[0], 1, device=k.device))
            z = self.embed_net(torch.cat((static_info, iteration_info), dim=-1))
            zlambda, znet, zknet = torch.chunk(z, 3, dim=1)
            x_net, hx = self.net(xi, emb=znet, hin=hx, x_rss=None if t else x_rss)
            k_x_net = torch.fft.fft(x_net, dim=-2, norm="ortho")  # k space along undersampled dim
            k_net, hk = self.knet(k_x_net, prepared_k0=prepared_k0, emb=zknet, hin=hk)
            x_dc = self.dc(k, k_net, mask, zlambda)
            xi = x_dc
            xs.append(x_net)
            ks.append(k_net)

        pred = rss(xi)
        return dict(prediction=pred, rss=x_rss, xs=xs, ks=ks, x=xi)


class CascadeXK(CineModel):
    def __init__(
        self,
        input_rss=True,
        input_k0=True,
        lr=1e-3,
        weight_decay=1e-4,
        schedule=True,
        Nc: int = 10,
        T: int = 3,
        embed_dim=192,
        crop_threshold: float = 0.005,
        phase2_pct: float = 0.5,
        l2_weight: tuple[float, float] | float = (0, 0.2),
        ssim_weight: tuple[float, float] | float = 0.3,
        greedy_weight: tuple[float, float] | float = (0.3, 0.05),
        l1_weight: tuple[float, float] | float = 0.6,
        charbonnier_weight: tuple[float, float] | float = 0.0,
        max_weight: tuple[float, float] | float = (1e-3, 1e-2),
        l1_coilwise_weight: tuple[float, float] | float = 0.0,
        l2_coilwise_weight: tuple[float, float] | float = 0.0,
        l2_k_weight: tuple[float, float] | float = 0.0,
        greedy_coilwise_weight: tuple[float, float] | float = 0.0,
        lambda_init: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        torch._dynamo.config.suppress_errors = True
        unet_args = dict(
            dim=2.5,
            layer=3,
            filters=48,
            padding_mode="zeros",
            residual="inner",
            latents=(False, "film_16", "film_16", "film_16"),
            norm="group16",
            feature_growth=(1, 2, 1.5, 1.34, 1, 1),
            activation="leakyrelu",
            change_filters_last=False,
            downsample_dimensions=((-1, -2), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3)),
            coordconv=((True, False), False),
        )

        k_unet_args = dict(
            dim=2,
            filters=64,
            layer=1,
            feature_growth=(1.0, 1.5, 1.0),
            latents=(False, "film_16"),
            conv_per_enc_block=3,
            conv_per_dec_block=3,
            downsample_dimensions=((-2,),),
            change_filters_last=False,
            up_mode="linear_reduce",
            residual="inner",
            coordconv=True,
            reszero=False,
            norm="group16",
        )

        unet_args.update(kwargs.pop("unet_args", {}))
        k_unet_args.update(kwargs.pop("k_unet_args", {}))

        self.save_hyperparameters({"unet_args": unet_args})
        self.save_hyperparameters({"k_unet_args": k_unet_args})

        self.net = CascadeNet(
            unet_args=unet_args,
            k_unet_args=k_unet_args,
            input_rss=input_rss,
            input_k0=input_k0,
            Nc=Nc,
            T=T,
            embed_dim=embed_dim,
            crop_threshold=crop_threshold,
            lambda_init=lambda_init,
            **kwargs,
        )
        self.EMANorm = EMA(alpha=0.9, max_iter=100)

    def forward(self, k: torch.Tensor, mask: torch.Tensor, **other) -> dict:
        k = k * self.EMANorm.ema_unbiased
        ret = self.net(k, mask, **other)
        if not self.training:
            unnorm = 1 / self.EMANorm.ema_unbiased
            ret = dict(prediction=ret["prediction"] * unnorm, rss=ret["rss"] * unnorm)
        return ret

    def get_weights(self):
        keys = [
            "l2_weight",
            "ssim_weight",
            "greedy_weight",
            "l1_weight",
            "charbonnier_weight",
            "max_weight",
            "l1_coilwise_weight",
            "l2_coilwise_weight",
            "greedy_coilwise_weight",
            "l2_k_weight",
        ]

        ret = {}
        for k in keys:
            v = self.hparams[k]
            if isinstance(v, tuple):
                v = v[self.trainer.global_step >= self.trainer.max_steps * self.hparams.phase2_pct]
            ret[k] = v
        return ret

    def training_step(self, batch, batch_idx):
        gt = batch.pop("gt")
        norm = self.EMANorm(1 / gt.std())
        gt *= norm

        gc.collect()
        torch.cuda.synchronize()
        ret = self(**batch)
        prediction, x_rss = ret["prediction"], ret["rss"]

        weights = self.get_weights()
        loss = 0.0
        rss_loss = F.mse_loss(x_rss, gt)
        l2_loss = F.mse_loss(prediction, gt)
        ssim_value = ssim(gt, prediction)

        if w := weights["l2_weight"]:
            loss = loss + w * l2_loss
        if w := weights["ssim_weight"]:
            loss = loss + w * (1 - ssim_value)
        if (w := weights["greedy_weight"]) and "xs" in ret:
            greedy_xrss = sum([F.mse_loss(rss(x), gt) for x in ret["xs"]])
            loss = loss + w * greedy_xrss
        if w := weights["l1_weight"]:
            l1_loss = F.l1_loss(prediction, gt)
            loss = loss + w * l1_loss
        if w := weights["max_weight"]:
            max_penalty = F.mse_loss(gt.amax(dim=(-1, -2)), prediction.amax(dim=(-1, -2)))
            loss = loss + w * max_penalty
        if w := weights["charbonnier_weight"]:
            charbonnier_loss = (F.mse_loss(prediction, gt, reduction="none") + 1e-3).sqrt().mean()
            loss = loss + w * charbonnier_loss
        if ("kfull" in batch) and (
            (l1w := weights["l1_coilwise_weight"])
            or (l2w := weights["l2_coilwise_weight"])
            or (l2gw := weights["greedy_coilwise_weight"])
            or (kw := weights["l2_k_weight"])
        ):
            kfull = batch.pop("kfull") * norm
            if l1w or l2w or l2gw:
                gt_coil_wise = c2r(torch.fft.ifft2(kfull, norm="ortho"))
                if l1w and ("xi" in ret):
                    loss = loss + l1w * F.l1_loss(c2r(ret["x"]), gt_coil_wise)
                if l2w and ("xi" in ret):
                    loss = loss + l2w * F.mse_loss(c2r(ret["x"]), gt_coil_wise)
                if l2gw and ("xs" in ret):
                    greedy_x_cw = sum([F.mse_loss(c2r(x), gt_coil_wise) for x in ret["xs"]])
                    loss = loss + l2gw * greedy_x_cw
            if kw and ("ks" in ret):
                greedy_k = sum([F.mse_loss(c2r(kfull), c2r(k)) for k in ret["ks"]])
                loss = loss + kw * greedy_k

        self.log("train_advantage", (rss_loss - l2_loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", l2_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("rss_loss", rss_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss
