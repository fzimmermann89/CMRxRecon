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


def scaling(kshape: int, scale: float = 1.0) -> torch.Tensor:
    x = torch.arange(-kshape // 2, kshape // 2)
    f = 1.0 + x.abs() ** scale
    f = f * (2 / f.mean())
    return f


class CNNWrapper(torch.nn.Module):
    """Wrapper for CNN that performs complex to real conversion, reshaping and complex to real conversion back
    Parameters
    ----------
    net: CNN
    checkpointing: checkpoint preprocessing and postprocessing
    """

    def __init__(self, net: torch.nn.Module, checkpointing=False):
        super().__init__()
        self.net = net
        self.chkpt = checkpointing

    @staticmethod
    def before(x: torch.Tensor, x_rss=None) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        xr = c2r(x)
        net_input = rearrange(xr, "b c z t x y r -> (b z) (r c) t x y")
        if x_rss is None:
            x_rss = rss(x)
        x_rss = rearrange(x_rss, "b z t x y -> (b z) t x y").unsqueeze(1)
        net_input = torch.cat((net_input, x_rss), 1)
        return net_input, xr

    @staticmethod
    def after(x_net: torch.Tensor, xr: torch.Tensor) -> torch.Tensor:
        x_net = rearrange(x_net, "(b z) (r c) t x y -> b c z t x y r", b=len(xr), r=2)
        return r2c(xr + x_net)

    def forward(self, x: torch.Tensor, *args, x_rss=None, **kwargs) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        run = lambda f, *args: torch.utils.checkpoint.checkpoint(f, *args, use_reentrant=False) if self.chkpt else f(*args)
        net_input, xr = run(self.before, x, x_rss)
        x_net, h = self.net(net_input, *args, **kwargs)
        return run(self.after, x_net, xr), h


class KCNNWrapper(torch.nn.Module):
    def __init__(self, net, k_scaling_factor: float = 0.0):
        super().__init__()
        self.net = net
        self.k_scaling_factor = k_scaling_factor

    def prepare_k0(self, k0):
        """prepare k0 for the network
        We need to do the following:
        - perform ifft along FS dim
        - fftshift along US dim
        - reshape to (b z t) (2*c) x y
        - apply scaling factor
        returns
        -------
        prepared k0, scaling factor
        """
        prepared = torch.fft.fftshift(torch.fft.ifft(k0, dim=-1, norm="ortho"), dim=-2)
        prepared = rearrange(c2r(prepared), "b c z t x y r-> (b z t) (r c) x y")
        if self.k_scaling_factor:
            f = scaling(k0.shape[-2], self.k_scaling_factor).to(k0.device)[:, None]
            prepared = prepared * f
        else:
            f = None
        return prepared, f

    @staticmethod
    def before(k, prepared_k0, f):
        """
        The input has to be in hybrid space, i.e.
        k space in US dim and image space in FS dim
        We shift the k space along the US dim to have DC in the center
        """
        k_netinput = rearrange(c2r(torch.fft.fftshift(k, dim=-2)), "b c z t x y r-> (b z t) (r c) x y")
        if f is not None:
            k_netinput = k_netinput * f
        k_netinput = torch.cat((k_netinput, prepared_k0), dim=1)
        return k_netinput

    @staticmethod
    def after(k_net: torch.Tensor, k: torch.Tensor, f) -> torch.Tensor:
        "unshift along US, reshape and residual connection"
        k_net = torch.fft.fftshift(
            rearrange(k_net, "(b z t) (r c) x y -> b c z t x y r", b=k.shape[0], z=k.shape[2], r=2), dim=-3
        )
        if f is not None:
            k_net = k_net * (1 / f[..., None])
        return k + r2c(k_net)

    def forward(self, k, *args, prepared_k0, f, **kwargs):
        """
        input is k space in US dim and image space in FS dim, output is the same
        """
        k_netinput = self.before(k, prepared_k0, f)
        k_net, h = self.net(k_netinput, *args, **kwargs)
        k_new = self.after(k_net, k, f)
        return k_new, h


class CascadeNet(torch.nn.Module):
    def __init__(
        self,
        unet_args=None,
        k_unet_args=None,
        Nc: int = 10,
        T: int = 2,
        embed_dim=192,
        lambda_init=1e-6,
        overwrite_k: bool = False,
        knet_init=0.01,
        xnet_init=1,
        k_scaling_factor: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.overwrite_k = overwrite_k
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

        net = Unet(channels_in=1 + 2 * Nc, channels_out=2 * Nc, emb_dim=embed_dim // 3, **unet_args)
        knet = Unet(channels_in=2 * (Nc + Nc), channels_out=2 * Nc, emb_dim=embed_dim // 3, **k_unet_args)

        with torch.no_grad():
            net.last[0].bias.zero_()
            knet.last[0].bias.zero_()
            knet.last[0].weight *= knet_init
            net.last[0].weight *= xnet_init

        self.net = CNNWrapper(net)
        self.knet = KCNNWrapper(knet, k_scaling_factor=k_scaling_factor)

        self.dc = torch.jit.script(
            MultiCoilDCLayer(Nc, lambda_init=lambda_init, embed_dim=embed_dim // 3, input_nn_k=(True, False))
        )

        self.embed_augment_channels = 6
        self.embed_axis_channels = 2
        self.embed_slice_channels = 1
        self.embed_acceleration_map = Mapper([2, 4, 6, 8, 10, 12])
        self.embed_iter_map = Mapper([0, 1, 2, 3, 4, 5])
        self.T = T
        embed_input_channels = (
            self.embed_augment_channels
            + self.embed_axis_channels
            + self.embed_acceleration_map.out_dim
            + self.embed_iter_map.out_dim
            + self.embed_slice_channels
        )

        self.embed_net = MLP([embed_input_channels, embed_dim, embed_dim])

    def forward(self, k: torch.Tensor, mask: torch.Tensor, **other) -> dict:
        # get all the conditioning information or defaults
        augmentinfo = other.get("augmentinfo", torch.zeros(k.shape[0], self.embed_augment_channels, device=k.device)).float()
        acceleration = other.get("acceleration", torch.ones(k.shape[0], device=k.device)).float()[:, None]
        accelerationinfo = self.embed_acceleration_map(acceleration)
        axis = other.get("axis", torch.zeros(k.shape[0], device=k.device)).float()[:, None]
        axisinfo = torch.cat((axis, 1 - axis), dim=-1)
        sliceinfo = other.get("slice", torch.zeros(k.shape[0], device=k.device)).float()[:, None] / 10
        static_info = torch.cat((augmentinfo, axisinfo, accelerationinfo, sliceinfo), dim=-1)

        x0 = torch.fft.ifftn(k, dim=(-2, -1), norm="ortho")
        x0_rss = rss(x0)

        xi = x0
        hk = None
        hx = None
        xs = []
        ks = []
        prepared_k0, f = self.knet.prepare_k0(k)
        for t in range(self.T):
            iteration_info = self.embed_iter_map(t * torch.ones(k.shape[0], 1, device=k.device))
            z = self.embed_net(torch.cat((static_info, iteration_info), dim=-1))
            zlambda, znet, zknet = torch.chunk(z, 3, dim=1)
            x_net, hx = self.net(xi, emb=znet, hin=hx, x_rss=None if t else x0_rss)
            k_x_net = torch.fft.fft(x_net, dim=-2, norm="ortho")  # k space along undersampled dim
            k_net, hk = self.knet(k_x_net, prepared_k0=prepared_k0, f=f, emb=zknet, hin=hk)
            x_dc = self.dc(k, k_net, mask, zlambda)
            xi = x_dc
            xs.append(x_net)
            ks.append(k_net)

        if self.overwrite_k:
            kp = torch.fft.fftn(xi, dim=(-2, -1), norm="ortho")
            kp = torch.where(mask > 0.5, k, kp)
            pred = rss(torch.fft.ifftn(kp, dim=(-2, -1), norm="ortho"))
        else:
            pred = rss(xi)

        return dict(prediction=pred, rss=x0_rss, xs=xs, ks=ks, x=xi)


class CascadeXKNew(CineModel):
    def __init__(
        self,
        lr=8e-4,
        weight_decay=1e-4,
        schedule=True,
        Nc: int = 10,
        T: int = 3,
        embed_dim=192,
        phase2_pct: float = 0.4,
        l2_weight: tuple[float, float] | float = (0.1, 0.5),
        ssim_weight: tuple[float, float] | float = (0, 0.4),
        greedy_weight: tuple[float, float] | float = (0, 0),
        l1_weight: tuple[float, float] | float = (0.3, 0.6),
        charbonnier_weight: tuple[float, float] | float = 0.0,
        max_weight: tuple[float, float] | float = (1e-4, 3e-3),
        l1_coilwise_weight: tuple[float, float] | float = (2.0, 0.0),
        l2_coilwise_weight: tuple[float, float] | float = (2.0, 0.5),
        l2_k_weight: tuple[float, float] | float = (0.2, 0.05),
        greedy_coilwise_weight: tuple[float, float] | float = (0.8, 0.1),
        ss_weight: tuple[float, float] | float = (0.5, 0.2),
        greedy_ss_weight: tuple[float, float] | float = (0.2, 0.0),
        k_loss_scaling_factor: float = 0.3,
        lambda_init: float = 0.5,
        overwrite_k: bool = False,
        knet_init=0.05,
        xnet_init=0.1,
        k_scaling_factor: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        unet_args = dict(
            dim=2.5,
            layer=3,
            filters=48,
            padding_mode="circular",
            residual="inner",
            latents=(False, "film_16", "film_32", "film_48"),
            norm="group16",
            feature_growth=(1, 2, 2, 1.334, 1, 1),
            activation="silu",
            change_filters_last=False,
            downsample_dimensions=((-1, -2), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3)),
            coordconv=((True, False), False),
            checkpointing=(True, False),
            reszero=0.1,
        )

        k_unet_args = dict(
            dim=2,
            filters=64,
            layer=2,
            feature_growth=(1.0, 1.5, 1.5),
            latents=(False, "film_16", "film_32"),
            conv_per_enc_block=3,
            conv_per_dec_block=3,
            downsample_dimensions=((-2,), (-1, -2)),
            change_filters_last=False,
            up_mode="linear_reduce",
            residual="inner",
            coordconv=True,
            norm="group16",
            activation="silu",
            checkpointing=(True, False),
        )

        unet_args.update(kwargs.pop("unet_args", {}))
        k_unet_args.update(kwargs.pop("k_unet_args", {}))

        self.save_hyperparameters({"unet_args": unet_args})
        self.save_hyperparameters({"k_unet_args": k_unet_args})

        self.net = CascadeNet(
            unet_args=unet_args,
            k_unet_args=k_unet_args,
            Nc=Nc,
            T=T,
            embed_dim=embed_dim,
            lambda_init=lambda_init,
            overwrite_k=overwrite_k,
            xnet_init=xnet_init,
            knet_init=knet_init,
            k_scaling_factor=k_scaling_factor,
            **kwargs,
        )
        self.EMANorm = EMA(alpha=0.9, max_iter=100)

    def forward(self, k: torch.Tensor, mask: torch.Tensor, **other) -> dict:
        k = k * torch.nan_to_num(self.EMANorm.ema_unbiased, nan=1000.0)
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
            "ss_weight",
            "greedy_ss_weight",
        ]

        ret = {}
        for k in keys:
            v = self.hparams[k]
            if isinstance(v, (tuple, list)):
                v = v[self.trainer.global_step >= self.trainer.max_steps * self.hparams.phase2_pct]
            ret[k] = v
        return ret

    def training_step_supervised(self, ret, gt, batch, norm, *args, **kwargs):
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

        l1w = weights["l1_coilwise_weight"]
        l2w = weights["l2_coilwise_weight"]
        l2gw = weights["greedy_coilwise_weight"]
        kw = weights["l2_k_weight"]
        if any([l1w, l2w, l2gw, kw]):
            if "kfull_ift_fs" in batch:
                kfull_ift_fs = batch.pop("kfull_ift_fs") * norm
            else:
                raise ValueError("kfull_ift_fs not in batch")

            if any([l1w, l2w, l2gw]):
                gt_coil_wise = c2r(torch.fft.ifft(kfull_ift_fs, dim=-2, norm="ortho"))
                if l1w and ("x" in ret):
                    loss = loss + l1w * F.l1_loss(c2r(ret["x"]), gt_coil_wise)
                if l2w and ("x" in ret):
                    loss = loss + l2w * F.mse_loss(c2r(ret["x"]), gt_coil_wise)
                if l2gw and ("xs" in ret):
                    greedy_x_cw = sum([F.mse_loss(c2r(x), gt_coil_wise) for x in ret["xs"]])
                    loss = loss + l2gw * greedy_x_cw

            if kw and ("ks" in ret):
                if scaling_factor := self.hparams.k_loss_scaling_factor:
                    f = scaling(kfull_ift_fs.shape[-2], scaling_factor).to(kfull_ift_fs.device)[:, None, None]
                    greedy_k = sum([F.mse_loss(f * c2r(kfull_ift_fs), f * c2r(k)) for k in ret["ks"]])
                else:
                    greedy_k = sum([F.mse_loss(c2r(kfull_ift_fs), c2r(k)) for k in ret["ks"]])
                loss = loss + kw * greedy_k

        self.log("train_advantage", (rss_loss - l2_loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", l2_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("rss_loss", rss_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def training_step_selfsupervised(self, ret, norm, batch, *args, **kwargs):
        skip = False
        if skip:
            return torch.tensor(0.0, device=ret["prediction"].device, requires_grad=True)
        weights = self.get_weights()
        mask = batch["mask_target"].unsqueeze(1)
        if "k_target_ift_fs" in batch:
            gt = batch.pop("k_target_ift_fs") * norm
        else:
            gt = torch.fft.ifft(batch.pop("k_target"), norm="ortho", dim=-1) * norm
        if scaling_factor := self.hparams.k_loss_scaling_factor:
            f = (torch.fft.fftshift(scaling(gt.shape[-2], scaling_factor))[:, None]).to(gt.device)
            gt = gt * f

        gt = c2r(torch.masked_select(gt, mask))
        loss = 0.0

        if w := weights["ss_weight"]:
            pred = torch.fft.fft2(ret["x"], axis=-2, norm="ortho")
            if scaling_factor:
                pred = pred * f
            pred = c2r(torch.masked_select(pred, mask))
            ss_loss = F.mse_loss(pred, gt)
            loss = loss + w * ss_loss

        if w := weights["greedy_ss_weight"] and "ks" in ret:
            ks = ret["ks"]
            if scaling_factor:
                ks = [k * f for k in ks]

            greedy_ss_loss = sum([F.mse_loss(c2r(torch.masked_select(k, mask)), gt) for k in ks])
            loss = loss + w * greedy_ss_loss

        self.log("train_ss_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        if "gt" in batch:
            gt = batch.pop("gt")
            norm = self.EMANorm(1 / gt.std())
            gt *= norm
            step = self.training_step_supervised
        else:
            norm = torch.nan_to_num(self.EMANorm.ema_unbiased, nan=1000.0)
            step = self.training_step_selfsupervised
            gt = None

        gc.collect()
        torch.cuda.synchronize()

        ret = self(**batch)

        loss = step(ret=ret, gt=gt, batch=batch, norm=norm)

        if loss > 0.5:
            loss = loss.detach().requires_grad_(True)

        return loss


class CascadeXKNewv2(CascadeXKNew):
    def __init__(
        self,
        lr=8e-4,
        weight_decay=1e-4,
        schedule=True,
        Nc: int = 10,
        T: int = 3,
        embed_dim=192,
        phase2_pct: float = 0.5,
        l2_weight: tuple[float, float] | float = (0.1, 0.4),
        ssim_weight: tuple[float, float] | float = (0, 0.5),
        greedy_weight: tuple[float, float] | float = (0, 0),
        l1_weight: tuple[float, float] | float = (0.3, 0.6),
        charbonnier_weight: tuple[float, float] | float = 0.0,
        max_weight: tuple[float, float] | float = (1e-4, 3e-3),
        l1_coilwise_weight: tuple[float, float] | float = (2.0, 0.0),
        l2_coilwise_weight: tuple[float, float] | float = (2.0, 0.5),
        l2_k_weight: tuple[float, float] | float = (0.2, 0.05),
        greedy_coilwise_weight: tuple[float, float] | float = (0.8, 0.1),
        ss_weight: tuple[float, float] | float = (2, 0.2),
        greedy_ss_weight: tuple[float, float] | float = (1.0, 0.0),
        k_loss_scaling_factor: float = 0.3,
        lambda_init: float = 0.5,
        overwrite_k: bool = False,
        knet_init=0.01,
        xnet_init=0.5,
        k_scaling_factor: float = 0.5,
        **kwargs,
    ):
        unet_args = dict(
            dim=2.5,
            layer=3,
            filters=48,
            padding_mode="zeros",
            residual="inner",
            latents=(False, "film_16", "film_32", "film_48"),
            norm="group16",
            feature_growth=(1, 2, 2, 1.334, 1, 1),
            activation="silu",
            change_filters_last=False,
            downsample_dimensions=((-1, -2), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3)),
            coordconv=((True, False), False),
            checkpointing=(True, False),
            reszero=0.05,
        )

        k_unet_args = dict(
            dim=2,
            filters=64,
            layer=2,
            feature_growth=(1.0, 1.5, 1.5),
            latents=(False, "film_16", "film_32"),
            conv_per_enc_block=3,
            conv_per_dec_block=3,
            downsample_dimensions=((-2,), (-1, -2)),
            change_filters_last=False,
            up_mode="linear_reduce",
            residual="inner",
            coordconv=True,
            norm="group16",
            activation="silu",
            checkpointing=(True, False),
            reszero=0.5,
        )

        unet_args.update(kwargs.pop("unet_args", {}))
        k_unet_args.update(kwargs.pop("k_unet_args", {}))
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            schedule=schedule,
            Nc=Nc,
            T=T,
            embed_dim=embed_dim,
            phase2_pct=phase2_pct,
            l2_weight=l2_weight,
            ssim_weight=ssim_weight,
            greedy_weight=greedy_weight,
            l1_weight=l1_weight,
            charbonnier_weight=charbonnier_weight,
            max_weight=max_weight,
            l1_coilwise_weight=l1_coilwise_weight,
            l2_coilwise_weight=l2_coilwise_weight,
            l2_k_weight=l2_k_weight,
            greedy_coilwise_weight=greedy_coilwise_weight,
            lambda_init=lambda_init,
            overwrite_k=overwrite_k,
            unet_args=unet_args,
            k_unet_args=k_unet_args,
            xnet_init=xnet_init,
            knet_init=knet_init,
            ss_weight=ss_weight,
            greedy_ss_weight=greedy_ss_weight,
            k_loss_scaling_factor=k_loss_scaling_factor,
            k_scaling_factor=k_scaling_factor,
            **kwargs,
        )


class CascadeXKNewv3(CascadeXKNew):
    # closer to v5 old
    def on_before_optimizer_step(self, optimizer):
        # log gradient norm for last layers
        self.log(
            "grad_norm_x_last",
            self.net.net.net.last[0].weight.grad.norm().item(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "grad_norm_k_last",
            self.net.knet.net.last[0].weight.grad.norm().item(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        # log lr
        self.log("lr", optimizer.param_groups[0]["lr"], on_step=True, on_epoch=True, prog_bar=False, logger=True)

    def __init__(
        self,
        lr=8e-4,
        weight_decay=1e-3,
        schedule=True,
        Nc: int = 10,
        T: int = 3,
        embed_dim=192,
        phase2_pct: float = 0.5,
        l2_weight: tuple[float, float] | float = (0.1, 0.5),
        ssim_weight: tuple[float, float] | float = (0, 0.4),
        greedy_weight: tuple[float, float] | float = (0, 0),
        l1_weight: tuple[float, float] | float = (0.3, 0.6),
        charbonnier_weight: tuple[float, float] | float = 0.0,
        max_weight: tuple[float, float] | float = (1e-4, 1e-3),
        l1_coilwise_weight: tuple[float, float] | float = (2.0, 0.0),
        l2_coilwise_weight: tuple[float, float] | float = (2.0, 0.5),
        l2_k_weight: tuple[float, float] | float = (0.2, 0.05),
        greedy_coilwise_weight: tuple[float, float] | float = (0.8, 0.1),
        lambda_init: float = 0.5,
        overwrite_k: bool = False,
        knet_init=0.01,
        xnet_init=1,
        ss_weight: tuple[float, float] | float = (1.0, 0.2),
        greedy_ss_weight: tuple[float, float] | float = (0.2, 0.0),
        k_loss_scaling_factor: float = 0.3,
        k_scaling_factor: float = 0.3,
        **kwargs,
    ):
        unet_args = dict(
            dim=2.5,
            layer=3,
            filters=48,
            padding_mode="zeros",
            residual="inner",
            latents=(False, "film_16", "film_24", "film_32"),
            norm="group16",
            feature_growth=(1, 1.667, 2, 1.6, 1, 1),
            activation="silu",
            change_filters_last=False,
            downsample_dimensions=((-1, -2), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3)),
            up_mode="linear",
            coordconv=(True, False),
            checkpointing=(True, False),
        )

        k_unet_args = dict(
            dim=2,
            filters=64,
            layer=2,
            padding_mode="zeros",
            feature_growth=(1.0, 1.25, 1.6),
            latents=(False, "film_16", "film_24"),
            conv_per_enc_block=3,
            conv_per_dec_block=2,
            downsample_dimensions=((-2,), (-1, -2)),
            change_filters_last=False,
            up_mode="linear_reduce",
            residual="inner",
            coordconv=True,
            reszero=False,
            norm="group16",
            activation="silu",
            checkpointing=(True, False),
        )

        unet_args.update(kwargs.pop("unet_args", {}))
        k_unet_args.update(kwargs.pop("k_unet_args", {}))
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            schedule=schedule,
            Nc=Nc,
            T=T,
            embed_dim=embed_dim,
            phase2_pct=phase2_pct,
            l2_weight=l2_weight,
            ssim_weight=ssim_weight,
            greedy_weight=greedy_weight,
            l1_weight=l1_weight,
            charbonnier_weight=charbonnier_weight,
            max_weight=max_weight,
            l1_coilwise_weight=l1_coilwise_weight,
            l2_coilwise_weight=l2_coilwise_weight,
            l2_k_weight=l2_k_weight,
            greedy_coilwise_weight=greedy_coilwise_weight,
            lambda_init=lambda_init,
            overwrite_k=overwrite_k,
            unet_args=unet_args,
            k_unet_args=k_unet_args,
            xnet_init=xnet_init,
            knet_init=knet_init,
            ss_weight=ss_weight,
            greedy_ss_weight=greedy_ss_weight,
            k_loss_scaling_factor=k_loss_scaling_factor,
            k_scaling_factor=k_scaling_factor,
            **kwargs,
        )
