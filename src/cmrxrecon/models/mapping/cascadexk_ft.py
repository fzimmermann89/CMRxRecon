from cmath import phase
import re
from typing import Any, Mapping
import torch
from einops import rearrange
from .base import MappingModel
from cmrxrecon.nets.unet import Unet
from cmrxrecon.nets.mlp import MLP
from cmrxrecon.models.utils import rss
from typing import *
import torch._dynamo
from cmrxrecon.models.utils.ema import EMA
from cmrxrecon.models.utils import rss
from cmrxrecon.models.utils.mapper import Mapper
from cmrxrecon.models.utils.multicoildc import MultiCoilDCLayer
import gc
from cmrxrecon.models.utils.ssim import ssim
import torch.nn.functional as F
from torch import conv2d, view_as_real as c2r, view_as_complex as r2c

from cmrxrecon.models.cine.cascadexk_new import *


def split_state_dict(state_dict, prefixes):
    splits = [{k.removeprefix(prefix + "."): v for k, v in state_dict.items() if k.startswith(prefix)} for prefix in prefixes]
    rest = {k: v for k, v in state_dict.items() if not any(k.startswith(prefix) for prefix in prefixes)}
    return splits, rest


class MappingCascadeNet(CascadeNet):
    def __init__(
        self,
        unet_args,
        k_unet_args,
        Nc: int = 10,
        T: int = 2,
        embed_dim=192,
        lambda_init=1e-6,
        overwrite_k: bool = False,
        knet_init=0.01,
        xnet_init=1,
        k_scaling_factor: float = 0.0,
        learned_norm: bool = True,
        learned_norm_global_scale: bool = False,
        learned_norm_part_inv: bool = False,
        learned_norm_local_scale: bool = True,
        learned_norm_emb_dim: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.overwrite_k = overwrite_k
        net = Unet(channels_in=1 + 2 * Nc, channels_out=2 * Nc, emb_dim=embed_dim // 3, **unet_args)
        knet = Unet(channels_in=2 * (Nc + Nc), channels_out=2 * Nc, emb_dim=embed_dim // 3, **k_unet_args)

        with torch.no_grad():
            net.last[0].bias.zero_()
            knet.last[0].bias.zero_()
            knet.last[0].weight *= knet_init
            net.last[0].weight *= xnet_init

        self.net = CNNWrapper_Mapping(
            net,
            learned_norm=learned_norm,
            learned_norm_global_scale=learned_norm_global_scale,
            learned_norm_part_inv=learned_norm_part_inv,
            learned_norm_local_scale=learned_norm_local_scale,
            learned_norm_emb_dim=learned_norm_emb_dim,
        )
        self.knet = KCNNWrapper(knet, k_scaling_factor=k_scaling_factor)

        self.dc = torch.jit.script(
            MultiCoilDCLayer(Nc, lambda_init=lambda_init, embed_dim=embed_dim // 3, input_nn_k=(True, False))
        )

        self.embed_augment_channels = 6
        self.embed_job_channels = 2
        self.embed_slice_channels = 1
        self.embed_acceleration_map = Mapper([2, 4, 6, 8, 10, 12])
        self.embed_iter_map = Mapper([0, 1, 2, 3, 4, 5])
        self.T = T
        self.embed_times_channels = 12
        self.embed_job_channels = 2

        embed_input_channels = (
            self.embed_augment_channels
            + self.embed_job_channels
            + self.embed_acceleration_map.out_dim
            + self.embed_iter_map.out_dim
            + self.embed_slice_channels
            + self.embed_times_channels
        )

        self.embed_net = MLP([embed_input_channels, embed_dim, embed_dim])

    def conditioning_info(self, other, batchsize, device) -> dict:
        augmentinfo = other.get("augmentinfo", torch.zeros(batchsize, self.embed_augment_channels, device=device)).float()
        acceleration = other.get("acceleration", torch.ones(batchsize, device=device)).float()[:, None]
        accelerationinfo = self.embed_acceleration_map(acceleration)
        job = other.get("job", torch.zeros(batchsize, device=device)).float()[:, None]
        jobinfo = torch.cat((job, 1 - job), dim=-1)
        sliceinfo = other.get("slice", torch.zeros(batchsize, device=device)).float()[:, None] / 10
        if "times" in other:
            times = other.get("times")[:, 0, :].float()
            if job:  # T1
                times = (times - 1000) / 2000
                times = torch.cat((times, torch.zeros(batchsize, 12 - times.shape[1], device=device)), dim=-1)
            else:
                times = (times - 30) / 40
                times = torch.cat((torch.zeros(batchsize, 12 - times.shape[1], device=device), times), dim=-1)
        else:
            times = torch.zeros(batchsize, 12, device=device)

        static_info = torch.cat((augmentinfo, jobinfo, accelerationinfo, sliceinfo, times), dim=-1)
        return static_info


class CascadeXKNew_Mapping(MappingModel):
    def configure_optimizers(self):
        retrain = (
            "net.net.net.encoder",
            "net.net.net.decoder",
            "net.net.net.last",
            "knet.net.net.encoder",
            "knet.net.net.decoder",
            "knet.net.last",
            "project",
            "lambda_proj",
        )
        newtrain = ("embed_net", "time_normalizer")

        param_ft = []
        param_retrain = []
        param_newtrain = []
        for name, param in self.named_parameters():
            if any([x in name for x in retrain]):
                param_retrain.append(param)
            elif any([x in name for x in newtrain]):
                param_newtrain.append(param)
            else:
                param_ft.append(param)
        param = [
            dict(params=param_retrain, lr=self.hparams.lr_rt),
            dict(params=param_ft, lr=self.hparams.lr_ft),
            dict(params=param_newtrain, lr=self.hparams.lr),
        ]
        optimizer = torch.optim.AdamW(param, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[
                self.hparams.lr_rt,
                self.hparams.lr_ft,
                self.hparams.lr,
            ],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.2,
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=10,
            final_div_factor=30,
            verbose=False,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step"},
        ]

    def __init__(
        self,
        lr_ft=3e-5,
        lr_rt=3e-4,
        lr=8e-4,
        weight_decay=1e-4,
        Nc: int = 10,
        T: int = 3,
        embed_dim=192,
        overwrite_k: bool = False,
        k_scaling_factor: float = 0.3,
        phase2_pct: float = 0.6,
        mapping_l2_weight: tuple[float, float] | float = (0.2, 0.6),
        mapping_ssim_weight: tuple[float, float] | float = (0.5, 0.8),
        mapping_l1_weight: tuple[float, float] | float = (1.0, 0.5),
        mapping_max_weight: tuple[float, float] | float = (1e-3, 3e-3),
        mapping_non_roi_weight: tuple[float, float] | float = (0.75, 0.5),
        cine_ckpt_path: str = "",
        learned_norm: bool = True,
        learned_norm_global_scale: bool = False,
        learned_norm_part_inv: bool = False,
        learned_norm_local_scale: bool = True,
        learned_norm_emb: bool = False,
        **kwargs,
    ):
        mapping_ft_parameters = dict(
            mapping_l2_weight=mapping_l2_weight,
            mapping_ssim_weight=mapping_ssim_weight,
            mapping_l1_weight=mapping_l1_weight,
            mapping_max_weight=mapping_max_weight,
            mapping_non_roi_weight=mapping_non_roi_weight,
            phase2_pct=phase2_pct,
        )
        self.save_hyperparameters(mapping_ft_parameters)

        super().__init__()

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
            downsample_dimensions=((-1, -2), (-1, -2), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3)),
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

        self.save_hyperparameters({"unet_args": unet_args})
        self.save_hyperparameters({"k_unet_args": k_unet_args})

        self.net = MappingCascadeNet(
            unet_args=unet_args,
            k_unet_args=k_unet_args,
            Nc=Nc,
            T=T,
            embed_dim=embed_dim,
            lambda_init=0.5,
            overwrite_k=overwrite_k,
            k_scaling_factor=k_scaling_factor,
            learned_norm=learned_norm,
            learned_norm_global_scale=learned_norm_global_scale,
            learned_norm_part_inv=learned_norm_part_inv,
            learned_norm_local_scale=learned_norm_local_scale,
            learned_norm_emb_dim=embed_dim // 3 if learned_norm_emb else 0,
            **kwargs,
        )
        self.EMANorm = EMA(alpha=0.9, max_iter=100)

        if cine_ckpt_path != "":
            print("Loading Cine checkpoint")
            ckpt = torch.load(cine_ckpt_path, map_location="cpu")
            (knet, xnet, dc, embed, ema), rest = split_state_dict(
                ckpt["state_dict"], ["net.knet", "net.net.net", "net.dc", "net.embed_net", "EMANorm"]
            )
            self.EMANorm.load_state_dict(ema)
            self.EMANorm.iter = 100
            self.EMANorm.max_iter = 200

            self.net.knet.load_state_dict(knet)
            self.net.net.net.load_state_dict(xnet)
            self.net.dc.load_state_dict(dc)

            embed_first_w = embed.pop("net.0.weight")
            std = embed_first_w.std() * 0.25
            embed_first_w[:, 6:8].normal_(std=std)  # overwrite job
            embed_first_t = torch.randn(embed_first_w.shape[0], 12, device=embed_first_w.device) * std
            embed_first_w = torch.cat((embed_first_w, embed_first_t), dim=-1)  # extend for times
            embed["net.0.weight"] = embed_first_w
            self.net.embed_net.load_state_dict(embed)

    def get_weights(self):
        keys = [
            "mapping_l2_weight",
            "mapping_ssim_weight",
            "mapping_l1_weight",
            "mapping_max_weight",
            "mapping_non_roi_weight",
        ]

        ret = {}
        for k in keys:
            v = self.hparams[k]
            if isinstance(v, (tuple, list)):
                v = v[self.trainer.global_step >= self.trainer.max_steps * self.hparams.phase2_pct]
            ret[k] = v
        return ret

    def training_step_supervised(self, ret, gt, batch, norm, *args, **kwargs):
        prediction, x_rss, roi = ret["prediction"], ret["rss"], batch["roi_dilated"]
        prediction_roi = prediction * roi
        prediction_rest = prediction - prediction_roi
        gt_roi = gt * roi
        gt_rest = gt - gt_roi
        weights = self.get_weights()

        loss = 0.0
        rss_loss_roi = F.mse_loss(x_rss * roi, gt_roi)
        rss_loss_rest = F.mse_loss(x_rss * (~roi), gt_rest)
        l2_loss_roi = F.mse_loss(prediction_roi, gt_roi)
        l2_loss_rest = F.mse_loss(prediction_rest, gt_rest)
        ssim_value_roi = ssim(gt_roi, prediction_roi)
        mapping_non_roi_weight = weights["mapping_non_roi_weight"]

        if w := weights["mapping_l2_weight"]:
            loss = loss + w * (l2_loss_roi + mapping_non_roi_weight * l2_loss_rest)
        if w := weights["mapping_ssim_weight"]:
            loss = loss + w * ((1 - ssim_value_roi) + mapping_non_roi_weight * (1 - ssim(gt_rest, prediction_rest)))
        if w := weights["mapping_l1_weight"]:
            loss = loss + w * (F.l1_loss(prediction_roi, gt_roi) + mapping_non_roi_weight * F.l1_loss(prediction_rest, gt_rest))
        if w := weights["mapping_max_weight"]:
            max_penalty_roi = F.mse_loss(gt_roi.amax(dim=(-1, -2)), prediction_roi.amax(dim=(-1, -2)))
            max_penalty_rest = F.mse_loss(gt_rest.amax(dim=(-1, -2)), prediction_rest.amax(dim=(-1, -2)))
            loss = loss + w * (max_penalty_roi + mapping_non_roi_weight * max_penalty_rest)
        if (rss_loss_roi := rss_loss_roi.item()) > 1e-6:
            self.log(
                "train_advantage_roi",
                ((rss_loss_roi - l2_loss_roi) / rss_loss_roi),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        if (rss_loss_rest := rss_loss_rest.item()) > 1e-6:
            self.log(
                "train_advantage_rest",
                ((rss_loss_rest - l2_loss_rest) / rss_loss_rest),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        with torch.no_grad():
            rss_loss = F.mse_loss(x_rss, gt)
            l2_loss = F.mse_loss(prediction, gt)
            self.log(
                "train_advantage",
                (rss_loss - l2_loss) / rss_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_ssim_roi", ssim_value_roi, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        if "gt" in batch:
            gt = batch.pop("gt")
            norm = self.EMANorm(1 / gt.std())
            gt *= norm
            step = self.training_step_supervised
        else:
            return

        gc.collect()
        torch.cuda.synchronize()

        ret = self(**batch)

        loss = step(ret=ret, gt=gt, batch=batch, norm=norm)

        if loss > 0.5 or torch.isnan(loss) or torch.isinf(loss):
            loss = loss.detach().requires_grad_(True)

        return loss

    def forward(self, k: torch.Tensor, mask: torch.Tensor, **other) -> dict:
        k = k * torch.nan_to_num(self.EMANorm.ema_unbiased, nan=1000.0)
        ret = self.net(k, mask, **other)
        if not self.training:
            unnorm = 1 / self.EMANorm.ema_unbiased
            ret = dict(prediction=ret["prediction"] * unnorm, rss=ret["rss"] * unnorm)
        return ret


class CNNWrapper_Mapping(torch.nn.Module):
    """Wrapper for CNN that performs complex to real conversion, reshaping and complex to real conversion back
    Parameters
    ----------
    net: CNN
    checkpointing: checkpoint preprocessing and postprocessing
    """

    def __init__(
        self,
        net: torch.nn.Module,
        checkpointing=False,
        learned_norm=True,
        learned_norm_global_scale=False,
        learned_norm_part_inv=False,
        learned_norm_local_scale=True,
        learned_norm_emb_dim=64,
    ):
        super().__init__()
        self.net = net
        self.chkpt = checkpointing
        if learned_norm:
            self.time_normalizer = torch.nn.ModuleList(
                [
                    MappingNormalizer(
                        t,
                        vc=4,
                        Nc=10,
                        global_scale=learned_norm_global_scale,
                        part_inv=learned_norm_part_inv,
                        local_scale=learned_norm_local_scale,
                        emb_dim=learned_norm_emb_dim,
                    )
                    for t in (9, 3)
                ]
            )
        else:
            self.time_normalizer = None

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
    def after(x_net: torch.Tensor, xr: torch.Tensor, normfactor: torch.Tensor) -> torch.Tensor:
        x_net = rearrange(x_net, "(b z) (r c) t x y -> b c z t x y r", b=len(xr), r=2)
        x_net = x_net * torch.view_as_real(normfactor)
        return r2c(xr + x_net)

    def forward(self, x: torch.Tensor, *args, emb, x_rss=None, **kwargs) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.time_normalizer is not None:
            for norm in self.time_normalizer:
                if norm.times == x.shape[3]:
                    break
            else:
                raise ValueError(f"Cannot find normalizer for times {x.shape[3]}")
            x, normfactor = norm(x, emb)
        else:
            normfactor = 1.0

        run = lambda f, *args: torch.utils.checkpoint.checkpoint(f, *args, use_reentrant=False) if self.chkpt else f(*args)
        net_input, xr = run(self.before, x, x_rss)
        x_net, h = self.net(net_input, *args, emb=emb, **kwargs)
        return run(self.after, x_net, xr, normfactor), h


class MappingNormalizer(torch.nn.Module):
    def __init__(self, times, vc=4, Nc=10, global_scale=False, part_inv=False, emb_dim=64, local_scale=True):
        super().__init__()
        self.times = times
        self.global_scale = global_scale
        self.part_inv = part_inv
        self.local_scale = local_scale
        self.emb_dim = emb_dim

        self.comp = torch.nn.Conv2d(Nc, vc, kernel_size=(1, 2))

        if local_scale:
            self.net = torch.nn.Sequential(
                torch.nn.Conv3d((vc + 1) * times, 64, kernel_size=(1, 3, 3), padding="same"),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding="same"),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Conv3d(128, times * (2 + 2 * part_inv), kernel_size=(1, 1, 1)),
            )
            with torch.no_grad():
                self.net[-1].weight *= 0.1
                self.net[-1].bias.zero_()
        else:
            self.net = lambda x, *args: torch.zeros(x.shape[0], times * (2 + 2 * part_inv) + emb_dim, 1, 1, 1, device=x.device)

        if global_scale:
            self.global_net = torch.nn.Sequential(
                torch.nn.Linear((vc + 1) * 4 + emb_dim, 64),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(64, 64),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(64, 2),
            )
            with torch.no_grad():
                self.global_net[-1].weight *= 0.1
                self.global_net[-1].bias.zero_()

    # @torch.jit.script
    # def get_scale(raw):
    #     amp, phase = torch.chunk(torch.tanh(raw), 2, dim=1)
    #     scale = torch.exp(2 * amp + phase * 3.14j)
    #     scale = scale.swapaxes(1, 2).unsqueeze(1)
    #     return scale

    @torch.jit.script
    def get_scale(raw, part_inv: bool = False, global_raw: Optional[torch.Tensor] = None):
        if part_inv:
            amp, phase, ampkeep, phasekeep = torch.chunk(torch.tanh(raw), 4, dim=1)
            scalekeep = 2 * ampkeep + phasekeep * 3.14j
            scalekeep = scalekeep.swapaxes(1, 2).unsqueeze(1)
            scale = 2 * amp + phase * 3.14j
            scale = scale.swapaxes(1, 2).unsqueeze(1)
            scalefw = scale + scalekeep
            scalebw = -scale
        else:
            amp, phase = torch.chunk(torch.tanh(raw), 2, dim=1)
            scalekeep = torch.tensor(0.0, device=raw.device)
            scalefw = 2 * amp + phase * 3.14j
            scalefw = scalefw.swapaxes(1, 2).unsqueeze(1)
            scalebw = -scalefw

        if global_raw is not None:
            gamp, gphase = torch.chunk(torch.tanh(global_raw), 2, dim=1)
            gscale = 2 * gamp + gphase * 3.14j
            scalebw = scalebw - gscale
            scalefw = scalefw + gscale

        return torch.exp(scalefw), torch.exp(scalebw)

    def get_netinput(self, x):
        xc = rearrange(torch.view_as_real(x), "b c z t x y r-> b c (z t x y) r")
        xc = rearrange(
            self.comp(xc), "b c (z t x y) r -> b c z t x (y r)", z=x.shape[2], t=x.shape[3], x=x.shape[4], y=x.shape[5], r=1
        )
        xc = torch.cat((rss(x).unsqueeze(1), xc), dim=1)
        x_local = rearrange(xc, "b c z t ...-> b (c t) z ...")
        if self.global_scale:
            x_global = rearrange(xc, "b c z t ...-> (b z t) c ...")
        else:
            x_global = None
        return x_local, x_global

    def forward(self, x, emb=None):
        # x: b c z t x y
        xc, xg = self.get_netinput(x)
        raw = self.net(xc)
        if self.global_scale:
            xg = self.global_part(xg, x.shape, emb)
        scalefw, scalebw = self.get_scale(raw, self.part_inv, xg)
        return x * scalefw, scalebw

    def global_part(self, x, shape, emb):
        #  (b z t) c ->  b 2 z t 1 1
        x_mean = x.mean(dim=(-1, 2))
        x_max = x.amax(dim=(-1, 2))
        x_min = x.amin(dim=(-1, 2))
        x_std = x.std(dim=(-1, 2))
        xn = torch.cat((x_mean, x_max, x_min, -x_std), dim=1)
        if self.emb_dim > 0:
            emb = torch.broadcast_to(emb, (xn.shape[0], emb.shape[1]))
            xn = torch.cat((xn, emb), dim=1)
        xn = self.global_net(xn)
        xn = rearrange(xn, "(b z t) c -> b c z t () ()", b=shape[0], z=shape[2], t=shape[3])
        return xn
