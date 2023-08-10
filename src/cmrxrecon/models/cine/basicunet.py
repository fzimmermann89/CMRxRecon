import torch
import einops
from cmrxrecon.nets.unet import Unet
from . import CineModel
from cmrxrecon.models.utils import rss
from cmrxrecon.models.utils.ssim import ssim


class BasicUNet(CineModel):
    def __init__(self, input_coils=True, output_coils=True, lr=1e-3, weight_decay=1e-5, schedule=True, normfactor=5000):
        super().__init__()
        self.net = Unet(
            dim=2.5,
            channels_in=1 + 20 * input_coils,
            channels_out=20 if output_coils else 1,
            layer=4,
            filters=32,
            residual="inner",
            norm="group8",
            feature_growth=lambda d: (1, 2, 1.5, 1.34, 1, 1)[d],
            activation="leakyrelu",
            change_filters_last=False,
        )
        self.input_coils = input_coils
        self.output_coils = output_coils
        self.normfactor = normfactor
        with torch.no_grad():
            self.net.last[0].bias.zero_()
            self.net.last[0].weight *= 0.1

    def forward(self, k: torch.Tensor, mask: torch.Tensor, **other) -> dict:
        k = k * self.normfactor
        x0 = torch.fft.ifftn(k, dim=(-2, -1), norm="ortho")
        x_rss = rss(x0)

        if self.input_coils:
            # input the coil reconstructions as channels alongsite the rss
            net_input = einops.rearrange(torch.view_as_real(x0), "b c z t x y r -> (b z) (r c) t x y")
            net_input = torch.cat((net_input, einops.rearrange(x_rss.unsqueeze(1), "b c z t x y -> (b z) c t x y")), 1)
        else:
            # only input the rss as a channel
            net_input = einops.rearrange(x_rss.unsqueeze(1), "b c z t x y -> (b z) c t x y")
        x_net = self.net(net_input)
        if self.output_coils:
            x_net = einops.rearrange(x_net, "(b z) (c r) t x y -> b c z t x y r", b=x0.shape[0], c=10, r=2)
            pred = torch.sum((torch.view_as_real(x0) + x_net).square(), dim=(-1, 1)).sqrt()
        else:
            x_net = einops.rearrange(x_net, "(b z) c t x y -> b z (c t) x y", b=x0.shape[0], c=1)
            pred = x_rss + x_net

        if not self.training:
            pred = pred * 1 / self.normfactor
            x_rss = x_rss * 1 / self.normfactor
        return dict(prediction=pred, rss=x_rss)

    def training_step(self, batch, batch_idx):
        gt = batch.pop("gt") * self.normfactor
        ret = self(**batch)
        prediction, x_rss = ret["prediction"], ret["rss"]

        rss_loss = torch.nn.functional.mse_loss(x_rss, gt)
        l2_loss = torch.nn.functional.mse_loss(prediction, gt)
        ssim_value = ssim(gt, prediction)
        l1_loss = torch.nn.functional.l1_loss(prediction, gt)

        loss = l1_loss - 0.5 * ssim_value + 0.2 * l2_loss
        self.log("train_advantage", (rss_loss - l2_loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", l2_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("rss_loss", rss_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss
