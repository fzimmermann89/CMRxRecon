import torch
import torch.nn as nn
from . import CineModel
from cmrxrecon.nets.unet import Unet
from cmrxrecon.models.utils.csm import CSM_Sriram
from cmrxrecon.models.utils.encoding import Dyn2DCartEncObj
from einops import rearrange
from cmrxrecon.models.utils.ssim import ssim
from neptune.new.types import File as neptuneFile
from pytorch_lightning.loggers.neptune import NeptuneLogger


class ImgUNetWrapper(nn.Module):
    """
    simple 2D + time UNet applied to an image of shape (Nb, Nz, Nt, Nu, Nuf)
    """

    def __init__(self, img_unet, mode="xyt"):
        super().__init__()
        self.img_unet = img_unet
        self.mode = mode

    def forward(self, x):
        Nb, Nz, Nt, Nu, Nf = x.shape
        # change to real view
        x = torch.view_as_real(x)
        if self.mode == "xyt":
            pattern_forw = "b z t y x ch -> (b z) ch t y x"
            pattern_adj = "(b z) ch t y x -> b z t y x ch"
        elif self.mode == "xyz":
            pattern_forw = "b z t y x ch -> (b t) ch y x z"
            pattern_adj = "(b t) ch y x z -> b z t y x ch"
        x = rearrange(x, pattern_forw)
        x = self.img_unet(x)
        x = rearrange(x, pattern_adj, b=Nb, z=Nz, ch=2)
        x = torch.view_as_complex(x.contiguous())
        return x


class _E2EVarNet(nn.Module):
    def __init__(
        self,
        img_unet_hparams: tuple[int, int, int],
        csm_unet_hparams: tuple[int, int, int],
        T: int,
    ):
        super().__init__()

        self.EncObj = Dyn2DCartEncObj()

        img_layer, img_blocks, img_filter = img_unet_hparams
        img_unet = Unet(
            dim=2.5,
            channels_in=2,
            channels_out=2,
            layer=img_layer,
            filters=img_filter,
            conv_per_dec_block=img_blocks,
            conv_per_enc_block=img_blocks,
            feature_growth=(1, 1.5, 2, 2, 1, 1),
            downsample_dimensions=((-1, -2), (-1, -2, -3), (-1, -2, -3), (-1, -2, -3)),
            activation="leakyrelu",
            change_filters_last=False,
        )
        csm_layer, csm_blocks, csm_filter = csm_unet_hparams
        csm_unet = Unet(
            dim=2,
            channels_in=20,
            channels_out=20,
            layer=csm_layer,
            filters=csm_filter,
            conv_per_dec_block=csm_blocks,
            conv_per_enc_block=csm_blocks,
            feature_growth=(1, 2, 2, 2, 1, 1),
            activation="leakyrelu",
        )

        self.net_img = ImgUNetWrapper(img_unet)
        self.net_csm = CSM_Sriram(csm_unet)

        self.etas = nn.Parameter(torch.zeros(T))
        self.softplus = nn.Softplus(beta=5)
        self.T = T

    def forward(self, k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        k_measured = k.clone()
        xrss = self.EncObj.apply_RSS(k)

        # refine CSMs
        csm = self.net_csm(k)

        for t in range(self.T):
            eta_t = self.softplus(self.etas[t])
            AHk = self.EncObj.apply_AH(k, csm, mask=None)
            xnn = self.net_img(AHk)
            Axnn = self.EncObj.apply_A(xnn, csm, mask=None)
            k = k - eta_t * self.EncObj.apply_mask(k - k_measured, mask) + Axnn

        p_csm = csm
        p_k = k
        p_x = self.EncObj.apply_RSS(k)

        return p_x, p_k, p_csm, xrss


def combined_loss(prediction, gt):
    l1 = nn.functional.l1_loss(prediction, gt)
    gtm = gt.amax(dim=(-1, -2), keepdim=True)
    pm = prediction.amax(dim=(-1, -2), keepdim=True)
    l2 = nn.functional.mse_loss(prediction / pm.detach(), gt / gtm.detach())
    ssim_loss = -ssim(prediction, gt)
    max_loss = nn.functional.mse_loss(pm, gtm)
    loss = 0.5 * l1 + 0.2 * l2 + 0.3 * ssim_loss + 1e-3 * max_loss
    return loss


class E2EVarNet(CineModel):
    def __init__(
        self,
        T: int = 4,
        lr: float = 6e-4,
        weight_decay: float = 1e-4,
        schedule: bool = True,
        loss_fct: str = "l1",
        normfactor: float = 5e3,
        img_unet_hparams: tuple[int, int, int] = (4, 2, 32),
        csm_unet_hparams: tuple[int, int, int] = (3, 2, 32),
    ):
        super().__init__()

        self.net = _E2EVarNet(T=T, csm_unet_hparams=csm_unet_hparams, img_unet_hparams=img_unet_hparams)
        self.normfactor = normfactor

        if loss_fct == "l2":
            self.loss_fct = nn.MSELoss()
        elif loss_fct == "l1":
            self.loss_fct = nn.L1Loss()
        elif loss_fct == "comb":
            self.loss_fct = combined_loss
        else:
            raise NotImplementedError(f"loss function {loss_fct} not implemented")

    def forward(self, k: torch.Tensor, mask: torch.Tensor, **other) -> dict:
        p_x, p_k, p_csm, xrss = self.net(k * self.normfactor, mask)
        if not self.training:
            p_x = p_x / self.normfactor
            xrss = xrss / self.normfactor
            p_k = p_k / self.normfactor
        return dict(prediction=p_x, p_k=p_k, p_csm=p_csm, rss=xrss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        ret = self(**batch)

        gt = self.normfactor * batch.pop("gt")
        prediction, rss = ret["prediction"], ret["rss"]

        loss = self.loss_fct(prediction, gt)
        with torch.no_grad():
            rssloss = torch.nn.functional.mse_loss(rss, gt)
            l2loss = torch.nn.functional.mse_loss(prediction, gt)

        self.log("train_advantage", (rssloss - l2loss) / rssloss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ret = super().validation_step(batch, batch_idx)
        p_csm = ret["p_csm"]

        if batch_idx == 0:
            for logger in self.loggers:
                if isinstance(logger, NeptuneLogger):
                    csm_img = p_csm[0, 0, 0, :, :].detach().abs().cpu().numpy()
                    csm_img = csm_img - csm_img.min()
                    csm_img = csm_img / csm_img.max()
                    logger.experiment["val/csm_image"].log(neptuneFile.as_image(csm_img))
