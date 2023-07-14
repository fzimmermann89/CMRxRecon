import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from cmrxrecon.models.utils.grad_ops import GradOperators
from cmrxrecon.models.utils.prox_ops import ClipAct
from cmrxrecon.nets.unet import Unet
from cmrxrecon.models.utils.csm import CSM_refine, CSM_Sriram
from cmrxrecon.models.utils.cg import conj_grad
from cmrxrecon.models.utils.encoding import Dyn2DCartEncObj
from . import CineModel

from einops import rearrange


class Laplace2DCSM(nn.Module):
    def __init__(self):
        super(Laplace2DCSM, self).__init__()
        self.laplace_kernel = torch.tensor([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]]).unsqueeze(0)

    def apply_L(self, csm):
        Nb, Nc, Nz, u, f = csm.shape
        csm = torch.view_as_real(csm)

        csm = rearrange(csm, "b c z u f r -> (b z c) r u f")

        kernel = torch.stack(2 * [self.laplace_kernel], dim=0).to(csm.device)
        csm = F.conv2d(csm, kernel, groups=2, padding=1)

        csm = rearrange(csm, "(b z c) r u f -> b c z u f r", b=Nb, c=Nc, z=Nz)
        csm = torch.view_as_complex(csm.contiguous())

        return csm


class LambdaCNN(nn.Module):
    def __init__(self, cnn_block=Unet(3, channels_in=2, channels_out=2, layer=1, conv_per_dec_block=4, filters=32)):
        super(LambdaCNN, self).__init__()

        # has to be a CNN with input
        self.cnn_block = cnn_block

    def forward(self, x, op_norm_AHA=1.0):
        # convert to 2-channel view: (1,Nx,Ny,Nt) (complex) --> (1,2,Ny,Ny,Nt) (real)
        Nb, Nz, Nt, Nu, Nf = x.shape

        x = torch.view_as_real(x)

        x = rearrange(x, "b z t u f ch -> (b z) ch u f t")

        # padding
        # arbitrarily chosen, maybe better to choose it depending on the
        # receptive field of the CNN or so;
        # seems to be important in order not to create "holes" in the
        # lambda_maps in t-direction
        npad_xy = 4
        npad_t = 8

        pad_refl = (0, 0, npad_xy, npad_xy, npad_xy, npad_xy)
        pad_circ = (npad_t, npad_t, 0, 0, 0, 0)

        x = F.pad(x, pad_refl, mode="reflect")
        x = F.pad(x, pad_circ, mode="circular")

        # estimate parameter map
        Lambda_cnn = self.cnn_block(x)

        # crop
        Lambda_cnn = Lambda_cnn[:, :, npad_xy:-npad_xy, npad_xy:-npad_xy, npad_t:-npad_t]

        # double spatial map and stack
        lambda_reg_xy = torch.stack(2 * [Lambda_cnn[:, 0, ...]], dim=1)
        lambda_reg_t = Lambda_cnn[:, 1, ...].unsqueeze(1)
        lambda_reg = torch.cat([lambda_reg_xy, lambda_reg_t], dim=1)

        lambda_reg = 1e-6 * op_norm_AHA * F.softplus(lambda_reg)

        # reshape to
        lambda_reg = rearrange(lambda_reg, "(b z) d u f t -> b d z t u f ", b=Nb, d=3, z=Nz)

        return lambda_reg


class DcompCNN(nn.Module):
    def __init__(self, cnn_block=Unet(2, channels_in=2, channels_out=1, layer=1, conv_per_dec_block=4, filters=32)):
        super(DcompCNN, self).__init__()

        # has to be a CNN with input
        self.cnn_block = cnn_block

    def forward(self, k):
        k = torch.mean(k, dim=(1, 3))

        Nb, Nz, Nu, Nf = k.shape

        k = torch.view_as_real(k)

        k = rearrange(k, "b z u f ch-> (b z) ch u f")

        dcomp = F.softplus(self.cnn_block(k).squeeze(1))  # will have shape ( bz u f)

        # normalize to have unit norm
        print("todo")

        dcomp = rearrange(dcomp, "(b z) u f -> b z u f ", b=Nb, z=Nz)

        dcomp = dcomp.unsqueeze(1).unsqueeze(3)

        return dcomp


class NNPDHG4DynMRIwTVNN(nn.Module):

    """
    alternate T times of solving problems

    solve

    """

    def __init__(
        self,
        Dyn2DEncObj,
        lambda_unet=Unet(3, channels_in=2, channels_out=2, layer=1, conv_per_dec_block=4, filters=8),
        csm_unet=Unet(2, channels_in=20, channels_out=20, layer=4, conv_per_dec_block=2, filters=16),
        T=64,
    ):
        super(NNPDHG4DynMRIwTVNN, self).__init__()

        # MR encoding objects
        self.Dyn2DEncObj = Dyn2DEncObj

        # gradient operators and clipping function
        dim = 3
        self.GradOps = GradOperators(dim, mode="forward")
        self.ClipAct = ClipAct()

        # operator norms
        self.op_norm_AHA = torch.sqrt(torch.tensor(1.0))  # op-norm is one for appropriate csms
        self.op_norm_GHG = torch.sqrt(torch.tensor(dim * 4.0))  # can be estimtaed by uncommenting below,
        self.L = np.sqrt(self.op_norm_AHA**2 + self.op_norm_GHG**2)

        # (log) constants depending on the operators
        self.tau = nn.Parameter(torch.tensor(10.0), requires_grad=True)  # starting value approximately  1/L
        self.sigma = nn.Parameter(torch.tensor(10.0), requires_grad=True)  # starting value approximately  1/L

        # theta should be in \in [0,1]
        self.theta = nn.Parameter(torch.tensor(10.0), requires_grad=True)  # starting value approximately  1

        self.T = T

        self.LaplaceOps = Laplace2DCSM()

        self.lambda_cnn = LambdaCNN(lambda_unet)
        # self.csm_cnn = CSM_refine(csm_unet)
        self.csm_cnn = CSM_Sriram(csm_unet)

    def apply_G4D(self, x: torch.Tensor) -> torch.Tensor:
        """apply G to a 4D tensor also contaiing the slices"""

        Nb, Nz, Nt, u, f = x.shape

        # reshape to move z to slices
        x = rearrange(x, "b z t u f -> (b z) u f t")

        Gx = self.GradOps.apply_G(x)

        Gx = rearrange(Gx, "(b z) ch u f t -> b ch z t u f", b=Nb, ch=3)

        return Gx

    def apply_GH4D(self, z: torch.tensor) -> torch.Tensor:
        """apply G^H to a 4D tensor also contaiing the slices"""

        Nb, ch, Nz, Nt, u, f = z.shape

        z = rearrange(z, "b ch z t u f -> (b z) ch u f t")

        GHz = self.GradOps.apply_GH(z)

        GHz = rearrange(GHz, "(b z) u f t -> b z t u f ", b=Nb)

        return GHz

    def forward(self, y: torch.Tensor, mask: torch.Tensor, csm: torch.Tensor) -> torch.Tensor:
        L = self.L
        # sigma, tau, theta
        sigma = (1 / L) * torch.sigmoid(self.sigma)  # \in (0,1/L)
        tau = (1 / L) * torch.sigmoid(self.tau)  # \in (0,1/L)
        theta = torch.sigmoid(self.theta)  # \in (0,1)

        # csm = self.csm_cnn(csm)
        csm = self.csm_cnn(y)  # for the SIRIAM approach

        # zerof filled reconstruction, i.e. A^H y with the estimated csms
        # x0 = self.Dyn2DEncObj.apply_AH(y, csm, mask)
        # x0 = torch.fft.ifftn(y,dim=(-2,-1),norm='ortho').sum(1)

        AHA = lambda x: self.Dyn2DEncObj.apply_AHA(x, csm, mask)
        AHy = self.Dyn2DEncObj.apply_AH(y, csm, mask)
        x0 = conj_grad(AHA, AHy, AHy.clone(), niter=6)

        if self.T == 0:
            print("solve only neqs")

        if self.T != 0:
            print("run pdhg")
            # x0 = AHy.clone()

            Nb, Nz, Nt, us, fs = x0.shape
            device = x0.device

            xbar = x0.clone()
            x0 = x0.clone()

            # dual variable
            p = torch.zeros(y.shape, dtype=y.dtype).to(device)
            q = torch.zeros(Nb, 3, Nz, Nt, us, fs, dtype=x0.dtype).to(device)

            Lambda_cnn = self.lambda_cnn(x0)

            for ku in range(self.T):
                print(ku)
                # update p
                p = (p + sigma * (self.Dyn2DEncObj.apply_A(xbar, csm, mask) - y)) / (1.0 + sigma)

                q = self.ClipAct(q + sigma * self.apply_G4D(xbar), Lambda_cnn)
                x1 = x0 - tau * self.Dyn2DEncObj.apply_AH(p, csm, mask) - tau * self.apply_GH4D(q)

                # update xbar
                xbar = x1 + theta * (x1 - x0)
                x0 = x1

        p_k = self.Dyn2DEncObj.apply_A(x0, csm, mask=None)

        p_x = x0.abs()

        xrss = self.Dyn2DEncObj.apply_RSS(y)

        return p_x, p_k, csm, xrss


class NNPDHG4DynMRIwTVRecon(CineModel):
    def __init__(self, T=0, lr=1e-6, weight_decay=0.0, schedule=False):
        super().__init__()
        # TODO: choose parameters

        self.net = NNPDHG4DynMRIwTVNN(Dyn2DCartEncObj(), T=T)

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, k: torch.Tensor, mask: torch.Tensor, csm: torch.Tensor = None, **other) -> dict:
        """
        JointModelBasedCSMImageRecon

        Parameters
        ----------
        k
            shape: (batch, coils, z, t, undersampled, fullysampled)
        mask
            shape: (batch, z, t, undersampled, fullysampled)
        csm
            shape: (batch, coils, z, undersampled, fullysampled)

        Returns
        -------
            x, ..., rss
        """
        p_x, p_k, p_csm, xrss = self.net(k, mask, csm)
        return dict(prediction=p_x, p_k=p_k, p_csm=p_csm, rss=xrss)

    def training_step(self, batch, batch_idx):
        gt = batch.pop("gt")
        k_full = batch.pop("kf")
        ret = self(**batch)

        # self.cdl_unit_norm_proj()

        # MSE loss on the images

        # prediction, rss = ret["prediction"], ret["rss"]
        rss = ret["rss"]

        # loss = torch.nn.functional.mse_loss(prediction, gt)
        print("KEYS OF ret: {}".format(ret.keys()))

        # MSE on the k-space data
        k_prediction = ret["p_k"]
        loss = torch.nn.functional.mse_loss(torch.view_as_real(k_prediction), torch.view_as_real(k_full))

        rss_loss = torch.nn.functional.mse_loss(rss, gt)
        self.log("train_advantage", (rss_loss - loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("rss_loss", rss_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
        # # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        # # 	optimizer, max_lr=3e-3, total_steps=self.trainer.estimated_stepping_batches, pct_start=0.05, anneal_strategy="cos", cycle_momentum=True, div_factor=30, final_div_factor=1e3, verbose=False
        # # )
        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
