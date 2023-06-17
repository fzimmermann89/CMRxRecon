import torch
import torch.nn as nn
import einops
from cmrxrecon.nets.unet import Unet
from .cg import conj_grad
from . import CineModel
from .csm import CSM_refine


class ImgUNetSequence(nn.Module):

    """
    a sequence of two 3D UNets which are applied to map a 4D input (x,y,z,t)
    to a 4D input by processing first
            2D + time: (x,y,t) --> (x,y,t)
    and then
            3D: (x,y,z) --> (x,y,z)

    """

    def __init__(self, net_xyz, net_xyt):
        super().__init__()

        self.net_xyt = net_xyt
        self.net_xyz = net_xyz

    def forward(self, x):
        Nb = x.shape[0]

        # change to real view
        x = torch.view_as_real(x)

        # from 4D to 3D (2D + time)
        x = einops.rearrange(x, "b z t y x ch -> (b z) ch y x t")

        # apply spatio-temporal NN
        x = self.net_xyt(x)

        # switch time and slices dimensions; i.e. to three spatial dimensions)
        x = einops.rearrange(x, "(b z) ch y x t -> (b t) ch y x z", b=Nb)

        # apply 3D spatial NN
        x = self.net_xyz(x)

        # change back to original shape and complex view
        x = torch.view_as_complex(einops.rearrange(x, "(b t) ch y x z -> b z t y x ch", b=Nb).contiguous())

        return x


class JointCSMImageReconNN(nn.Module):
    def __init__(self, EncObj, net_img, net_csm, needs_csm=True):
        """
        JointCSMImageReconNN

        Parameters
        ----------
        EncObj : Encoding object, e.g. Dyn2DEncObj
        net_img : learned mapping between two imaages
        net_csm : learned mapping between two csms
        needs_csm : bool, whether the network needs the csm as input
        """
        super().__init__()
        self.EncObj = EncObj
        self.net_img = net_img
        self.net_csm = net_csm
        self.needs_csm = needs_csm

    def forward(self, k: torch.Tensor, mask: torch.Tensor, csm: torch.Tensor = None) -> torch.Tensor:
        # y: (batch, coils, z, t, undersampled, fullysampled)
        # csm: (batch, coils, z, undersampled, fullysampled)
        # mask: (batch, z, t, undersampled, fullysampled)

        if self.needs_csm:
            if csm is None:
                raise ValueError("csm is required for this network")
            csm = self.net_csm(csm)
        else:
            csm = self.net_csm(k)

        # RSS recon
        xrss = self.EncObj.apply_RSS(k)

        # zerof filled reconstruction, i.e. A^H y with the estimated csms
        AHy = self.EncObj.apply_AH(k, csm, mask)

        # approximately solve the normal equations to get a better
        # initialization for the image
        AHA = lambda x: self.EncObj.apply_AHA(x, csm, mask)
        xneq = conj_grad(AHA, AHy, AHy, niter=4)

        # create x0 = r * exp(i * phi)
        # with r = xrss (magnitude image) and phi = angle(xneq),
        # where xneq is the approximate solution of the normal equations A(C)^H A x = A(C)^Hy
        x = xrss * torch.exp(1j * xneq.angle())

        # apply CNN
        x = self.net_img(x)

        # apply (full) forward model with estimated csms to xcnn
        kest = self.EncObj.apply_A(x, csm, mask=None)

        # estimated image using RSS
        # xest = self.Dyn2DEncObj.apply_RSS(kest)
        xest = x.abs()

        return xest, kest, csm


class JointCSMImageRecon(CineModel):
    def __init__(self):
        self.net = JointCSMImageReconNN()

    def forward(self, k: torch.Tensor, mask: torch.Tensor, csm: torch.Tensor) -> torch.Tensor:
        return self.net(k, mask, csm)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-5)
        return optimizer
        # # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        # # 	optimizer, max_lr=3e-3, total_steps=self.trainer.estimated_stepping_batches, pct_start=0.05, anneal_strategy="cos", cycle_momentum=True, div_factor=30, final_div_factor=1e3, verbose=False
        # # )
        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
