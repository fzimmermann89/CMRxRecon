import einops
import torch
import torch.nn as nn

from cmrxrecon.models.utils.cg import conj_grad
from cmrxrecon.models.utils.csm import CSM_refine
from cmrxrecon.models.utils.encoding import Dyn2DCartEncObj
from cmrxrecon.nets.unet import Unet

from . import CineModel


class ImgUNetSequence(nn.Module):

    """
    a sequence of two 3D UNets which are applied to map a 4D input (x,y,z,t)
    to a 4D input by processing first
            2D + time: (x,y,t) --> (x,y,t)
    and then
            3D: (x,y,z) --> (x,y,z)

    """

    def __init__(self, net_xyt, net_xyz=None):
        super().__init__()

        self.net_xyt = net_xyt
        self.net_xyz = net_xyz

    def forward(self, x):
        Nb = x.shape[0]

        # change to real view
        x = torch.view_as_real(x)

        # 2d + time unet
        x = einops.rearrange(x, "b z t y x ch -> (b z) ch y x t")
        x = x + self.net_xyt(x)

        if self.net_xyz is not None:
            # xyz unet
            x = einops.rearrange(x, "(b z) ch y x t -> (b t) ch y x z", b=Nb)
            x = x + self.net_xyz(x)
            x = torch.view_as_complex(einops.rearrange(x, "(b t) ch y x z -> b z t y x ch", b=Nb).contiguous())
        else:
            # change back to original shape and complex view
            x = torch.view_as_complex(einops.rearrange(x, "(b z) ch y x t -> b z t y x ch", b=Nb).contiguous())

        return x


class JointCSMImageReconNN(nn.Module):
    def __init__(self, EncObj, net_img, net_csm, needs_csm: bool = True, normfactor: float = 1e2):
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
        self.normfactor = normfactor

    def forward(self, k: torch.Tensor, mask: torch.Tensor, csm: torch.Tensor = None) -> torch.Tensor:
        # y: (batch, coils, z, t, undersampled, fullysampled)
        # csm: (batch, coils, z, undersampled, fullysampled)
        # mask: (batch, z, t, undersampled, fullysampled)

        if self.needs_csm:
            if csm is None:
                raise ValueError("csm is required for this network")
            p_csm = self.net_csm(csm)
        else:
            p_csm = self.net_csm(k)

        # RSS recon
        xrss = self.EncObj.apply_RSS(k)  # (batch, z, t, undersampled, fullysampled)

        # zerof filled reconstruction, i.e. A^H y with the estimated csms
        AHy = self.EncObj.apply_AH(k, csm, mask)  # (batch, coils, z, t, undersampled, fullysampled)

        # approximately solve the normal equations to get a better
        # initialization for the image
        AHA = lambda x: self.EncObj.apply_AHA(x, csm, mask)
        xneq = conj_grad(AHA, AHy, AHy, niter=4)  # (batch, coils, z, t, undersampled, fullysampled)

        # create x0 = r * exp(i * phi)
        # with r = xrss (magnitude image) and phi = angle(xneq),
        # where xneq is the approximate solution of the normal equations A(C)^H A x = A(C)^Hy
        x = xrss * torch.exp(1j * xneq.angle())

        # apply CNN
        x = self.net_img(x * self.normfactor) * (1 / self.normfactor)  # (batch, coils, z, t undersampled, fullysampled)

        # apply (full) forward model with estimated csms to xcnn
        p_k = self.EncObj.apply_A(x, csm, mask=None)

        # estimated image using RSS
        p_x = self.EncObj.apply_RSS(p_k)
        # p_x = x.abs()

        return p_x, p_k, p_csm, xrss


class JointCSMImageRecon(CineModel):
    def __init__(self, precalculated_csms=True, lr=1e-4, weight_decay=1e-6, schedule=False):
        super().__init__()
        # TODO: choose parameters

        net_xyt = Unet(3, channels_in=2, channels_out=2, layer=3, filters=8)
        net_xyz = None  # Unet(3, channels_in=2, channels_out=2, layer=3, filters=8), #todo: fix z dimension

        with torch.no_grad():  # init close to identity
            net_xyt.last[0].weight.data *= 1e-1
            net_xyt.last[0].bias.zero_()

        net_img = ImgUNetSequence(
            net_xyt=net_xyt,
            net_xyz=net_xyz,
        )

        Ncoils = 10
        net_csm = Unet(  # TODO: choose parameters
            2,
            channels_in=2 * Ncoils,
            channels_out=2 * Ncoils,
            layer=2,
            filters=32,
        )
        with torch.no_grad():  # init close to identity
            net_csm.last[0].weight.data *= 1e-1
            net_csm.last[0].bias.zero_()

        net_csm = CSM_refine(net_csm)
        self.net = JointCSMImageReconNN(EncObj=Dyn2DCartEncObj(), net_img=net_img, net_csm=net_csm, needs_csm=precalculated_csms)

    def forward(self, k: torch.Tensor, mask: torch.Tensor, csm: torch.Tensor = None, **other) -> dict:
        """
        JointCSMImageRecon

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
