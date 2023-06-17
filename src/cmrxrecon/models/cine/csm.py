import torch
from torch import nn
import einops


class CSM_Sriram(nn.Module):

    """
    CNN to estimate CSM from y based on Sriram
    https://arxiv.org/pdf/2004.06688.pdf; figure 1; eq (12)
    (i.e. from the zero-filled recon of the center lines)

    """

    def __init__(self, net_csm: nn.Module):
        super().__init__()
        self.refine_csm = CSM_refine(net_csm)

    def forward(self, y: torch.Tensor, n_center_lines: int = 24):
        # temporal mean
        ym = y.mean(3, keepdim=True)

        # mask out everything but acs
        mask_center = torch.ones_like(ym)
        mask_center[..., n_center_lines // 2 : -n_center_lines // 2, :] = 0
        ym = mask_center * ym

        # zero-filled recon
        x0 = torch.fft.ifftn(ym, dim=(-2, -1), norm="ortho")
        norm_factor = torch.pow(torch.sum(x0.conj() * x0, dim=1, keepdim=True), -0.5)
        x0 = x0 * norm_factor

        csm = self.refine_csm(x0)
        return csm


class CSM_refine(nn.Module):

    """
    CNN to "refine" CSM from the CSMs initially estimated with ESPIRIT

    """

    def __init__(self, net_csm:nn.Module):
        super().__init__()
        self.net_csm = net_csm

    def forward(self, csm:torch.Tensor):
        Nb = csm.shape[0]
        
        # to real 2D input tensors
        csm = torch.view_as_real(csm)
        csm = einops.rearrange(csm, "b c z y x r -> (b z) (c r) y x")

        # apply cnn
        csm = self.net_csm(csm)

        # rearrange to complex
        csm = torch.view_as_complex(einops.rearrange(csm, "(b z) (c r) y x -> b c z y x r", b=Nb, r=2).contiguous())

        # normalize output
        norm_factor = torch.pow(torch.sum(csm.conj() * csm, dim=1, keepdim=True), -0.5)

        return norm_factor * csm
