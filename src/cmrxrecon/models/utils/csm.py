import einops
import numpy as np
import torch
from torch import nn
from . import masked_mean, reciprocal_rss, rss


class CSM_Sriram(nn.Module):

    """
    CNN to estimate CSM from y based on Sriram
    https://arxiv.org/pdf/2004.06688.pdf; figure 1; eq (12)
    (i.e. from the zero-filled recon of the center lines)

    """

    def __init__(
        self,
        net_csm: nn.Module | None,
        normalize: bool = True,
        threshold: float = 0,
        fill_x0=False,
        mask=False,
        limit_acs_fullysampled=True,
        norm_x0=False,
        batch_coils=False,
    ):
        """

        Parameters
        ----------
        net_csm
            net used to get from x0 to csm
        normalize
            normlize CSM such that Sum_coils S.H.*S = 1
        threshold
            rel. threshold for filling or masking
        fill_x0
            fill x0s under the relative (to max) threshold by (1/Nc)**(1/2)*exp(1j mean phase)
        mask
            mask x0s and final csms under the relative (to max) threshold
        limit_acs_fullysampled
            also mask acs in fullysampled diretion
        norm_x0
            normalize x0s such that Sum_coils S.H.*S = 1
        batch_coils
            batch the coils dimension before feeding to net_csm



        """
        super().__init__()
        self.refine_csm = CSM_refine(net_csm, batch_coils=batch_coils, normalize=False) if net_csm is not None else None
        self.threshold = threshold
        self.normalize = normalize
        self.fill_x0 = fill_x0
        self.mask = mask
        self.limit_acs_fullysampled = limit_acs_fullysampled
        self.norm_x0 = norm_x0

        if mask and fill_x0:
            raise ValueError("mask and fill_x0 cannot be True at the same time")
        if (mask or fill_x0) and threshold <= 0:
            raise ValueError("threshold must be > 0 if mask or fill_x0 is True")

    def forward(self, y: torch.Tensor, n_center_lines: int = 24):
        # temporal mean
        ym = y.mean(3, keepdim=False)  # (batch, coils, z, undersampled, fullysampled)

        # mask out everything but acs
        mask_center = torch.ones(*((1,) * (ym.ndim - 2)), *ym.shape[-2:], device=ym.device, dtype=ym.dtype)
        mask_center[..., n_center_lines // 2 : -n_center_lines // 2, :] = 0
        if self.limit_acs_fullysampled:  # also mask acs in fullysampled diretion
            mask_center[..., n_center_lines:-n_center_lines] = 0

        ym = mask_center * ym

        # zero-filled recon
        x0: torch.Tensor = torch.fft.ifftn(ym, dim=(-2, -1), norm="ortho")

        if self.fill_x0 and self.threshold > 0:  # fill low intensity x0s with average phase
            x_rss = rss(x0).unsqueeze(1)
            mask = x_rss > self.threshold * x_rss.max()
            fill_mag = (1 / x0.shape[1]) ** 0.5
            fill_phase = masked_mean(torch.angle(x0), mask, dim=(-1, -2, -3), keepdim=True)
            fill_value = fill_mag * torch.exp(1j * fill_phase)
            x0 = torch.where(mask, x0, fill_value)

        if self.norm_x0:
            if self.fill_x0:
                normfactor = torch.nan_to_num(x_rss.inverse(), nan=0.0, posinf=0)
            else:
                normfactor = reciprocal_rss(x0).unsqueeze(1)
            x0 = x0 * normfactor

        if self.mask and self.threshold > 0:
            x_sum_abs = x0.sum(1).abs()
            mask = (x_sum_abs > self.threshold * x_sum_abs.max()).unsqueeze(1)
            x0 = x0 * mask

        if self.refine_csm is None:
            csm: torch.Tensor = x0
        else:
            csm: torch.Tensor = self.refine_csm(x0)

        if self.normalize:
            norm_factor = torch.nan_to_num(reciprocal_rss(csm).unsqueeze(1), nan=0.0, posinf=0)
            csm = norm_factor * csm

        if self.mask and self.threshold > 0 and self.refine_csm is not None:
            csm = csm * mask

        return csm


class CSM_Sriram_support(CSM_Sriram):

    """
    CNN to estimate CSM from y based on Sriram
    https://arxiv.org/pdf/2004.06688.pdf; figure 1; eq (12)
    (i.e. from the zero-filled recon of the center lines)
    with added normalization before the refinement and support mask

    """

    def __init__(
        self,
        net_csm: nn.Module,
        mask_fullysampled: bool = True,
        threshold: float = 0.005,
        fill_x0: bool = False,
        mask: bool = True,
        norm_x0=True,
        normalize: bool = True,
    ):
        super().__init__(
            net_csm=net_csm,
            threshold=threshold,
            fill_x0=fill_x0,
            mask=mask,
            norm_x0=norm_x0,
            mask_fullysampled=mask_fullysampled,
            normalize=normalize,
        )


class CSM_Yiasemis(CSM_Sriram):

    """
    from Figure 2
    https://openaccess.thecvf.com/content/CVPR2022/papers/Yiasemis_Recurrent_Variational_Network_A_Deep_Learning_Inverse_Problem_Solver_Applied_CVPR_2022_paper.pdf

    """

    def __init__(
        self,
        threshold: float = 0,
        fill_x0: bool = False,
        mask: bool = False,
    ):
        super().__init__(
            net_csm=None, threshold=0.0, fill_x0=fill_x0, mask=mask, norm_x0=False, normalize=True, mask_fullysampled=True
        )


class CSM_refine(nn.Module):

    """
    CNN to "refine" CSM from the CSMs initially estimated with ESPIRIT or x0s

    """

    def __init__(self, net_csm: nn.Module, batch_coils: bool = False, normalize: bool = False):
        """
        Parameters
        net_csm:
            the CNN to refine the CSMs
        batch_coils:
            if True, the input tensor batches are assumed to be (batch, coils, z, y, x, real/imag) "version 2"
        normalize:
            if True, the output CSMs are normalized to RSS=1
        """
        super().__init__()
        self.net_csm = net_csm
        self.batch_coils = batch_coils
        self.normalize = normalize

    def forward(self, csm: torch.Tensor):
        Nb, Nc = csm.shape[:2]

        # to real 2D input tensor batches
        csm = torch.view_as_real(csm)
        if self.batch_coils:
            csm = einops.rearrange(csm, "b c z y x r -> (b z c) r y x")
        else:
            csm = einops.rearrange(csm, "b c z y x r -> (b z) (c r) y x")

        # apply cnn
        csm = csm + self.net_csm(csm)

        # rearrange to back complex 3D tensor batches
        if self.batch_coils:
            csm = torch.view_as_complex(einops.rearrange(csm, "(b z c) r y x -> b c z y x r", b=Nb, c=Nc, r=2).contiguous())
        else:
            csm = torch.view_as_complex(einops.rearrange(csm, "(b z) (c r) y x -> b c z y x r", b=Nb, r=2).contiguous())

        if self.normalize:
            norm_factor = torch.nan_to_num(reciprocal_rss(csm).unsqueeze(1), nan=0.0, posinf=0)
            csm = norm_factor * csm

        return csm


class CSM_refine2(CSM_refine):
    def __init__(self, net_csm: nn.Module, batch_coils: bool = True, normalize: bool = False):
        super().__init__(net_csm, batch_coils, normalize)


def sigpy_espirit(k_centered: torch.Tensor, threshold: float = 0.01, max_iter: int = 150, crop: float = 0.7, fill: bool = True):
    """
    Calulate CSM from the k-space data using sigpy's espirit function

    Parameters
    ----------
    k_centered
        complex k-space data with k-space center lines in the center of the array
        shape: (batch, coil, undersamped, fullysampled)
    threshold
        threshold for the espirit algorithm
    max_iter
        maximum number of iterations for the espirit algorithm
    crop
        cropping value for the espirit algorithm
    fill
        if True, fill all-zero spatial regions in the csms with mean values


    Returns
    -------
    csm
        complex coil sensitivity maps
        shape: (batch, coil, undersamped, fullysampled)
    """

    # coil sensitivity maps
    from sigpy.mri.app import EspiritCalib

    if isinstance(k_centered, torch.Tensor):
        k_centered = k_centered.detach().cpu().numpy()
    csm = [EspiritCalib(sample, max_iter=max_iter, thresh=threshold, crop=crop, show_pbar=False).run() for sample in k_centered]
    csm = np.array(csm)
    if fill:
        # fill all-zero spatial regions (i.e. underdetermined areas) in the csms with
        # 1/sqrt(Nc) * exp(1j * mean(angle(csm)))
        Nc = csm.shape[1]
        zeros = np.all(np.abs(csm) < 1e-6, axis=1)
        fills = np.sqrt(1 / Nc) * np.exp(1j * np.angle(csm).mean((-1, -2)))
        for c, f, m in zip(csm, fills, zeros):
            c[:, m] = f[:, None]
    return csm
