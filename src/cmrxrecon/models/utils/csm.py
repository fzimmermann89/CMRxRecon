import einops
import numpy as np
import torch
from torch import nn


class CSM_Sriram(nn.Module):

    """
    CNN to estimate CSM from y based on Sriram
    https://arxiv.org/pdf/2004.06688.pdf; figure 1; eq (12)
    (i.e. from the zero-filled recon of the center lines)

    """

    def __init__(self, net_csm: nn.Module):
        super().__init__()
        self.refine_csm = CSM_refine2(net_csm)

    def forward(self, y: torch.Tensor, n_center_lines: int = 24):
        # temporal mean
        ym = y.mean(3, keepdim=False)

        # mask out everything but acs
        mask_center = torch.ones_like(ym)
        mask_center[..., n_center_lines // 2 : -n_center_lines // 2, :] = 0
        mask_center[..., n_center_lines : - n_center_lines] = 0
        
        ym = mask_center * ym

        # zero-filled recon
        x0 = torch.fft.ifftn(ym, dim=(-2, -1), norm="ortho")
        x0_img = x0.sum(1)
        norm_factor = torch.pow(torch.sum(x0.conj() * x0, dim=1, keepdim=True), -0.5)
        x0 = x0 * norm_factor
        
        supp = torch.heaviside(torch.nn.functional.threshold(x0_img.abs(), 0.005 * x0_img.abs().max(), 0), torch.tensor([0.]).to(y.device)).unsqueeze(1)

        csm = self.refine_csm(x0 * supp) #+ supp * x0
        #csm =  x0 * supp #* x0
        
        norm_factor = torch.sum(csm.conj() * csm, dim=1, keepdim=True).real
        norm_factor = torch.pow(norm_factor, -0.5)
        norm_factor = torch.nan_to_num(norm_factor, nan=.0, posinf = 0)
        
        csm =norm_factor * csm * supp
        
        return csm
    
    
class CSM_Yiasemis(nn.Module):

    """
    from Figure 2
    https://openaccess.thecvf.com/content/CVPR2022/papers/Yiasemis_Recurrent_Variational_Network_A_Deep_Learning_Inverse_Problem_Solver_Applied_CVPR_2022_paper.pdf

    """

    def __init__(self, net_csm: nn.Module):
        super().__init__()
        #self.refine_csm = CSM_refine2(net_csm)

    def forward(self, y: torch.Tensor, n_center_lines: int = 24):
        # temporal mean
        ym = y.mean(3, keepdim=False)

        # mask out everything but acs
        mask_center = torch.ones_like(ym)
        mask_center[..., n_center_lines // 2 : -n_center_lines // 2, :] = 0
        mask_center[..., n_center_lines : - n_center_lines] = 0
        
        ym = mask_center * ym

        # zero-filled recon
        x0 = torch.fft.ifftn(ym, dim=(-2, -1), norm="ortho")
        xRSS = x0.abs().square().sum(1).pow(0.5)
        
        csm = x0 / xRSS
        
        return csm


class CSM_refine(nn.Module):

    """
    CNN to "refine" CSM from the CSMs initially estimated with ESPIRIT

    """

    def __init__(self, net_csm: nn.Module):
        super().__init__()
        self.net_csm = net_csm

    def forward(self, csm: torch.Tensor):
        Nb = csm.shape[0]

        # to real 2D input tensors
        csm = torch.view_as_real(csm)
        csm = einops.rearrange(csm, "b c z y x r -> (b z) (c r) y x")

        # apply cnn
        csm = csm + self.net_csm(csm)

        # rearrange to complex
        csm = torch.view_as_complex(einops.rearrange(csm, "(b z) (c r) y x -> b c z y x r", b=Nb, r=2).contiguous())

        # normalize output
        norm_factor = torch.pow(torch.sum(csm.conj() * csm, dim=1, keepdim=True), -0.5)

        return norm_factor * csm
    
    
class CSM_refine2(nn.Module):

    """
    CNN to "refine" CSM from the CSMs initially estimated with ESPIRIT

    """

    def __init__(self, net_csm: nn.Module):
        super().__init__()
        self.net_csm = net_csm

    def forward(self, csm: torch.Tensor):
        Nb, Nc, Nz = csm.shape[0], csm.shape[1], csm.shape[2]

        version = 'version2'
        if version == 'version1':
            # to real 2D input tensors
            csm = torch.concatenate([csm.real, csm.imag],dim=1)
            
            pattern = "b c2 z u f-> (b z) c2 u f"
            
            
            csm = einops.rearrange(csm, pattern)
    
            # apply cnn
            csm = csm + self.net_csm(csm)
            
            pattern = "(b z) c2 u f-> b c2 z u f"
            csm = einops.rearrange(csm, pattern, b=Nb, c2=2*Nc, z=Nz)
            
            csm = torch.stack([csm[:,:Nc,...], csm[:,Nc:,...]],dim=-1)
            
            csm = torch.view_as_complex(csm.contiguous())
            
        elif version == 'version2':
            
            csm = torch.view_as_real(csm)
            
            pattern = "b c z u f c2-> (b z c) c2 u f"
            
            csm = einops.rearrange(csm, pattern)
            
            csm = csm + self.net_csm(csm)
            
            pattern = " (b z c) c2 u f -> b c z u f c2"
            csm = einops.rearrange(csm, pattern, b=Nb, c=Nc, z=Nz)
            
            csm = torch.view_as_complex(csm.contiguous())

        # normalize output
        #norm_factor = torch.pow(torch.sum(csm.conj() * csm, dim=1, keepdim=True), -0.5)

        return csm


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
