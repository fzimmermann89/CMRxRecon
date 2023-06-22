import torch
from torch import nn


class Dyn2DCartEncObj(nn.Module):

    """
    Implementation of operators for MR image reconstruction
    using pytorch v.1.10.1

    A: X --> Y, where A = I_Nz \kron ( (I_Nc \kron S F) C )

    with shapes:
                    x    = (mb, Nz, Nt, N_undersampled, N_fullysampled)
                    csm  = (mb, Nc, Nz, N_undersampled, N_fullysampled)
                    mask = (mb, Nz, Nt, N_undersampled, N_fullysampled)
                    y    = (mb, Nc, Nz, Nt, N_undersampled, N_fullysampled)
    """

    def __init__(self, norm="ortho"):
        self.norm = norm
        super().__init__()

    def apply_C(self, x, csm):
        return csm.unsqueeze(3) * x.unsqueeze(1)

    def apply_E(self, x):
        return torch.fft.fftn(x, dim=(-2, -1), norm=self.norm)

    def apply_mask(self, k, mask):
        if mask is None:
            return k
        return k * mask.unsqueeze(1)

    def apply_A(self, x, csm, mask=None):
        k = self.apply_E(self.apply_C(x, csm))
        if mask is not None:
            k = self.apply_mask(k, mask)
        return k

    def apply_CH(self, xc, csm):
        return torch.sum(csm.conj().unsqueeze(-3) * xc, dim=1, keepdim=False)

    def apply_EH(self, k):
        return torch.fft.ifftn(k, dim=(-2, -1), norm=self.norm)

    def apply_EC(self, x, csm):
        return self.apply_E(self.apply_C(x, csm))

    def apply_AH(self, k, csm, mask):
        k = self.apply_mask(k, mask)
        return self.apply_CH(self.apply_EH(k), csm)

    def apply_AHA(self, x, csm, mask):
        return self.apply_AH(self.apply_A(x, csm, mask), csm, mask)

    def apply_RSS(self, k):
        return torch.fft.ifftn(k, dim=(-2, -1), norm=self.norm).abs().square().sum(1).pow(0.5)
