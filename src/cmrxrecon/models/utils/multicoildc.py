import torch


class MultiCoilDCLayer(torch.nn.Module):
    def __init__(self, Nc: int, embed_dim: int = 0, lambda_bias: float = 1.0):
        """
        Data-consistency layer for multiple receiver coils
        computes the argmin of min_x 1/2|| F_I x - y||_2**2 + lambda/2|| x - xnn||_2**2
        for each coil separately
        y.shape = [Nb, Nc, Nz, Nt, Nu, Nf]
        xnn.shape = [Nb, Nc, Nz, Nt, Nu, Nf]
        mask.shape = [Nb, 1, 1, Nu, 1]

        Parameters
        ----------
        Nc: Number of coils
        ebmed_dim:
            Embedding dimension for the lambda conditioning.
            If 0, static lambdas for each coil are used.
        """
        super().__init__()
        if embed_dim > 0:
            self.lambda_proj = torch.nn.Linear(embed_dim, Nc)
            with torch.no_grad():
                self.lambda_proj.weight.zero_()
                self.lambda_proj.bias[:] = lambda_bias
        else:
            self.lambda_proj = lambda x: x + lambda_bias

    def forward(self, k: torch.Tensor, xnn: torch.Tensor, mask: torch.Tensor, lambda_embed: torch.Tensor) -> torch.Tensor:
        lam = self.lambda_proj(lambda_embed)[:, :, None, None, None, None]  # [Nb, Nc, 1, 1, 1, 1]
        mask = mask.unsqueeze(1)  # [Nb, Nc=1, Nz=1, Nz=1, Nu, Nf=1]
        fk = mask * (1.0 / (1.0 + lam))  # facor for k data, 0 for missing data
        fn = 1 - fk  # factor for xnn data
        knn = torch.fft.fft2(xnn, norm="ortho")
        xreg = fn * knn + fk * k
        xdc = torch.fft.ifft2(xreg, norm="ortho")
        return xdc
