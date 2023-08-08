import torch


class MultiCoilDCLayer(torch.nn.Module):
    def __init__(self, Nc: int, embed_dim: int = 0, lambda_bias: float = 1.0, return_k: bool = False, input_nn_k: bool = False):
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
        embed_dim:
            Embedding dimension for the lambda conditioning.
            If 0, static lambdas for each coil are used.
        lambda_bias: Initial bias for the lambda conditioning
        return_k: If True, return k-space data, otherwise return image space data
        input_nn_k: If True, the NN input is assumed to be k-space data, otherwise image space data.
        """
        super().__init__()
        if embed_dim > 0:
            self.lambda_proj = torch.nn.Linear(embed_dim, Nc)
            with torch.no_grad():
                self.lambda_proj.weight.zero_()
                self.lambda_proj.bias[:] = lambda_bias
        else:
            self.lambda_proj = lambda x: x + lambda_bias
        self.input_nn_k = input_nn_k
        self.return_k = return_k

    def forward(self, k: torch.Tensor, nn: torch.Tensor, mask: torch.Tensor, lambda_embed: torch.Tensor) -> torch.Tensor:
        knn = nn if self.input_nn_k else torch.fft.fft2(nn, norm="ortho")
        lam = self.lambda_proj(lambda_embed)[:, :, None, None, None, None]  # [Nb, Nc, 1, 1, 1, 1]
        mask = mask.unsqueeze(1)  # [Nb, Nc=1, Nz=1, Nz=1, Nu, Nf=1]
        fk = mask * (1.0 / (1.0 + lam))  # facor for k data, 0 for missing data
        fn = 1 - fk  # factor for knn data
        kdc = fn * knn + fk * k
        ret = kdc if self.return_k else torch.fft.ifft2(kdc, norm="ortho")
        return ret
