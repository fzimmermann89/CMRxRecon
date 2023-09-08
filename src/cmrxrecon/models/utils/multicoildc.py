import torch


class MultiCoilDCLayer(torch.nn.Module):
    def __init__(
        self,
        Nc: int,
        embed_dim: int = 0,
        lambda_init: float = 1.0,
        input_nn_k: bool | tuple[bool, bool] = False,
        output_k: bool | tuple[bool, bool] = False,
    ):
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
        lambda_init:  initial bias for the lambda conditioning
        input_nn_k: If True, nn input is assumed to be k-space data, otherwise image space data along axis.
        output_k: If True, the output will be k-space data, otherwise image space data along axis.
        """
        super().__init__()
        if embed_dim > 0:
            self.lambda_proj = torch.nn.Linear(embed_dim, Nc)
            with torch.no_grad():
                self.lambda_proj.weight.zero_()
                self.lambda_proj.bias[:] = lambda_init
        else:
            self.lambda_proj = lambda x: x + lambda_init

        def parse_dims(k_dims: bool | tuple[bool, ...]) -> tuple[int, ...]:
            if k_dims == False:
                return (-1, -2)
            if k_dims == True:
                return ()
            return tuple([-1 - i for i, dim in enumerate(k_dims[::-1]) if not dim])

        self.input_ft_dims = parse_dims(input_nn_k)
        self.output_ft_dims = parse_dims(output_k)

    def forward(self, k: torch.Tensor, nn: torch.Tensor, mask: torch.Tensor, lambda_embed: torch.Tensor) -> torch.Tensor:
        if len(self.input_ft_dims) == 0:
            knn = nn
        else:
            knn = torch.fft.fftn(nn, dim=self.input_ft_dims, norm="ortho")
        lam = self.lambda_proj(lambda_embed)[:, :, None, None, None, None]  # [Nb, Nc, 1, 1, 1, 1]
        mask = mask.unsqueeze(1)  # [Nb, Nc=1, Nz=1, Nz=1, Nu, Nf=1]
        fk = mask * (1.0 / (1.0 + lam))  # factor for k data, 0 for missing data
        fn = 1 - fk  # factor for knn data
        kdc = fn * knn + fk * k
        if len(self.output_ft_dims) == 0:
            ret = kdc
        else:
            ret = torch.fft.ifftn(kdc, dim=self.output_ft_dims, norm="ortho")
        return ret
