import torch
from torch import nn


class RandomFlipAlongDimensions(nn.Module):
    def __init__(self, p: float | tuple[float, ...] = 0.5, dim: int | tuple[int, ...] = (-1, -2)):
        """
        Randomly flip along dimensions.

        Parameters
        ----------
        p, default 0.5
            Propability of flip along a dimension, default 0.5 means 50% for all dimensions in dim)
        dim, default (-1,-2)
            Dimensions to flip in
        """
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        if isinstance(p, float):
            p = len(dim) * (p,)
        self.p = p
        self.dim = dim

    def forward(self, x):
        for d, p in zip(self.dim, self.p):
            if torch.rand(1) < p:
                x = torch.flip(x, dims=(d,))
        return x


class RandomKSpaceFlipAlongDimensions(nn.Module):
    def __init__(self, p: float | tuple[float, ...] = 0.5, dim: int | tuple[int, ...] = (-1, -2), centered: bool = False):
        """
        Randomly flip along dimensions in reciprocal space.
        I.e. receive k-space data and modify it such that the resulting image is flipped.

        Parameters
        ----------
        p, default 0.5
            Propability of flip along a dimension,  default 0.5 means 50% for all dimensions in dim)
        dim, default (-1,-2)
            Dimensions to flip in
        centered, default False
            If True, k=0 is in the center of the daa. If False, k=0 is in the top left corner.
        """
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        if isinstance(p, float):
            p = len(dim) * (p,)
        self.p = p
        self.dim = dim
        self.centered = centered

    def forward(self, k):
        flips = 0
        for d, p in zip(self.dim, self.p):
            if torch.rand(1) < p:
                # perform flip in image-space. this is equivalent to a multiplication with -1 in k-space
                # multiply center of dths dimension with -1
                if self.centered:
                    k[(slice(None),) * d + (k.shape[d] // 2,)] *= -1
                else:
                    k[(slice(None),) * d + (0,)] *= -1
                flips += 1
        if flips % 2 == 1:
            k *= -1
        return k


class RandomShuffleAlongDimension(nn.Module):
    def __init__(self, p: float | tuple[float, ...] = 0.5, dim: int | tuple[int, ...] = (1,)):
        """
        Randomly shuffle along dimensions

        Parameters
        ----------
        p, default 0.5
            Propability of shuffle along a dimension,  default 0.5  means 50% for all dimensions in dim
        dim, default (-1,-2)
            Dimensions to shuffle in,
        """
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        if isinstance(p, float):
            p = len(dim) * (p,)
        self.p = p
        self.dim = dim

    def forward(self, x):
        for d, p in zip(self.dim, self.p):
            if torch.rand(1) < p:
                x = x[(slice(None),) * d + torch.randperm(x.shape[d])]
        return x
