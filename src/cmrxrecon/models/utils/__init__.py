import torch


@torch.jit.script
def rss(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return (x.real.square() + x.imag.square()).sum(dim).sqrt()


@torch.jit.script
def reciprocal_rss(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return (x.real.square() + x.imag.square()).sum(dim).rsqrt()


@torch.jit.script
def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: list[int] = (-1,), keepdim: bool = False) -> torch.Tensor:
    return (x * mask).sum(dim=dim, keepdim=keepdim) / mask.sum(dim=dim, keepdim=keepdim)
