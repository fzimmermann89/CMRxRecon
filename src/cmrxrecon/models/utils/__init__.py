import torch


@torch.jit.script
def rss(x: torch.Tensor) -> torch.Tensor:
    return x.abs().square().sum(1).sqrt()
