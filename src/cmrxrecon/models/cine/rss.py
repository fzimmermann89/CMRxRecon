import torch
from . import CineModel


class RSS(CineModel):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, k: torch.Tensor, mask: torch.Tensor, **other) -> dict:
        x0 = torch.fft.ifftn(k, dim=(-2, -1), norm="ortho")
        rss = x0.abs().square().sum(1).sqrt()
        return dict(prediction=rss, rss=rss)
