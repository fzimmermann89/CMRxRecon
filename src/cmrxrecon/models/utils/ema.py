import torch
from typing import Callable


class EMA(torch.nn.Module):
    def __init__(self, alpha=0.9, max_iter=float("inf")):
        """
        Exponential moving average

        Parameters
        ----------
        alpha: decay factor
        max_iter: maximum number of iterations, after which the average is no longer updated
        """
        super().__init__()
        self.alpha = alpha
        self.register_buffer("ema", torch.tensor(0.0))
        self.register_buffer("ema_unbiased", torch.tensor(float("nan")))
        self.iter: int = 0
        self.max_iter = max_iter

    def forward(self, input: torch.Tensor | Callable):
        if callable(input):
            x: torch.Tensor = input()
        else:
            x: torch.Tensor = input
        if torch.isnan(self.ema_unbiased):
            self.ema_unbiased = x
        if not self.training or self.iter > self.max_iter:
            return self.ema_unbiased
        self.ema = self.ema * self.alpha + x * (1 - self.alpha)
        self.iter += 1
        self.ema_unbiased = self.ema / (1 - self.alpha**self.iter)
        return self.ema_unbiased
