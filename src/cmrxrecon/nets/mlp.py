import torch
from typing import Callable


class MLP(torch.nn.Module):
    def __init__(self, dims: list[int], activation: Callable = lambda: torch.nn.SiLU(True), dropout: float = 0.2):
        """
        Multi-layer perceptron

        Parameters
        ----------
        dims: list of dimensions for the MLP
        activation: activation function
        dropout: dropout probability.
        """
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must have at least two elements")

        layers = []
        for curr_dim, next_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(torch.nn.Linear(curr_dim, next_dim))
            layers.append(torch.nn.Dropout(dropout)),
            layers.append(activation())
        layers.append(torch.nn.Linear(dims[-2], dims[-1]))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
