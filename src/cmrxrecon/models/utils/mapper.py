import torch


class Mapper(torch.nn.Module):
    def __init__(self, classes, eps=0.5):
        """
        Maps a value to a (soft) onehot vector and a scaled value

        Parameters
        ----------
        classes: list of values representing the individual classes
        eps: smoothing parameter
        """
        super().__init__()
        self.register_buffer("classes", torch.tensor(classes), persistent=False)
        self.scale = self.classes.max() - self.classes.min()
        self.shift = self.classes.min()
        self.eps = eps

    def forward(self, x):
        onehot = torch.tanh((1 / ((self.classes - x) ** 2 + self.eps)))
        scaled = torch.atleast_1d((x - self.shift) / self.scale)
        res = torch.cat([onehot, scaled], dim=-1)
        return res

    @property
    def out_dim(self):
        return len(self.classes) + 1
